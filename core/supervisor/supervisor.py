from typing import Optional, Literal
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph_supervisor import create_supervisor
from langgraph.graph import START, END
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.config import OPENAI_API_KEY, logger
from backend.db import get_async_pool
from core.agents.prediction_agent import build_prediction_agent
from core.agents.research_agent import build_research_agent
from core.agents.data_agent import build_data_agent
from core.agents.planning_agent import build_planning_agent
from core.agents.report_agent import build_report_agent
from core.prompts.prompts import SUPERVISOR_SYSTEM_PROMPT_ver3


def initialize_agents(llm, user_request: Optional[str] = None, use_episodic_learning: bool = True):
    """Initialize all agents with optional episodic learning for planning agent."""
    planning_llm = init_chat_model("gpt-4o", model_provider="openai", api_key=OPENAI_API_KEY)
    data_llm = init_chat_model("gpt-5.2", model_provider="openai", api_key=OPENAI_API_KEY)
    research_llm = init_chat_model("gpt-5-mini", model_provider="openai", api_key=OPENAI_API_KEY)
    prediction_llm = init_chat_model("gpt-5-mini", model_provider="openai", api_key=OPENAI_API_KEY)
    report_llm = init_chat_model("gpt-5.2", model_provider="openai", api_key=OPENAI_API_KEY)

    research_agent = build_research_agent(research_llm)
    data_agent = build_data_agent(data_llm)
    prediction_agent = build_prediction_agent(prediction_llm)
    planning_agent = build_planning_agent(planning_llm, user_request, use_episodic_learning)
    report_agent = build_report_agent(report_llm)
    
    return research_agent, data_agent, prediction_agent, planning_agent, report_agent



# Global PostgreSQL state
_postgres_checkpointer = None
_postgres_setup_completed = False
_approval_judge_llm = None


def _get_approval_judge_llm():
    global _approval_judge_llm
    if _approval_judge_llm is None:
        _approval_judge_llm = init_chat_model("gpt-5-nano", model_provider="openai", api_key=OPENAI_API_KEY)
    return _approval_judge_llm


def _judge_plan_feedback(feedback: str) -> Literal["approve", "revise"]:
    """Use a lightweight LLM to classify whether the user approved or requested revisions."""
    if not feedback:
        return "revise"
    llm = _get_approval_judge_llm()
    prompt = (
        "You evaluate a human's feedback on an execution plan.\n"
        "Reply with EXACTLY one word:\n"
        "- APPROVE → the human explicitly authorizes execution immediately.\n"
        "- REVISE → the human asks for changes, more info, or expresses uncertainty.\n"
        "Do not add punctuation or commentary.\n"
        f"Feedback: {feedback}\n"
    )
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response)).strip().lower()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Approval judge failed, defaulting to revise: %s", exc)
        return "revise"
    if content.startswith("approve"):
        return "approve"
    if content.startswith("revise"):
        return "revise"
    logger.info("Approval judge returned unrecognized answer '%s'; defaulting to revise", content)
    return "revise"

async def check_postgres_connection():
    """Debug function to check PostgreSQL connection health."""
    try:
        checkpointer = await get_postgres_checkpointer()
        pool = getattr(checkpointer, "conn", None)
        if pool is None:
            raise RuntimeError("Checkpointer has no connection pool")
        
        async with pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT 1")
                result = await cursor.fetchone()
                logger.info(f"✅ PostgreSQL connection healthy: {result}")
                return True
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection check failed: {e}")
        return False

async def get_postgres_checkpointer():
    """Get or create the global PostgreSQL checkpointer."""
    global _postgres_checkpointer, _postgres_setup_completed

    if _postgres_checkpointer is not None:
        return _postgres_checkpointer

    pool = await get_async_pool()

    checkpointer = AsyncPostgresSaver(pool)

    if not _postgres_setup_completed:
        try:
            await checkpointer.setup()
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in [
                "already exists", "relation", "table", "duplicate"
            ]):
                logger.info("PostgreSQL checkpoint tables already exist")
            else:
                raise
        _postgres_setup_completed = True

    _postgres_checkpointer = checkpointer
    logger.info("PostgreSQL checkpointer initialized")
    return _postgres_checkpointer


def _latest_user_text(state) -> str:
    """Extract the latest user message from the state."""
    msgs = state.get("messages") or []
    # Walk from the end to get the most recent user message
    for m in reversed(msgs):
        # LangChain message objects
        if isinstance(m, HumanMessage):
            return m.content or ""
        # Fallbacks for other LC message types that expose `.type`
        if getattr(m, "type", None) == "human":
            return getattr(m, "content", "") or ""
        # Dict-style messages
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


def route_from_start(state) -> Literal["plan", "skip"]:
    """Route user requests either to planning agent first or directly to supervisor."""
    user_text = _latest_user_text(state)

    # Default to planning if we couldn't find any user text
    if not user_text:
        return "plan"

    # Initialize LLM for routing decision
    llm = init_chat_model("gpt-4o", model_provider="openai", api_key=OPENAI_API_KEY)

    prompt = (
        "You are a router for an agent workflow.\n"
        "If the request is concrete and ready to execute, answer 'skip'.\n"
        "If the request is vague/complex and needs decomposition, answer 'plan'.\n"
        f"Request: {user_text}\n"
        "Answer with exactly one word: skip or plan."
    )

    out = llm.invoke(prompt)
    # llm.invoke returns an AIMessage (LangChain) or a string depending on your wrapper
    out_text = getattr(out, "content", str(out)).strip().lower()

    return "skip" if out_text.startswith("skip") else "plan"


def route_from_planning(state) -> Literal["human_chat", "supervisor"]:
    """Route from planning agent - check if human message contains approval terms."""
    messages = state.get("messages", [])
    
    # Get all human messages in chronological order
    human_messages = []
    for msg in messages:
        human_content = None
        
        # Check different message formats for human messages
        if hasattr(msg, 'type') and msg.type == 'human':
            human_content = msg.content
        elif isinstance(msg, dict) and msg.get('role') == 'user':
            human_content = msg.get('content', '')
        elif hasattr(msg, 'role') and msg.role == 'user':
            human_content = msg.content
        
        if human_content:
            human_messages.append(human_content)
    
    # CRITICAL FIX: Only check messages AFTER the first one (exclude original user request)
    if len(human_messages) <= 1:
        # Only the original request exists, no approval possible yet
        logger.info("Only original request exists, routing to human_chat for plan review")
        return "human_chat"
    
    # Check the most recent human message (excluding the first) for approval terms
    most_recent_feedback = human_messages[-1]
    decision = _judge_plan_feedback(most_recent_feedback)
    if decision == "approve":
        logger.info("Approval judge confirmed plan approval")
        return "supervisor"
    logger.info("Approval judge requested more revisions")
    return "human_chat"



def human_chat_node(state):
    """Handle human-in-the-loop conversation for plan approval."""
    from langgraph.types import interrupt
    
    # Get the plan from state
    messages = state.get("messages", [])
    planning_output = ""
    
    # Extract the latest planning agent output
    for msg in reversed(messages):
        if hasattr(msg, 'name') and msg.name == 'planning_agent':
            planning_output = msg.content
            break
        elif isinstance(msg, dict) and msg.get('name') == 'planning_agent':
            planning_output = msg.get('content', '')
            break
    
    # Interrupt for human input with the current plan
    human_input = interrupt({
        "type": "plan_review",
        "plan": planning_output, 
        "message": "Please review the plan above. You can:\n1. Ask for changes or refinements\n2. Click 'Approve Plan' to proceed with execution"
    })
    
    # Check if human approved the plan or wants to refine it
    if human_input and human_input.lower().strip() == "approved":
        return {"plan_approved": True}
    else:
        # Continue conversation - add human feedback to messages
        if human_input:
            messages = state.get("messages", [])
            messages.append(HumanMessage(content=human_input))
        return {"messages": messages, "plan_approved": False}




async def _create_app_with_checkpointer(checkpointer, user_request: Optional[str] = None, use_episodic_learning: bool = True):
    """Create app with the provided checkpointer."""
    llm = init_chat_model("gpt-5-mini", model_provider="openai", api_key=OPENAI_API_KEY)
    
    # Build agents with episodic learning for planning agent
    research_agent, data_agent, prediction_agent, planning_agent, report_agent = initialize_agents(
        llm, user_request, use_episodic_learning
    )
    
    # Create supervisor with execution agents (planning agent added separately, report agent included but routed to END)
    supervisor_agent = create_supervisor(
        [research_agent, prediction_agent, data_agent, report_agent],
        model=llm,
        output_mode="full_history",
        prompt=SUPERVISOR_SYSTEM_PROMPT_ver3,
        add_handoff_message = True,
        supervisor_name='supervisor'
    )

    # Modify the graph structure to add routing and human-in-the-loop
    # Remove the default START -> supervisor edge
    supervisor_agent.edges.remove(('__start__', 'supervisor'))
    
    # Add planning agent as a separate node
    supervisor_agent.add_node('planning_agent', planning_agent)
    
    # Add human chat node for plan approval
    supervisor_agent.add_node('human_chat', human_chat_node)
    
    # Remove default edge from report_agent back to supervisor (similar to planning_agent)
    supervisor_agent.edges.remove(('report_agent', 'supervisor'))
    
    # Add conditional routing from START
    supervisor_agent.add_conditional_edges(
        START,
        route_from_start,
        {"plan": "planning_agent", "skip": "supervisor"},
    )
    
    # Add conditional routing from planning agent
    supervisor_agent.add_conditional_edges(
        'planning_agent',
        route_from_planning,
        {"human_chat": "human_chat", "supervisor": "supervisor"},
    )
    
    # Add edge from human_chat back to planning_agent for refinements
    supervisor_agent.add_edge('human_chat', 'planning_agent')
    
    # Add edge from report_agent to END (report_agent is managed by supervisor, no direct edge needed)
    supervisor_agent.add_edge('report_agent', END)

    app = supervisor_agent.compile(checkpointer=checkpointer)
    
    if use_episodic_learning and user_request:
        logger.info("Created app with episodic learning enhancement and separate planning node")
    else:
        logger.info("Created app with standard agents and separate planning node")
    
    return app

async def create_app(user_request: Optional[str] = None, use_episodic_learning: bool = True):
    """
    Initialize the LangGraph application with PostgreSQL (Supabase).
    
    Args:
        user_request: Current user request for context-aware planning agent enhancement
        use_episodic_learning: Whether to use episodic learning for planning agent
    """
    # Get the global PostgreSQL checkpointer (with connection pooling)
    checkpointer = await get_postgres_checkpointer()
    
    # Create and return the app with the checkpointer
    return await _create_app_with_checkpointer(checkpointer, user_request, use_episodic_learning)

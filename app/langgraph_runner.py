from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from langchain_core.messages import convert_to_messages
from langgraph.types import Command

from app.app_config import AppRunConfig
from app.config import RECURSION_LIMIT, logger
from backend.utils.output_paths import (
    reset_current_task_id,
    reset_current_user_id,
    set_current_task_id,
    set_current_user_id,
)
from core.supervisor.supervisor import create_app


def _resolve_agent_name(metadata: dict, default: str | None = None) -> str:
    """Best-effort mapping from LangGraph metadata to a display-friendly agent name."""
    node = metadata.get("langgraph_node")
    if node and node not in {"agent"}:
        return node

    triggers = metadata.get("langgraph_triggers") or ()
    for trig in triggers:
        if isinstance(trig, str) and "branch:to:" in trig:
            candidate = trig.split("branch:to:", 1)[-1]
            if candidate and candidate not in {"agent", "__start__", "__end__", "__pregel_pull"}:
                return candidate

    checkpoint = metadata.get("langgraph_checkpoint_ns")
    if isinstance(checkpoint, str) and ":" in checkpoint:
        prefix = checkpoint.split(":", 1)[0]
        if prefix and prefix not in {"agent", "__start__", "__end__", "__pregel_pull"}:
            return prefix

    path = metadata.get("langgraph_path")
    if isinstance(path, (list, tuple)):
        for element in reversed(path):
            if element not in {"__pregel_pull", "__start__", "__end__", "agent"}:
                return element

    return (node or default or "agent")


@asynccontextmanager
async def app_session(app_config: AppRunConfig):
    """Create a LangGraph app for a single async operation."""
    app = await create_app(
        user_request=app_config.user_request,
        use_episodic_learning=app_config.use_episodic_learning,
    )
    yield app


def _is_interrupt_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        keyword in message
        for keyword in ("nodeinterrupt", "interrupt", "interrupted", "human input required")
    )


async def stream_langgraph_events(
    app_config: AppRunConfig,
    stream_input: Any,
    thread_id: str,
    *,
    user_id: Optional[str] = None,
    check_for_interrupts: bool = False,
):
    """Yield LangGraph stream chunks followed by a completion event.

    Token-level streaming is disabled; we forward structured events:
    - ai_message: completed AI message (and inline tool_calls if present)
    - tool_call_start: tool invocation began
    - tool_result: tool invocation completed
    - chunk: checkpoint/state updates
    - complete: finished (payload includes interruption flag + timestamp)
    """
    if not thread_id:
        raise ValueError("No active conversation thread is selected.")

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": RECURSION_LIMIT,
    }

    task_token = set_current_task_id(thread_id)
    user_token = set_current_user_id(user_id)
    try:
        async with app_session(app_config) as app:
            event_iterator = app.astream_events(
                stream_input,
                config=config,
                version="v1",
            )

            async for event in event_iterator:
                event_type = event.get("event")
                data = event.get("data") or {}
                metadata = event.get("metadata") or {}
                agent_name = _resolve_agent_name(metadata, event.get("name"))

                if event_type == "on_chat_model_end":
                    message = data.get("output")
                    if message:
                        tool_calls = getattr(message, "tool_calls", None) or (
                            message.get("tool_calls") if isinstance(message, dict) else None
                        )
                        yield ("ai_message", {"agent": agent_name, "message": message, "tool_calls": tool_calls})

                elif event_type == "on_tool_start":
                    call_args = data.get("input")
                    call_id = None
                    if isinstance(call_args, dict):
                        call_id = call_args.get("id")
                    else:
                        call_id = getattr(call_args, "id", None)
                    call_name = event.get("name")
                    if call_id and call_name:
                        yield (
                            "tool_call_start",
                            {
                                "agent": agent_name,
                                "call": {
                                    "id": str(call_id) if call_id else None,
                                    "name": call_name,
                                    "args": call_args,
                                },
                            },
                        )

                elif event_type == "on_tool_end":
                    result = data.get("output")
                    call = data.get("input")
                    call_id = None
                    if isinstance(result, dict):
                        call_id = result.get("tool_call_id")
                    if call_id is None:
                        if isinstance(call, dict):
                            call_id = call.get("id")
                        else:
                            call_id = getattr(call, "id", None)
                    tool_name = event.get("name")
                    if result is not None:
                        normalized_result = result
                        if not isinstance(result, dict):
                            normalized_result = {"name": tool_name, "content": result}
                        else:
                            normalized_result = dict(result)
                            if tool_name and "name" not in normalized_result:
                                normalized_result["name"] = tool_name
                        payload = {"agent": agent_name, "result": normalized_result}
                        if call_id:
                            payload["call_id"] = str(call_id)
                        yield ("tool_result", payload)

                elif event_type == "on_chain_stream":
                    chunk = data.get("chunk")
                    if not chunk:
                        continue
                    if metadata.get("langgraph_node"):
                        payload = chunk
                        if not isinstance(chunk, dict) or "messages" not in chunk:
                            payload = {"messages": [chunk]}
                        yield ("chunk", {agent_name: payload})
                    else:
                        yield ("chunk", chunk)

            interrupted = False
            if check_for_interrupts:
                try:
                    current_state = await app.aget_state(config)
                    if hasattr(current_state, "next") and current_state.next:
                        if "human_chat" in current_state.next:
                            interrupted = True
                except Exception as state_error:  # pragma: no cover - defensive
                    logger.warning("Could not check execution state: %s", state_error)

            yield ("complete", {"interrupted": interrupted, "completed_at": time.time()})

    except Exception as exc:
        if check_for_interrupts and _is_interrupt_exception(exc):
            yield ("complete", {"interrupted": True, "completed_at": time.time()})
            return
        raise
    finally:
        reset_current_task_id(task_token)
        reset_current_user_id(user_token)


def build_stream_input(user_message: str, *, resume: bool = False) -> Any:
    """Utility for constructing graph input compatible with the Gradio UI."""
    if resume:
        return Command(resume=user_message)
    return {"messages": convert_to_messages([user_message])}

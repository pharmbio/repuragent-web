# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Repuragent — a multi-agent system for drug repurposing, deployed as a multi-user web app:
Gradio mounted on FastAPI, PostgreSQL for conversation checkpoints and accounts, a scoped
filesystem per conversation. A planning agent proposes a workflow, **the user approves it**,
and a supervisor executes it by delegating to three specialists (research, prediction, data).
Built on **LangChain 1.x / LangGraph 1.x**; the LLM backend is OpenAI.

A separate repo (`pharmbio/repuragent`) is the single-user local variant. Docs:
https://repuragent.readthedocs.io/

## Commands

```bash
python main.py                  # serve on GRADIO_SERVER_PORT (default 7860)
python -m tests.run_all         # the whole suite; no pytest needed
python -m tests.test_graph      # one module (each is runnable on its own)
pip install -r requirements.txt # needs a JRE too, for the CPSign ADMET models

# Rebuild the SOP index after changing the PDFs under $DATA_ROOT/SOP.
# The built index is committed, so this is only needed deliberately.
python -m backend.sop_rag.sop_indexer

docker build -t repuragent .
docker run -p 7860:7860 --env-file .env -v repuragent-data:/home/repuragent/app/persistence repuragent
```

`.env` (see `.env.example`): `OPENAI_API_KEY` and `DATABASE_URL` are required; everything
else has a default. **The database schema is applied at startup** from
`backend/auth/migrations/initialise_db.sql`, which is idempotent — no manual `psql` step.

There is no linter config. `pytest` is not installed in the development environment, which
is why `tests/harness.py` exists; tests are still plain `test_*` functions so `pytest tests`
works wherever it is available.

## Version constraint that is not optional

The v0.3 LangChain line **cannot** run this code: `langchain.retrievers`,
`langchain.storage` and `langchain.schema` no longer exist (use `langchain_classic.*` /
`langchain_core.*`), `pre_model_hook` was replaced by agent middleware, and
`langgraph.prebuilt.create_react_agent` is deprecated in favour of
`langchain.agents.create_agent`. Versions are pinned in `requirements.txt`.

## Architecture

### The graph (`core/agents/agentic_system.py`)

```
START → task_classifier
  ├ complex   → planning_agent → human_chat ⇄ planning_agent
  │                                   └(approved)→ approval_ack → plan_init
  ├ simple    → plan_init
  ├ follow_up → plan_init
  └ meta_query→ report_agent_meta → context_summary_meta → END

plan_init → execution → plan_finalize → report_agent_{complex|simple}
          → context_summary_{complex|simple} → END

execution (nested subgraph):
  START → supervisor ⇄ {research_agent, prediction_agent, data_agent}
```

`task_classifier` routes on **dependence, not size**, and is given the conversation goal
plus the previous exchange — "now rank those by hERG risk" cannot be classified from the
bare message.

**Specialists are context-isolated; only the supervisor holds the conversation.** A
specialist's model is given the brief the supervisor wrote for it plus its own working
messages — no conversation, no plan, no other specialist's work
(`build_specialist_input_messages`). The supervisor keeps the conversation but sees each
specialist's *reported result* in place of its tool traffic
(`collapse_specialist_spans`). The report agent keeps the full view, because it is the
one that has to cite evidence.

That makes the brief the entire input, so `transfer_to_<agent>` asks for it in parts
rather than as one free-text `task`: `objective`, `inputs`, `artifacts`, `constraints`,
`expected_output`, `context`.

**`artifacts` is the field that earns its keep.** A specialist does receive the pinned
artifact ledger, but that is a directory listing — ten paths with byte counts and nothing
saying which is the candidate table, which is the pathway extract, or how they join. Each
entry is `<full path> — <what it contains>`, taken from the ledger by the supervisor, which
is the difference between a specialist working and a specialist guessing.

**The level is a two-sided constraint** and both sides are addressed:

- Too thin and the specialist guesses. `validate_brief` refuses the mechanical failures —
  dangling references ("the file from the previous step"), an artifact named without saying
  what it holds, an absolute path that does not exist, a one-word objective, a missing
  expected output. The refusal comes back as the tool result, so the supervisor rewrites and
  retries, exactly like `plan_update` on a bad step number.
- Too prescriptive and the specialist becomes a typist: its domain knowledge goes unused and
  a mistake in the supervisor's method survives unchallenged. That half cannot be validated,
  so it is taught — a worked *too thin / too prescriptive / right* example in
  `SUPERVISOR_SYSTEM_PROMPT`, and "**the method is yours**" in `ISOLATION_NOTICE`, which also
  tells a specialist to override a prescribed method that is wrong for the data and say so.

**`parse_docstring=True` on the handoff tool is load-bearing.** With `description=` alone,
LangChain leaves every argument description empty — so all of the per-field guidance above
silently never reaches the model. `tests/test_isolation.py` asserts the descriptions are
present, along with what the scripted models were actually handed.

Each specialist prompt also ends with a `WHAT TO HAND BACK` section, because its reply is
now the only thing the supervisor reads.

**The supervisor and its specialists are a nested subgraph, and the supervisor has no
outgoing edge.** This is the one piece of graph shape you must not casually change. A
handoff tool returns `Command(goto=<specialist>, graph=Command.PARENT)`, which *adds* a
task rather than replacing the node's edges — so a static `supervisor → plan_finalize` edge
fires *in parallel* with the handoff, and `plan_finalize` runs before any specialist has
done anything (observably: two nodes writing `plan_progress` in one superstep, which
LangGraph rejects with `InvalidUpdateError`). With no outgoing edge the subgraph simply
ends when the supervisor stops delegating, and the parent's single `execution →
plan_finalize` edge takes over.

**The other half of that `Command`: it abandons the supervisor's own graph, so the
delegating `AIMessage` has to be carried up by hand.** `graph=Command.PARENT` means the
parent receives the `Command`'s `update` and *nothing else* — whatever the supervisor's
inner agent graph accumulated this turn, including the `AIMessage` holding the
`transfer_to_<agent>` `tool_call`, is left behind in the graph that was just abandoned.
An update of only the rendered brief therefore writes a `ToolMessage` into the parent
whose `tool_call` is nowhere the parent can see, and the run dies on the supervisor's
*next* model call — which reads as the specialist finishing and the work never coming
back:

```
BadRequestError: 400 - Invalid parameter: messages with role 'tool' must be a
response to a preceeding message with 'tool_calls'.   (param: messages.[6].role)
```

`handoff_messages` is what prevents it: the update carries the supervisor's whole inner
message list plus the brief. `add_messages` keys on id, so the parent's own copies are
replaced by themselves and only this turn's additions are appended, in order. Sending
just the tail would mean guessing where the parent's copy ended, and guessing short
drops a pair. A **parallel call in the delegating turn** — `plan_update` beside the
handoff — is the mirror-image failure: the sibling's `ToolMessage` dies with the
abandoned graph, so the propagated `AIMessage` keeps only the call the brief answers and
the supervisor makes the other again.

**Two handoffs in one turn cannot both be delivered**, which is what `superseded_handoff`
is for. Each returns its own `Command` carrying its own copy of the same `AIMessage`, and
both copies share its id — so `add_messages` keeps whichever lands last and the other
specialist's brief is left answering a call that is no longer there. Two specialists
could not run concurrently here anyway: they share the message list and the plan file,
and simultaneous `plan_progress` writes are exactly what LangGraph rejects. So the first
handoff *as the model wrote them* proceeds — written order, not tool-node execution
order, so the choice is deterministic — and the rest come back with the reason, like any
other refused brief. Their steps stay unresolved in `plan.md`, so the supervisor
delegates them on its next turn.

Belt and braces: `repair_tool_pairing` (`core/agents/context.py`) is the last thing
every message-list builder does. Every view is a *slice* — spans collapsed, turns
anchored, a specialist's window opened after its brief — and a slice can cut a pair in
half from either side, so it drops orphaned `ToolMessage`s and strips unanswered
`tool_call`s rather than letting a provider reject the request. It is also what lets a
conversation whose checkpoint was poisoned before the fix carry on being used. A
scripted test model accepts any message list, which is why nothing caught this:
`tests/test_tool_pairing.py` applies the provider's own rules, via
`tests.harness.tool_pairing_problems`, to every list the middleware assembles — and
`test_graph.py`'s full run now does the same.

`plan_init` and `plan_finalize` contain **no LLM call**. Approval is decided by a
structured-output judge (`_judge_plan_feedback`) that also extracts **conditional-approval
constraints** — "go ahead, but only phase 3 drugs" is an approval carrying a constraint,
which is stored in state, written into `plan.md`, and pinned into the executor's context.
It defaults to `revise` on any failure.

Every agent is built with `state_schema=AgentGraphState`. Without it LangGraph filters the
parent state down to the plain `AgentState` and `user_id`, `conversation_id`,
`approved_plan` and the plan pointer never reach the middleware.

### Progress is a file (`backend/utils/plan_store.py`, `core/tools/plan_tools.py`)

`plan.md` lives in the conversation's output scope, one document per conversation, each run
appended as a `## Run N` section — so it is the contract the supervisor executes, the live
progress the UI panel renders, *and* a durable work log a follow-up can read.

1. The planning agent emits the canonical breakdown (`[1] **Title**` + `**Agent:**` +
   `**Details:**`). **`PLANNING_SYSTEM_PROMPT`'s format is parsed by
   `parse_plan_steps`** — change one and the other must follow, or plans degrade to a
   single catch-all step (deliberately: a mis-parsed plan must not block the science).
2. `plan_init` parses it and writes the file. It runs **after** approval, so revising a
   plan several times leaves no spurious runs.
3. The supervisor reads with `plan_status` and records with `plan_update(step, status,
   note)`, which validates the step number and status against the file, timestamps it,
   rewrites atomically, and returns the refreshed ledger. **That return value is the
   progress display** — do not reintroduce prose tracking beside it.
4. `plan_finalize` reads back what happened and records the outcome. Note `blocked` and
   `skipped` are terminal, so `summarize_outcome` reserves "All N steps completed" for
   actual completion rather than calling a blocked run resolved.

The plan *text* is not pinned into context — only the path and a progress line;
`plan_status` returns the authoritative copy on demand.

This replaced a prompt-only `📋 BREAKDOWN / ⏳ CURRENT / ✓ COMPLETED` block the supervisor
had to re-emit before every delegation, which drifted and could not be checked.

**`plan_status` / `plan_update` read their scope from graph state** via `InjectedState`,
falling back to contextvars. Other tools read contextvars only, and when the two disagree
the plan is written in one place and looked for in another — which surfaces as
`plan_update` reporting "No execution plan exists for this conversation yet" while
`plan.md` sits there correctly written.

### Context (`core/agents/context.py`)

An async `@wrap_model_call` middleware, built per role, overrides `system_message` (role
prompt + posture + pinned block) and `messages`. Three views exist: `isolated=True` for a
specialist, `collapse_delegations=True` for the supervisor, and the plain compressed view
for the planner and the report agents. It **must** be async — `astream` raises
`NotImplementedError` on a sync-only `wrap_model_call`, and every run here is streamed.

The message list is turn-anchored: pinned block → compressed summary → the last
`CONTEXT_KEEP_TURNS` completed exchanges **verbatim** → the live turn with tool traffic
bounded *in place* (dropping a `ToolMessage` would orphan its `tool_call` and fail history
validation). Turn boundaries come from a marker on the opening `HumanMessage`, because plan
review writes its own `HumanMessage`s into the same transcript.

The three supervisor **postures** (plan-driven / free / follow-up) are injected here from
`state["task_category"]` rather than being three separate agents, since its tools and
specialists are identical in all three cases. Report formats genuinely differ, so those are
three nodes — and `REPORT_NODES` styling keys off the node name.

**Episodic memory is injected here too**, which is what lets the compiled graph be cached:
the previous build recompiled the whole app — five chat models, five agents — on *every user
message*, purely so the planner's prompt string could change.

### Tools (`core/tools/`)

Agent-facing `@tool` functions live in `core/tools/`; pure API and domain helpers stay under
`backend/`. So `core/tools/` is the complete inventory of what the system can do, and
`tests/test_imports.py::test_tool_inventory_is_complete` pins that inventory.

- `plan_tools` — `plan_status`, `plan_update`
- `python_executor` — the data agent's engine; see below
- `read_files` — bounded reads, with a **preview envelope** above
  `READ_FILES_PREVIEW_THRESHOLD_CHARS` so a 40 000-row candidate table does not land in the
  transcript. An explicit `offset`/`limit` range is always returned verbatim.
- `research_tools` — LitSense literature search, SOP retrieval
- `knowledge_graph_tools` — OpenTargets/ChEMBL/UniProt/Reactome/KEGG traversal over
  `backend/kgg/api_utils.py`. These run **in order**: `search_disease_id` →
  `create_knowledge_graph` (pickles `kg_<ID>.pkl`) → `extract_*_from_kg` (reads that
  pickle) → `getDrugsfor*`.
- `prediction_tools` — the CPSign panel, driven from the `CPSIGN_MODELS` table
- `chemical_tools` — ChEMBL/PubChem annotation

**Reading ADMET output correctly.** The classifiers are conformal: `{0}` / `{1}` /
`{0, 1}` maps to `0` / `1` / **`0.5`, which means both labels are plausible at the
configured confidence** — an abstention, not a probability, and not something to average.
CPSign also **silently omits** structures it cannot featurize (small fragments especially),
so `_predict` compares row counts and appends a warning naming what is missing; join results
on `smiles`, never on row order.

### The sandbox

`python_executor` runs code through `backend/utils/local_python_executor.py` (an
AST-walking interpreter adapted from smolagents), not `exec`. Three properties matter more
than the sandbox itself:

- **State is per `(user_id, conversation_id)`**, an LRU registry with a per-session lock.
  It used to be one namespace for the whole process, so concurrent users interleaved in it
  and one `reset_python_state` wiped everybody's variables.
- **Writes are scoped.** Every call is injected with `output_root`,
  `prepare_output_path`, `ensure_output_dir` and a sandboxed `open`;
  `resolve_output_folder` clamps every escape attempt back to the scope root. Tool code
  must go through it rather than building paths itself.
- **Figures are rescued.** The server is headless, so `plt.show()` is a no-op and an
  unsaved figure — often the deliverable — is lost. `backend/utils/figure_capture.py` saves
  it into the scope and reports the path.

Every call has a wall-clock budget (`PYTHON_EXEC_TIMEOUT_SECONDS`) because LangChain runs
sync tools in a thread pool: cancelling a run does not kill the thread. On expiry the
session is dropped. An undefined name **raises with suggestions** rather than resolving to
the closest-named variable, which upstream did — quietly turning `herg_scores` into
`hergs_scores`.

### Stop (`app/run_controller.py`, `backend/utils/cancellation.py`)

Two mechanisms, because one is not enough:

1. the run loop breaks and `stream.aclose()` abandons the graph mid-run — the checkpoint is
   left mid-run deliberately, so the next message continues from there;
2. a per-conversation cancellation flag that long tool loops check. Both
   `knowledge_graph_tools` and `api_utils` already wrap every long loop in `tqdm`, so
   `cancellable_tqdm` is imported *as* `tqdm` there and every one of those loops becomes a
   checkpoint without touching a loop body. `python_executor` gets the same flag as
   `cancel_event`, and the CPSign subprocess is polled rather than waited on.

Without (2), "stopped" meant the browser went quiet while a knowledge-graph traversal kept
running for another ten minutes.

Stopping appends a visible "Run stopped" notice: resolving the spinner alone is
indistinguishable from finishing.

### Persistence and scoping

Everything is keyed by `(user_id, thread_id)`, where a thread id is `"{user_id}:{uuid4}"`:

- **Conversation state** — `AsyncPostgresSaver` (`backend/db/checkpointer.py`) over a
  resilient auto-reconnecting pool (`backend/db/pool.py`), both process-wide singletons.
- **Filesystem** — uploads in `DATA_ROOT/<user>/<thread>/`, outputs in
  `RESULTS_ROOT/<user>/<thread>/`. The scope travels in **contextvars**, and
  `app/run_controller.py::build_conversation_context` pins it into an explicit
  `contextvars.Context` that every task of the run is created in. This is load-bearing:
  `asyncio.Task` copies the ambient context *at creation* and the run loop wraps each
  `__anext__` in a Task to race the file-refresh tick, so a scope merely set inside the
  streaming generator lives only in the first task's copy.
- **Rendered timeline** — snapshotted as JSON into `user_threads.ui_timeline`, so reopening
  a conversation restores the exact tool entries without replaying graph state.
- **Auth** — argon2 + pepper, sessions in Postgres, email verification and password reset
  as plain FastAPI pages (`app/auth_routes.py`), since the user arrives from an email link
  with no Gradio session.

`persistence/` is **committed on purpose**: it ships the demo conversations
(`persistence/memory/demo_threads.json` + their results directories), the prebuilt SOP
index, and the episodic-memory store. Do not add it to `.gitignore`.

### UI (`app/`, `app/ui/`)

`app/gradio_app.py` is a 76-line entry point; it replaced a 2 733-line module that held the
widget tree, the CSS, the run loop, the auth routes, the download signing and the timeline
renderer together.

| Module | Responsibility |
| --- | --- |
| `app/langgraph_runner.py` | `astream(["messages","updates"], subgraphs=True)` → `chunk` / `token` / `complete`; `read_pending_approval`; compiled-app cache |
| `app/run_controller.py` | one run: per-thread lock, pinned context, tick, stop, detached buffer |
| `app/session.py` | sign-in, conversation switching, uploads, episode extraction |
| `app/conversation_store.py` · `timeline_store.py` | threads and rendered timelines |
| `app/files.py` · `downloads.py` · `auth_routes.py` · `demo_threads.py` | uploads, signed links, email flows, read-only demos |
| `app/ui/layout.py` | the widget tree and every event wiring |
| `app/ui/projection.py` | the single `render(state)` → output tuple |
| `app/ui/chat_timeline.py` · `tool_display.py` | three visual tiers; one line per tool call |
| `app/ui/approval.py` · `progress_panel.py` · `conversation_panel.py` | approval gate, live plan, sidebar |
| `app/ui/theme.py` · `scripts.py` · `assets.py` | palette + stylesheet, injected JS, inlined images |

A specialist still receives the pinned **output scope and artifact ledger** even when
isolated — it has to know where to write, and briefs refer to files by path. It does not
receive the conversation goal or the plan: that framing belongs in the brief, and leaving
it out is what stops a thin brief from being silently rescued.

Things that will bite if you change them:

- **Why not `astream_events`.** It flattens the graph, so every event inside an agent is
  labelled `model`/`tools` and the producing node had to be *inferred*. `subgraphs=True`
  gives the namespace, making attribution a lookup — `resolve_agent_name` prefers the node
  name and falls back to the **deepest** namespace entry, because with the nested
  `execution` subgraph the shallowest entry would only ever say `execution`.
- **Id de-duplication in the timeline is load-bearing.** A node wrapping a subgraph
  re-emits every message that subgraph produced; `execution` re-emits the whole
  conversation each time the supervisor commits. And **suppressing a node is not enough —
  its message ids must be recorded** (`_suppress_messages`), or the skipped message renders
  when the aggregate arrives under a name that is not suppressed. That is how `plan_init`'s
  ledger once landed inside the report block.
- **Whether a thread is paused is always re-read from the graph**, never trusted from
  session state. Sending plain input to an interrupted thread makes LangGraph restart from
  `START` and re-plan, which trapped users in an approval loop.
- **Side panels are only sent when they change** (`gr.skip()`), download tokens are minted
  in time buckets, thumbnails are cached by `(path, mtime, size)`, and `render(live=True)`
  skips both panels for token frames. Otherwise `gr.HTML` swaps its DOM on every streamed
  event and the file list cannot be scrolled during a run.
- **The report headings are load-bearing.** The report prompts emit `# Response Summary` →
  `## Answer` / `## Evidence` / `## Open Issues`, and `.agent-message-section--report` is
  built around exactly those. Keep the prompt and the CSS in step.
- **`render(clear_input=True)` only on the yield that accepts a submission.** A blanket
  `gr.update(value="")` on every yield wiped out text typed while the agents worked.
- **Type in the transcript is a fight with gradio's own chatbot CSS, and it has to be
  won per element.** A finished agent block is a gradio "thought", whose stylesheet
  contains `.content * { font-size: var(--text-sm) }` — a *universal descendant*
  selector at 12px. A size set on a container of ours therefore reached the paragraphs
  we name and nothing else, so a plan rendered 16.6px prose with 12px `strong`, `code`,
  `li` and tool lines. `#chatbot-panel` now redefines `--text-sm` (gradio's own rule then
  resolves to our compact tier) *and* every element states its size and family, because
  `#chatbot-panel .prose p` outranks anything inherited. Sizes come from the six-tier
  scale in `:root`; `tests/test_ui.py::test_the_stylesheet_survives_gradios_prefixer`
  pins that, plus two traps worth knowing: gradio rewrites the arguments inside `:is()`
  when it prefixes selectors (which silently inflates that rule's specificity), and a
  nested `/* */` truncates a comment so the prose after it is parsed as CSS and eats the
  next rules.
- **`data-testid` on a chat message is `user` / `bot`.** Several rules said `assistant`
  and had never matched anything, which is why the user's own message rendered in the
  editorial serif that was meant only for agent prose.
- **`markdown-it` runs the strict `commonmark` preset, so `table` is enabled explicitly.**
  Tables are a GFM extension: without that call a report's evidence table rendered as raw
  pipes while `.agent-message-section--report table` matched nothing.

## Tests (`tests/`)

`python -m tests.run_all`. `tests/fakes.py` provides a scripted tool-calling chat model and
`tests/harness.py` provides temporary-root and scope fixtures, so the suite runs offline
with no API key and no database.

`tests/test_tool_pairing.py` guards the one failure mode a scripted model cannot see:
`tests.harness.tool_pairing_problems` applies the chat API's own two rules — a `tool`
message must answer an earlier `tool_call`, and every `tool_call` must be answered — to
each message list the middleware assembles, so a view that only a real provider would
reject fails here instead. Keep `assert_every_model_call_is_valid` in any new end-to-end
test that delegates.

`tests/test_graph.py` is the one that matters most: it drives a full complex run
(classify → plan → interrupt → approve with a constraint → delegate to two specialists →
`plan_update` per step → finalize → report) and asserts the node order **and** the contents
of `plan.md`. It also covers the simple, follow-up-shaped and meta routes, a blocked step,
and that a revision loop creates no spurious runs.

## Gotchas

- **Model ids** live only in `app/config.py`, one per role. Nothing else names a model.
- **Paths are absolute.** The CPSign jar comes from `CPSIGN_JAR` and images resolve against
  `REPO_ROOT`, so the app no longer has to be started from the repository root.
- `smiles_csv` writes into the conversation's scope. It used to write one shared
  `DATA_ROOT/modelling_data.csv`, so simultaneous users overwrote each other's model input.
- **Gradio telemetry is off** by default (`GRADIO_ANALYTICS_ENABLED`), because this
  deployment handles unpublished research data.
- The knowledge-graph tools are slow, rate-limited network calls; expect partial results and
  retries, and reuse an existing `kg_*.pkl` rather than rebuilding.
- `REFACTOR_PLAN.md` records why this structure is the way it is, including the design
  alternatives that were tried and rejected.

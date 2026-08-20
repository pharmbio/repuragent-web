# Repuragent - An AI Scientist for Drug Repurposing

## Version announcement
This version is a web-app version of the RepurAgent, used to deploy the web app to multiple users. Access https://repuragent.serve.scilifelab.se to use the agent.
For a local app setup that runs entirely on your local machine, please refer to [Repuragent Local](https://github.com/pharmbio/repuragent).

Please read [Repuragent documentation](https://repuragent.readthedocs.io/) for more details.

## Overview

Drug repurposing offers an efficient strategy to accelerate therapeutic discovery by identifying new indications for existing drugs. However, the process remains hindered by the heterogeneity of biological and chemical data and the difficulty of forming early, evidence-based hypotheses about candidate drugs, targets, and clinical endpoints. We introduce Repuragent (Drug Repurposing Agentic System), a proof-of-concept multi-agent framework designed to autonomously plan, execute, and refine data-driven repurposing workflows under human-in-the-loop supervision. The system integrates autonomous research, data extraction, knowledge graph (KG) construction, and analytical reasoning with an adaptive long-term memory mechanism that improves the system over time.

<div align="center">
  <img src="https://raw.githubusercontent.com/pharmbio/repuragent/main/images/agent_architecture.png" width="500">
</div>


### How a task runs

```
your request → task classifier ─┬─ complex   → planning agent → YOUR APPROVAL ─┐
                                ├─ simple    ────────────────────────────────► │
                                ├─ follow-up ────────────────────────────────► │
                                └─ meta      → answer about the assistant      │
                                                                               ▼
                                            plan.md written  →  supervisor executes it
                                                                  ├─ research agent
                                                                  ├─ prediction agent
                                                                  └─ data agent
                                                                       ▼
                                            plan.md reconciled  →  report
```

A **complex** request is planned first and executed only after you approve the plan —
including approving it with conditions ("go ahead, but only phase 3 drugs"), which are
recorded and override the matching steps. A **short** request skips planning entirely,
and a **follow-up** continues from work already done without asking for approval again.

### One agent holds the context

Each specialist receives a written brief and works from that alone: it cannot see the
conversation, the plan, or another specialist's output. The brief carries the objective, the
input values, **the files earlier steps produced and what each one holds**, the constraints
for this step, and what to hand back. The supervisor holds the context and is responsible
for making the brief complete — a brief that refers to "the file from the previous step" is
refused before it reaches anyone.

The brief states the outcome, not the procedure: the specialist chooses its own method,
because it is the one that knows the tools. That keeps each agent's attention on its own
step, makes a delegation reproducible, and keeps the supervisor's context free of tool
payloads it does not need to read.

### Progress is a file, not a promise

On approval the plan is written to `plan.md` in your conversation's folder, one section
per run. The supervisor reads that file and records each step's outcome in it as the
work completes; the panel under the transcript renders the same file live. So what you
see is what actually happened — the structure, validation and accounting are code, and
the agent supplies only which step reached which status. The file is also a durable
work log, which is how a follow-up knows what has already been done.

### The agents

- **Planning agent** — decomposes the request into an executable breakdown, using
  standard operating procedures, literature, and precedent from earlier successful
  runs (adaptive episodic memory).
- **Supervisor** — writes a task brief for each step, delegates it to the specialist that
  can do it, checks what comes back, and keeps `plan.md` current. It is the only agent
  that holds the whole conversation.
- **Research agent** — literature search, SOP retrieval, disease-identifier
  resolution, knowledge-graph construction from OpenTargets, ChEMBL, UniProt,
  Reactome, KEGG and GWAS data, and mining that graph for drugs, proteins, pathways
  and mechanisms.
- **Prediction agent** — pre-trained CPSign conformal ADMET models (CYP3A4, CYP2C19,
  CYP2D6, CYP1A2, CYP2C9, hERG, Ames, P-gp, PAMPA, BBB, solubility, lipophilicity)
  and new-indication prediction.
- **Data agent** — Python for inspecting uploads, combining tables, scoring and
  ranking candidates, statistics and figures.
- **Report agent** — the final answer, tied to the evidence the run produced.

### Memory

- **Episodic memory** — how successful tasks were decomposed, retrieved as precedent
  when planning a similar one. Recording an episode is deliberate: press *Remember
  this plan*.
- **Conversation state** — checkpointed in PostgreSQL, so a conversation survives a
  restart and a plan can wait at its approval gate indefinitely.
- **SOP retrieval** — a prebuilt index over regulatory guidance and protocol
  documents, so procedural claims are grounded in the wording of the source.

## Running it

```bash
pip install -r requirements.txt      # needs Java for the CPSign ADMET models
cp .env.example .env                 # then fill in OPENAI_API_KEY and DATABASE_URL
python main.py                       # http://localhost:7860
python -m tests.run_all              # the test suite (no pytest required)
```

Or with Docker:

```bash
docker build -t repuragent .
docker run -p 7860:7860 --env-file .env -v repuragent-data:/home/repuragent/app/persistence repuragent
```

Requires LangChain 1.x / LangGraph 1.x — see `requirements.txt`. The database schema is
created on first start.

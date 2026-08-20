# Repuragent - An AI Scientist for Drug Repurposing

## Version announcement
Version 2 keeps the same core system as version 1, with a refactored UI/UX that improves the experience of interacting with the system and waiting for tasks to complete.

Main updates:
- Supervisor: Deterministic plan monitoring: the supervisor writes and updates the plan in an external plan.md file. This also makes task resumption smoother and more reliable.
- UI: Added a persistent plan monitoring panel so users can see which step they're on while a task is still running.
- Agent interaction: The supervisor now holds persistent context for the entire task, while sub-agents receive only a task briefing. Each briefing contains the objective and all artifacts produced so far.

## Overview

Drug repurposing offers an efficient strategy to accelerate therapeutic discovery by identifying new indications for existing drugs. However, the process remains hindered by the heterogeneity of biological and chemical data and the difficulty of forming early, evidence-based hypotheses about candidate drugs, targets, and clinical endpoints. We introduce Repuragent (Drug Repurposing Agentic System), a proof-of-concept multi-agent framework designed to autonomously plan, execute, and refine data-driven repurposing workflows under human-in-the-loop supervision. The system integrates autonomous research, data extraction, knowledge graph (KG) construction, and analytical reasoning with an adaptive long-term memory mechanism that improves the system over time.

<div align="center">
  <img src="https://raw.githubusercontent.com/pharmbio/repuragent/main/images/agent_architecture.png" width="500">
</div>


### The agents

- **Planning agent:** decomposes the request into an executable breakdown, using
  standard operating procedures, literature, and precedent from earlier successful
  runs (adaptive episodic memory).
- **Supervisor:** writes a task brief for each step, delegates it to the specialist that
  can do it, checks what comes back, and keeps `plan.md` current. It is the only agent
  that holds the whole conversation.
- **Research agent:** literature search, SOP retrieval, disease-identifier
  resolution, knowledge-graph construction from OpenTargets, ChEMBL, UniProt,
  Reactome, KEGG and GWAS data, and mining that graph for drugs, proteins, pathways
  and mechanisms.
- **Prediction agent:** pre-trained CPSign conformal ADMET models (CYP3A4, CYP2C19,
  CYP2D6, CYP1A2, CYP2C9, hERG, Ames, P-gp, PAMPA, BBB, solubility, lipophilicity)
  and new-indication prediction.
- **Data agent:** Python for inspecting uploads, combining tables, scoring and
  ranking candidates, statistics and figures.
- **Report agent:** the final answer, tied to the evidence the run produced.

### Memory

- **Episodic memory** — how successful tasks were decomposed, retrieved as precedent
  when planning a similar one. Recording an episode is deliberate: press *Remember
  this plan*.
- **Conversation state** — checkpointed in PostgreSQL, so a conversation survives a
  restart and a plan can wait at its approval gate indefinitely.
- **SOP retrieval** — a prebuilt index over regulatory guidance and protocol
  documents, so procedural claims are grounded in the wording of the source.

## Running it with Docker

First, you need to set up .env file, then:

```bash
docker build --platform linux/amd64 -t repuragent:trial .
docker run -p 7860:7860 --env-file .env repuragent:trial
```

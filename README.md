# SNN Learning Lab

Learn **Spiking Neural Networks** by building → running → visualizing → interpreting → iterating, one experiment at a time.

## Quick start

Paste **[`AGENT_PROMPT.md`](AGENT_PROMPT.md)** into your AI coding/research agent. That file carries the whole lab workflow (the 10-step experiment loop) so you can start immediately with **Experiment 1 — a single LIF neuron** and let the agent drive the rest.

## What we're after

Three guiding questions:

1. **Computation** — how does an SNN compute with spikes and time?
2. **Expressivity** — what functions can SNNs represent, versus ordinary networks?
3. **Dynamics** — how do recurrent spiking nets evolve over time, and how stable are they?

## How the work flows

Every experiment follows a fixed loop: **Question → Hypothesis → Build → Run → Visualize → Read the graph → Insight → Math → Challenge → Decide.** This keeps each step small (~30–90 min), visual-first, and always driven by what the last experiment revealed.

## Repo layout

| Path | What it is |
|:---|:---|
| `AGENT_PROMPT.md` | The paste-into-agent execution prompt |
| `state-of-the-field/` | Reference — the 10 landmark papers and where the field is headed |
| `requirements.txt` | Minimal Python deps (numpy, matplotlib) |

Experiments get added as scripts (numbered by experiment) as we go — no pre-built roadmap.

## First step

Ask the agent to run **Experiment 1 — Single LIF neuron** and deliver a plot of membrane potential and spikes over time.

# SNN Learning Lab

Recently I came across some research on **spiking neural networks** and got genuinely curious about how to build on it. I had no background in them - nothing to lean on - so my first question was simply: *what's the best way to actually learn and use this?*

The answer I kept coming back to was AI. Plenty of universities now run AI as a **lab partner** alongside their students and researchers - helping people learn by doing, not just by reading. That's what I wanted to build here. I'm naturally curious, open to trying new things, and this felt like the most honest way to actually pick the field up.

To be clear, the AI isn't doing the work for me. I frame each question, predict what should happen, and explain the graphs we get. The AI helps me set up and run the experiments and catches where my reasoning is off - which is how it should be used, as a partner that sharpens my own understanding rather than replacing it. I am still not quite there yet with using AI efficiently but do star this repo as I will keep on improving.

Learn **Spiking Neural Networks** by building → running → visualizing → interpreting → iterating, one experiment at a time.

## Quick start

Paste **[`AGENT_PROMPT.md`](docs/AGENT_PROMPT.md)** into your AI coding/research agent. That file carries the whole lab workflow (the 10-step experiment loop) so you can start immediately with **Experiment 1 - a single LIF neuron** and let the agent drive the rest.
New to using an AI agent as a learning partner? See **[`Using AI as a Research Lab Partner`](docs/ai-lab-partner.md)** - how to start fast and actually learn, not just get answers.

## What we're after

Three guiding questions:

1. **Computation** - how does an SNN compute with spikes and time?
2. **Expressivity** - what functions can SNNs represent, versus ordinary networks?
3. **Dynamics** - how do recurrent spiking nets evolve over time, and how stable are they?

## How the work flows

Every experiment follows a fixed loop: **Question → Hypothesis → Build → Run → Visualize → Read the graph → Insight → Math → Challenge → Decide.** This keeps each step small (~30-90 min), visual-first, and always driven by what the last experiment revealed.

## Repo layout

| Path | What it is |
|:---|:---|
| `docs/AGENT_PROMPT.md` | The paste-into-agent execution prompt |
| `docs/ai-lab-partner.md` | How to use an AI as a learning/research partner |
| `state-of-the-field/` | Reference - the 10 landmark papers and where the field is headed |
| `requirements.txt` | Minimal Python deps (numpy, matplotlib) |

Experiments get added as scripts (numbered by experiment) as we go - no pre-built roadmap.

## First step

Ask the agent to run **Experiment 1 - Single LIF neuron** and deliver a plot of membrane potential and spikes over time.

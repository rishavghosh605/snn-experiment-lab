# Experiment 2 - Encoding a changing input (rate coding)

The question: how does a LIF neuron encode an input that changes over time?

## The idea

In Experiment 1 the input was constant, giving a steady rhythm. Here the input is a smoothly changing sine, and we ask how the spike train tracks it.

## Quiz (answers, all correct)

1. Rising then falling input -> spike rate follows it, but with a **lag** (set by tau).
2. Spikes are **bunched** where the input is strong, spread out where it is weak.
3. Below the firing condition -> **silence** (no spikes).

## Decisions made

### How to run it
- Input waveform: sine
- Peak strength: high (fires most of the cycle, dips below the firing condition at the troughs)
- Length: 1000 time steps

### What the plot looks like
- Layout: input on top, spike train below (stacked, shared time axis)

## What we found

- The spike train shows bursts where the input is above the firing condition, and silence where it is below.
- The firing rate follows the input, offset (lagged) by roughly the time constant tau.
- Confirmed numerically: 39 spikes over 1000 steps.

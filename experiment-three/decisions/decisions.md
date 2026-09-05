# Experiment 3 - Spiking vs standard network on XOR

The question: can a spiking network be trained to solve a task a standard network can, and does the choice of smooth activation change how well it trains?

## Quiz (answers, all correct)

1. A single unit cannot learn XOR because the outputs are **not linearly separable**.
2. A smooth activation helps because it **substitutes a smooth derivative** for the spike's undefined one.
3. Spiking energy is counted by **spike-driven synaptic operations**.

## Decisions made

### Setup
- Task: XOR, compared against the correct answer.
- Surrogates compared: STE, triangle, fast-sigmoid, arctan.
- Hidden layer: 4 neurons.
- Timesteps per input: 8.

### What we show
- 2D decision boundaries: truth, ANN, and each activation.
- Training loss curves per activation.
- Metric table (accuracy, energy, speed).

## What we found

- arctan matched the ANN (accuracy 1.00); triangle and fast-sigmoid got 0.75; STE failed (0.50).
- On this dense binary task the spiking net used more energy (more synaptic ops) than the ANN (MACs).
- The smooth activation's shape and scale decide whether training works.

# Questions and answers - Experiment 4

Active-recall notes. Name pattern: `questions-<N>.md` per experiment (named by purpose, like ADK agent roles). Re-read to self-test.

## Q1. Why is the spiral harder than XOR?

**Answer:** it needs a **curved, high-capacity decision boundary**, not "more input values." The spiral still has two inputs (like XOR). The difference is the shape of the boundary.

- XOR: the two classes are separated by two straight segments; a single hidden neuron can approximate that.
- Spiral: the classes curl around each other, so the boundary must bend and wrap. That needs more neurons (capacity) and enough nonlinearity.

**Analogy:** a straight line cannot separate two interleaved spirals. It is like trying to cut two intertwined spiral arms apart with one straight scissors stroke. You need a curving cut. This is the same reason deep, nonlinear networks replaced linear classifiers - the decision surface can bend around complex data.

## Q2. Are timesteps similar to a learning rate?

**Answer:** No - they are different knobs.

- **Timesteps (T)** is the temporal resolution: how long the net runs and how finely the spike rate encodes the input. More T = finer rate coding, a cleaner/reliable readout, but it costs **latency and energy**. It does not change how big a weight update is.
- **Learning rate** is the step size of gradient descent. Too big = overshoot/oscillate around a minimum; too small = slow convergence.
- They can interact: a cleaner rate code (more T) can make gradients more stable, which can help, but it is not a substitute for tuning the learning rate.

## Q3. What is a "high-capacity" boundary?

Capacity = how many functions/patterns the network can represent (expressivity). A high-capacity boundary is one that needs many parameters (more neurons, more nonlinearity) to carve - like the spiral. It is about what the network *can* represent, not about training.

## Preference (recording)

- Visualization: **multiselect**, and always show the **metrics table** and the **boundary grid**.

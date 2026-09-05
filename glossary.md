# Glossary - ML terms we use

The plain definitions, in standard ML language, for terms we use in these experiments.

## Expressivity

The set of functions a network can **represent**. It is about capacity, not learning. It is set by the **architecture** (depth, width, connectivity) and the **nonlinear activation**. A network can be expressive (it *could* represent a function) yet never be trained to.

## Trainability

Whether gradient descent can actually **find** the functions a network can represent. It is set by the **activation's derivative** and the resulting optimization landscape (gradient flow and curvature). A network that is expressive can still be untrainable if the gradient signal is missing or ill-conditioned.

> In one line: expressivity = what it *can* represent; trainability = whether we can *get* there.

## Smooth activation

A **differentiable** function used in place of a non-differentiable activation (like the spike's step, or ReLU at 0) so that backpropagation has a gradient to flow through. Examples: a smooth sigmoid-like curve, GELU, Swish.

*In the spiking literature this is often called a "surrogate gradient" or "surrogate" - but it is the same idea as choosing a smooth activation. A flat one (constant derivative, like a linear/identity activation) is the analogue of a hard, non-differentiable activation; it gives no useful gradient shape.*

## Rate coding

Encoding a continuous value as the *number* of spikes over a window. A denser spike train = a larger value. This is how a binary spike signal carries graded information over time.

## Firing condition

The rule for whether a neuron can spike at all on a steady current: it fires only if `I * tau > threshold`. Below that it stays silent.

## MACs

Multiply-accumulate operations - how a standard ANN's energy is counted (one per weight use, paid every forward pass).

## Synaptic operations (SOPs)

How a spiking net's energy is counted - each spike triggers one weight-accumulate per postsynaptic neuron it connects to. Only happens when a spike fires, so it scales with actual activity (sparsity).

## Latency / timesteps

How long a network takes to give an answer. An ANN answers in one forward pass. A spiking net needs several timesteps, because the spikes unfold over time.

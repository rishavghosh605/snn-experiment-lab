# Experiment 1 - Single LIF Neuron

The first experiment. Goal: build the smallest LIF (Leaky Integrate-and-Fire) neuron from scratch and see how a neuron turns a steady input into spikes.

## The idea

A normal network moves numbers around. A spiking network moves events (spikes) that happen at moments in time. A neuron has an internal voltage. Input raises it; when it crosses a threshold, the neuron fires a spike and resets. The timing of the spikes is the signal.

## The LIF neuron

- Integrate: input raises the voltage.
- Fire: when the voltage touches the threshold, it emits a spike.
- Reset: right after, the voltage drops to a low value.
- Leak: between inputs, the voltage slowly drains back down.

## Quiz (answer before we build)

1. After a neuron fires, what happens to its voltage?
   - It stays high and keeps firing
   - **It drops to a low value and starts again**

2. What carries the information in an SNN?
   - The size of each spike
   - **The timing of when spikes happen**

3. If I inject a bigger steady current, should it fire?
   - **More often**
   - Less often

## Decisions to make together

### How to run the experiment
- Injected current: low / medium / high
- Simulation length: short / long
- Spike threshold: default / higher
- Reset after firing: to zero / subtract the threshold

### What the plot should look like
- Layout: stacked panels (voltage on top, spike ticks below) / one overlaid panel
- Show the threshold as a dashed line: yes / no
- Show spike ticks on a separate row: yes / no

Suggested default (and the example in the interactive page): two stacked panels, voltage on top with a dashed threshold line, spike ticks below, shared time axis.

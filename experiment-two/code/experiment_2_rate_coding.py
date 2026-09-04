"""Experiment 2: encoding a time-varying (sine) input into a spike train.

Same LIF neuron as Experiment 1 (tau, threshold, reset), but the current now
oscillates as a sine. We look at how the spike pattern follows the input.

Visualization (stacked, shared time axis):
  top    - the input current I(t), with the firing-condition line
  bottom - a tick at every spike
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = 1000
dt = 1.0
tau = 20.0
threshold = 1.0
reset = 0.0

# Sine input. Peak high: above the firing condition most of the cycle,
# dipping below it near the troughs so the neuron can go silent there.
mean_current = 0.07
amp = 0.05
period = 250.0
firing_condition = threshold / tau  # I above this -> can fire


def input_current(t):
    return mean_current + amp * np.sin(2 * np.pi * t / period)


def run_lif(T, tau, threshold, reset, input_fn):
    v = 0.0
    voltage = np.zeros(T)
    spikes = []
    for t in range(T):
        v += (-v / tau + input_fn(t)) * dt
        if v >= threshold:
            spikes.append(t)
            v = reset
        voltage[t] = v
    return voltage, np.array(spikes)


def main():
    input_trace = np.array([input_current(t) for t in range(T)])
    _, spikes = run_lif(T, tau, threshold, reset, input_current)

    fig, (ax_in, ax_s) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5), height_ratios=[3, 1]
    )
    ax_in.plot(input_trace, color="#3d6df2", lw=1.5)
    ax_in.axhline(firing_condition, color="#d9534f", ls="--", lw=1.1, label="firing condition")
    ax_in.set_ylabel("input current")
    ax_in.set_title("LIF neuron with a sine input (rate coding)")
    ax_in.legend(loc="upper right")

    ax_s.eventplot(spikes, colors="#3d6df2")
    ax_s.set_yticks([])
    ax_s.set_ylabel("spikes")
    ax_s.set_xlabel("time step")

    fig.tight_layout()
    out = "experiment-two/figures/experiment_2_rate_coding.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)

    print("saved", out)
    print(f"spikes: {len(spikes)} of {T} steps")
    print(f"spike times (first 20): {spikes[:20].tolist()}")


if __name__ == "__main__":
    main()

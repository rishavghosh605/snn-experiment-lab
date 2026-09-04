"""Experiment 1: a single Leaky Integrate-and-Fire (LIF) neuron from scratch.

A constant current is injected for the whole run. The neuron integrates it
into a membrane voltage, leaks a little every step, fires a spike when it
crosses the threshold, then resets to zero.

Visualization: two stacked panels sharing one time axis.
  top    - membrane voltage, with the threshold as a dashed line
  bottom - a tick at every moment the neuron fired
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = 1000          # number of time steps
dt = 1.0          # time step (1 unit)
tau = 20.0        # membrane time constant (leak)
threshold = 1.0   # firing threshold
reset = 0.0       # voltage set to this right after a spike
current = 0.08    # injected steady current (medium -> a steady rhythm)


def lif_current(t, current=current):
    """Return the injected current at time t. Constant here."""
    return current


def run_lif(T, tau, threshold, reset, current_func):
    """Simulate one LIF neuron. Returns (voltage, spike_times)."""
    voltage = np.zeros(T)
    spikes = []
    v = 0.0
    for t in range(T):
        v += (-v / tau + current_func(t)) * dt      # leak + integrate
        if v >= threshold:                          # fire
            spikes.append(t)
            v = reset                               # reset
        voltage[t] = v
    return voltage, np.array(spikes)


def main():
    voltage, spikes = run_lif(T, tau, threshold, reset, lif_current)

    # Two stacked panels, shared x axis.
    fig, (ax_v, ax_s) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5), height_ratios=[3, 1]
    )
    ax_v.plot(voltage, color="#3d6df2", lw=1.5)
    ax_v.axhline(threshold, color="#d9534f", ls="--", lw=1.2, label="threshold")
    ax_v.set_ylabel("membrane voltage")
    ax_v.legend(loc="upper right")
    ax_v.set_title("Single LIF neuron with a constant current")

    ax_s.eventplot(spikes, colors="#3d6df2")
    ax_s.set_yticks([])
    ax_s.set_ylabel("spikes")
    ax_s.set_xlabel("time step")

    fig.tight_layout()
    fig.savefig("experiment-one/experiment_1_lif.svg", format="svg")
    plt.close(fig)

    print(f"spikes: {len(spikes)} of {T} steps")
    print(f"first 12 spike times: {spikes[:12].tolist()}")
    print(f"mean inter-spike interval: {np.diff(spikes).mean():.2f} steps" if len(spikes) > 1 else "n/a")


if __name__ == "__main__":
    main()

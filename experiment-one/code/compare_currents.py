"""Compare three LIF neurons under different currents (same tau and threshold).

Three columns (I = 0.04, 0.08, 0.16):
  top row    - membrane voltage with threshold line
  bottom row - spike ticks
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


def run_lif(I):
    v = 0.0
    spikes = []
    voltage = np.zeros(T)
    for t in range(T):
        v += (-v / tau + I) * dt
        if v >= threshold:
            spikes.append(t)
            v = reset
        voltage[t] = v
    return voltage, np.array(spikes)


def main():
    currents = [0.04, 0.08, 0.16]
    results = {I: run_lif(I) for I in currents}

    fig, axes = plt.subplots(2, 3, sharex="col", figsize=(12, 5),
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, I in enumerate(currents):
        voltage, spikes = results[I]
        # voltage panel
        ax_v = axes[0, col]
        ax_v.plot(voltage, color="#3d6df2", lw=1.4)
        ax_v.axhline(threshold, color="#d9534f", ls="--", lw=1.1, label="threshold")
        ax_v.set_title(f"I = {I}  (V_inf = {I * tau:.1f})")
        ax_v.set_ylabel("voltage")
        if col == 0:
            ax_v.legend(loc="upper right")
        # spike panel
        ax_s = axes[1, col]
        ax_s.eventplot(spikes, colors="#3d6df2")
        ax_s.set_yticks([])
        ax_s.set_ylabel(f"spikes ({len(spikes)})")
        ax_s.set_xlabel("time step")

    fig.suptitle("Single LIF neuron: effect of current strength (three cases)", y=1.02)
    fig.tight_layout()
    out = "experiment-one/figures/compare_currents.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    print("saved", out)
    for I in currents:
        s = results[I][1]
        print(f"I={I}: spikes={len(s)}, mean ISI={np.diff(s).mean() if len(s) > 1 else float('nan'):.2f}")


if __name__ == "__main__":
    main()

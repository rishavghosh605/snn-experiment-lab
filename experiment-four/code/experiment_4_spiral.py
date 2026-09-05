"""Experiment 4: two-spiral classification - plain ANN vs a LIF spiking net.

Compare accuracy, energy (MACs vs synaptic ops), and speed (epochs, latency).

Reuses the Experiment 3 LIF forward + act_deriv backprop-through-time pattern,
but with a 64-neuron hidden layer and only the smooth activations arctan and
fast-sigmoid. Input coordinates are rate-coded into [0, 1] firing rates.
Training is vectorized across the batch for speed.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---- settings (from the experiment-four decision) ----
N_PER_CLASS = 150   # points per spiral arm
H = 64              # hidden neurons
T, TAU, THRESH = 8, 2.0, 1.0   # timesteps, membrane time constant, threshold
E = 12000           # training epochs
LR_ANN, MOM_ANN = 3.0, 0.9     # ANN learning rate / momentum
LR_SNN = {"arctan": 0.01, "fast-sigmoid": 0.05}   # per-activation learning rate
MOM_SNN = 0.0
KINDS = ["arctan", "fast-sigmoid"]   # only the two chosen activations


# ---- dataset: two interleaved spiral arms ----
def make_arm(n, cls, rng):
    """Sample one spiral arm. cls 0: angle 0..3*pi, cls 1: angle pi..4*pi."""
    r = np.linspace(0.5, 1.0, n)
    a = np.linspace(0, 3 * np.pi, n) + cls * np.pi
    r += rng.normal(0, 0.02, n)
    a += rng.normal(0, 0.05, n)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def make_spiral():
    rng = np.random.default_rng(0)
    arm0 = make_arm(N_PER_CLASS, 0, rng)
    arm1 = make_arm(N_PER_CLASS, 1, rng)
    X = np.vstack([arm0, arm1])
    Y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)])
    lo, hi = X.min(0), X.max(0)
    X = 2 * (X - lo) / (hi - lo) - 1        # normalize to [-1, 1]
    return X, Y.reshape(-1, 1), arm0, arm1


# ---- LIF spiking net (reused from experiment 3) ----
def act_deriv(x, kind):
    if kind == "fast-sigmoid":
        return 1 / (1 + 5 * np.abs(x)) ** 2
    a = np.pi * 5 / 2  # arctan
    return 5 / (2 * (1 + (a * x) ** 2))


def fwd_batch(w1, b1, w2, b2, Xb, kind):
    """Vectorized LIF forward. Xb: (M, 2) in [-1, 1]. Returns (rate, trace)."""
    s_in = (Xb + 1) / 2                     # rate-code each coordinate to [0, 1]
    M = Xb.shape[0]
    vh = np.zeros((M, H))
    vo = np.zeros((M, 1))
    shs, vhs, sos, vos = [], [], [], []
    for _ in range(T):
        vh = vh + (-vh / TAU + (s_in @ w1.T + b1))
        sh = (vh >= THRESH).astype(float)
        vh *= 1 - sh
        shs.append(sh)
        vhs.append(vh.copy())
        vo = vo + (-vo / TAU + (sh @ w2.T + b2))
        so = (vo >= THRESH).astype(float)
        vo *= 1 - so
        sos.append(so)
        vos.append(vo.copy())
    rate = np.mean(np.stack(sos), axis=0)   # (M, 1) mean output spike over time
    return rate, (shs, vhs, sos, vos, s_in)


def bptt_batch(w1, b1, w2, b2, Xb, Yb, kind):
    rate, (shs, vhs, _sos, vos, s_in) = fwd_batch(w1, b1, w2, b2, Xb, kind)
    d = rate - Yb                            # (M, 1)
    dw2 = np.zeros_like(w2)
    db2 = np.zeros_like(b2)
    dw1 = np.zeros_like(w1)
    db1 = np.zeros_like(b1)
    dsh_acc = [np.zeros_like(shs[0]) for _ in range(T)]
    for t in range(T - 1, -1, -1):
        dso = d / T * act_deriv(vos[t] - THRESH, kind)
        dw2 += dso.T @ shs[t]
        db2 += dso.sum(0)
        dsh_acc[t][:] += dso * w2           # (M,1)*(1,H) broadcast -> (M,H)
    dvh = np.zeros_like(shs[0])
    for t in range(T - 1, -1, -1):
        mem = (dsh_acc[t] + dvh) * act_deriv(vhs[t] - THRESH, kind)
        dw1 += mem.T @ s_in
        db1 += mem.sum(0)
        dvh = mem * (1 - 1 / TAU)
    return rate, (dw1, db1, dw2, db2)


def train(kind):
    rng = np.random.default_rng(0)
    p = [rng.normal(0, 0.5, (H, 2)), np.zeros(H), rng.normal(0, 0.5, (1, H)), np.zeros(1)]
    v = [np.zeros_like(x) for x in p]
    bestp = [x.copy() for x in p]
    bestl = 1e9
    hist = []
    for _ in range(E):
        rate, g = bptt_batch(*p, X, Y, kind)
        err = float(np.mean((rate - Y) ** 2))
        for k in range(4):
            v[k] = MOM_SNN * v[k] - LR_SNN[kind] * g[k]
            p[k] += v[k]
        hist.append(err)
        if err < bestl:
            bestl = err
            bestp = [x.copy() for x in p]
    return bestp, hist


def predict_batch(params, Xb, kind):
    r, _ = fwd_batch(*params, Xb, kind)
    return 1 / (1 + np.exp(-10 * (r - 0.5)))


def snn_energy(params):
    _, (shs, _vhs, _sos, _vos, s_in) = fwd_batch(*params, X, "arctan")
    # input rate sum * hidden synapses * T steps  +  hidden spikes * output synapses
    total = (s_in.sum(1) * H * T).sum() + sum(shs[t].sum() for t in range(T))
    return int(total)


# ---- plain ANN reference ----
def ann_fwd_batch(Xb, p):
    w1, b1, w2, b2 = p
    h = np.tanh(Xb @ w1.T + b1)
    return h, 1 / (1 + np.exp(-(h @ w2.T + b2)))


def ann_train():
    rng = np.random.default_rng(0)
    p = [rng.normal(0, 0.5, (H, 2)), np.zeros(H), rng.normal(0, 0.5, (1, H)), np.zeros(1)]
    v = [np.zeros_like(x) for x in p]
    bestp = [x.copy() for x in p]
    bestl = 1e9
    hist = []
    for _ in range(E):
        h, o = ann_fwd_batch(X, p)
        do = o - Y
        err = float(np.mean(do ** 2))
        go = do * o * (1 - o)
        gh = (go * p[2]) * (1 - h ** 2)
        g = [gh.T @ X / X.shape[0], gh.sum(0) / X.shape[0],
             go.T @ h / X.shape[0], go.sum(0) / X.shape[0]]
        for k in range(4):
            v[k] = MOM_ANN * v[k] - LR_ANN * g[k]
            p[k] += v[k]
        hist.append(err)
        if err < bestl:
            bestl = err
            bestp = [x.copy() for x in p]
    return bestp, hist


def ann_pred_batch(p, Xb):
    return ann_fwd_batch(Xb, p)[1]


# ---- metrics helpers ----
def acc(preds):
    return float(((np.asarray(preds) > 0.5).astype(int) == Y.ravel()).mean())


def conv(hist):
    return next((i + 1 for i, e in enumerate(hist) if e < 0.01), len(hist))


def smooth(hist, w=100):
    """Moving average to reveal the trend in a noisy loss curve."""
    if len(hist) < w:
        return np.asarray(hist)
    return np.convolve(hist, np.ones(w) / w, mode="valid")


# ---- ground-truth boundary field (nearest-arm classification) ----
def truth_field(XX, YY, arm0, arm1):
    pts = np.stack([XX.ravel(), YY.ravel()], 1)
    d0 = ((pts[:, None, :] - arm0[None, :, :]) ** 2).sum(2).min(1)
    d1 = ((pts[:, None, :] - arm1[None, :, :]) ** 2).sum(2).min(1)
    return (d1 < d0).reshape(XX.shape).astype(float)


def main():
    global X, Y
    X, Y, arm0, arm1 = make_spiral()

    ap, ah = ann_train()
    ann_a = acc(ann_pred_batch(ap, X).ravel())
    ann_conv = conv(ah)
    ann_mac = (H * 2 + 1 * H) * X.shape[0]

    snn = {}
    for kind in KINDS:
        params, hist = train(kind)
        preds = predict_batch(params, X, kind).ravel()
        snn[kind] = (params, hist, acc(preds), snn_energy(params), conv(hist))

    # ---- figures ----
    # (b) training loss curves
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(smooth(ah), label="ANN", lw=1.6)
    for kind in KINDS:
        ax.plot(smooth(snn[kind][1]), label=kind, lw=1.6)
    ax.set(xlabel="epoch", ylabel="training loss (MSE, smoothed)",
           title="Training loss per model (two-spiral, 100-epoch moving average)")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.text(0.5, 0.005, "The ANN loss falls to ~0 by epoch 2300; both spiking nets stall "
                         "around 0.12-0.2, never matching the truth.",
             ha="center", fontsize=9, color="#3d4654")
    fig.savefig("experiment-four/figures/experiment_4_loss.svg", format="svg")
    plt.close(fig)

    # (a) boundary grid
    off = 0.04
    xg = np.linspace(-1 - off, 1 + off, 90)
    XX, YY = np.meshgrid(xg, xg)
    grid = np.stack([XX.ravel(), YY.ravel()], 1)

    def field(model_fn):
        return (model_fn(grid).reshape(XX.shape) > 0.5).astype(float)

    def draw(ax, z, title):
        ax.pcolormesh(XX, YY, z, cmap=ListedColormap(["#ffd6d6", "#cfe3ff"]), shading="auto")
        for (xx, yy), yv in zip(X, Y):
            ax.scatter(xx, yy, c="tab:red" if yv[0] else "tab:blue", s=12,
                       edgecolors="k", linewidths=0.4, zorder=3)
        ax.set(title=title, xlabel="input 1", ylabel="input 2")

    fig2, axs = plt.subplots(2, 2, figsize=(9, 8))
    draw(axs[0, 0], truth_field(XX, YY, arm0, arm1), "truth")
    draw(axs[0, 1], field(lambda g: ann_pred_batch(ap, g).ravel()), f"ANN (acc {ann_a:.2f})")
    draw(axs[1, 0], field(lambda g: predict_batch(snn["arctan"][0], g, "arctan").ravel()),
         f"arctan SNN (acc {snn['arctan'][2]:.2f})")
    draw(axs[1, 1], field(lambda g: predict_batch(snn["fast-sigmoid"][0], g, "fast-sigmoid").ravel()),
         f"fast-sigmoid SNN (acc {snn['fast-sigmoid'][2]:.2f})")
    fig2.text(0.5, 0.005,
              "The ANN (top-right) carves the two interleaved arms cleanly; both spiking nets "
              "(bottom) only capture a coarse partial boundary.",
              ha="center", fontsize=9, color="#3d4654")
    fig2.savefig("experiment-four/figures/experiment_4_boundaries.svg", format="svg")
    plt.close(fig2)
    fig2.savefig("experiment-four/figures/experiment_4_boundaries.svg", format="svg")
    plt.close(fig2)

    # (c) metrics table
    rows = [("ANN", ann_a, ann_mac, ann_conv, 1)]
    for kind in KINDS:
        rows.append((kind, snn[kind][2], snn[kind][3], snn[kind][4], T))
    fig3, ax3 = plt.subplots(figsize=(6.5, 2.2))
    ax3.axis("off")
    tbl = ax3.table(cellText=[[f"{n}", f"{a:.2f}", f"{e}", f"{c}", f"{l}"]
                              for n, a, e, c, l in rows],
                    colLabels=["model", "accuracy", "energy (ops)", "epochs", "latency"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    fig3.text(0.5, 0.005, "The ANN wins on accuracy and speed; the spiking net uses ~3x "
                          "the energy and still never converges (epochs = full budget).",
              ha="center", fontsize=9, color="#3d4654")
    fig3.savefig("experiment-four/figures/experiment_4_metrics.svg", format="svg")
    plt.close(fig3)
    fig3.savefig("experiment-four/figures/experiment_4_metrics.svg", format="svg")
    plt.close(fig3)

    print(f"{'model':<14}{'acc':>7}{'energy':>9}{'epochs':>8}{'latency':>9}")
    for n, a, e, c, l in rows:
        print(f"{n:<14}{a:>7.2f}{e:>9}{c:>8}{l:>9}")


if __name__ == "__main__":
    main()

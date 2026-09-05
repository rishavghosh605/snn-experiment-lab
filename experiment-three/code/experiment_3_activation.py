"""Train a plain ANN and a LIF spiking net (smooth activations) on XOR.

Compare accuracy, energy (synaptic ops vs MACs), and speed.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])
T, TAU, THRESH, E, LR, LR_ANN = 8, 2.0, 1.0, 1500, 0.05, 0.5


def act_deriv(x, kind):
    if kind == "ste":
        return np.ones_like(x)
    if kind == "triangle":
        return np.clip(1 - np.abs(x), 0, 1)
    if kind == "fast-sigmoid":
        return 1 / (1 + 5 * np.abs(x)) ** 2
    a = np.pi * 5 / 2  # arctan
    return 5 / (2 * (1 + (a * x) ** 2))


def fwd(w1, b1, w2, b2, x):
    s_in = np.array([x[0], x[1]])
    vh = np.zeros(4)
    vo = np.zeros(1)
    shs, vhs, vos, sos = [], [], [], []
    for _ in range(T):
        vh = vh + (-vh / TAU + (w1 @ s_in + b1))
        sh = (vh >= THRESH).astype(float)
        vh *= 1 - sh
        shs.append(sh.copy())
        vhs.append(vh.copy())
        vo = vo + (-vo / TAU + (w2 @ sh + b2))
        so = (vo >= THRESH).astype(float)
        vo *= 1 - so
        sos.append(so.copy())
        vos.append(vo.copy())
    return float(np.mean(sos)), (shs, vhs, sos, vos, s_in)


def bptt(w1, b1, w2, b2, x, y, kind):
    rate, (shs, vhs, _sos, vos, s_in) = fwd(w1, b1, w2, b2, x)
    d = rate - y[0]
    dw2 = np.zeros_like(w2)
    db2 = np.zeros_like(b2)
    dw1 = np.zeros_like(w1)
    db1 = np.zeros_like(b1)
    dsh_acc = [np.zeros(4) for _ in range(T)]
    for t in range(T - 1, -1, -1):
        dso = d / T * act_deriv(vos[t] - THRESH, kind)
        dw2 += np.outer(dso, shs[t]).reshape(1, 4)
        db2 += dso
        dsh_acc[t][:] += w2[0, :] * dso[0]
    dvh = np.zeros(4)
    for t in range(T - 1, -1, -1):
        mem = (dsh_acc[t] + dvh) * act_deriv(vhs[t] - THRESH, kind)
        dw1 += np.outer(mem, s_in).reshape(4, 2)
        db1 += mem
        dvh = mem * (1 - 1 / TAU)
    return rate, (dw1, db1, dw2, db2)


def predict(params, x, kind):
    w1, b1, w2, b2 = params
    r, _ = fwd(w1, b1, w2, b2, x)
    return 1 / (1 + np.exp(-10 * (r - 0.5)))


def train(kind):
    rng = np.random.default_rng(0)
    w1 = rng.normal(0, 0.5, (4, 2))
    b1 = np.zeros(4)
    w2 = rng.normal(0, 0.5, (1, 4))
    b2 = np.zeros(1)
    hist = []
    for _ in range(E):
        err = 0.0
        gs = [np.zeros_like(w1), np.zeros_like(b1), np.zeros_like(w2), np.zeros_like(b2)]
        for i in range(4):
            r, g = bptt(w1, b1, w2, b2, X[i], Y[i], kind)
            err += (r - Y[i][0]) ** 2
            for k in range(4):
                gs[k] += g[k]
        err /= 4
        hist.append(err)
        for k in range(4):
            w1, b1, w2, b2 = w1 - LR * gs[0] / 4, b1 - LR * gs[1] / 4, w2 - LR * gs[2] / 4, b2 - LR * gs[3] / 4
    return (w1, b1, w2, b2), hist


def energy(params, kind):
    w1, b1, w2, b2 = params
    total = 0
    for x in X:
        _, c = fwd(w1, b1, w2, b2, x)
        total += c[4].sum() * 4 + sum(s.sum() for s in c[0])  # input spikes x4 + hidden spikes x1
    return int(total * T)


def acc(preds):
    return float(((np.array(preds) > 0.5).astype(int) == Y.ravel()).mean())


def conv(hist):
    return next((i + 1 for i, e in enumerate(hist) if e < 0.01), len(hist))


# ---- plain ANN reference ----
def ann_fwd(x, p):
    w1, b1, w2, b2 = p
    h = np.tanh(w1 @ x + b1)
    return h, 1 / (1 + np.exp(-(w2 @ h + b2)))


def ann_train():
    rng = np.random.default_rng(0)
    p = [rng.normal(0, 0.5, (4, 2)), np.zeros(4), rng.normal(0, 0.5, (1, 4)), np.zeros(1)]
    hist = []
    for _ in range(E):
        err = 0.0
        g = [np.zeros_like(p[0]), np.zeros_like(p[1]), np.zeros_like(p[2]), np.zeros_like(p[3])]
        for i in range(4):
            x = X[i]
            y = Y[i][0]
            h, o = ann_fwd(x, p)
            do = o - y
            err += do**2
            go = do * o * (1 - o)
            g[2] += np.outer(go, h)
            g[3] += go
            gh = (p[2].T @ go) * (1 - np.tanh(p[0] @ x + p[1]) ** 2)
            g[0] += np.outer(gh, x)
            g[1] += gh
        err /= 4
        hist.append(err)
        p = [p[k] - LR_ANN * g[k] / 4 for k in range(4)]
    return p, hist


def ann_pred(p, x):
    return ann_fwd(x, p)[1][0]


def main():
    ap, ah = ann_train()
    apred = [ann_pred(ap, x) for x in X]
    ann_a, ann_conv = acc(apred), conv(ah)
    ann_mac = (4 * 2 + 1 * 4) * 4

    kinds = ["ste", "triangle", "fast-sigmoid", "arctan"]
    snn = {}
    for kind in kinds:
        params, hist = train(kind)
        preds = [predict(params, x, kind) for x in X]
        snn[kind] = (params, hist, acc(preds), energy(params, kind), conv(hist))

    # figures
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in kinds:
        ax.plot(snn[kind][1], label=kind, lw=1.3)
    ax.set(xlabel="epoch", ylabel="training loss", title="SNN training loss per activation")
    ax.legend()
    fig.tight_layout()
    fig.savefig("experiment-three/figures/experiment_3_loss.svg", format="svg")
    plt.close(fig)

    off = 0.02
    xg = np.linspace(-off, 1 + off, 80)
    XX, YY = np.meshgrid(xg, xg)

    def field(fn):
        z = np.zeros_like(XX)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                z[i, j] = fn((XX[i, j], YY[i, j]))
        return z

    def draw(ax, z, title):
        ax.pcolormesh(XX, YY, z, cmap=ListedColormap(["#ffd6d6", "#cfe3ff"]), shading="auto")
        for (xx, yy), yv in zip(X, Y):
            ax.scatter(xx, yy, c="tab:red" if yv[0] else "tab:blue", edgecolors="k", zorder=3)
        ax.set(title=title, xlabel="input 1", ylabel="input 2")

    fig2, axs = plt.subplots(2, 3, figsize=(11, 6))
    draw(axs[0, 0], np.zeros_like(XX), "truth")
    draw(axs[0, 1], field(lambda p: ann_pred(ap, p)) > 0.5, f"ANN (acc {ann_a:.2f})")
    draw(axs[0, 2], field(lambda p: predict(snn["ste"][0], p, "ste")) > 0.5,
         f"STE (acc {snn['ste'][2]:.2f})")
    draw(axs[1, 0], field(lambda p: predict(snn["triangle"][0], p, "triangle")) > 0.5,
         f"triangle (acc {snn['triangle'][2]:.2f})")
    draw(axs[1, 1], field(lambda p: predict(snn["fast-sigmoid"][0], p, "fast-sigmoid")) > 0.5,
         f"fast-sigmoid (acc {snn['fast-sigmoid'][2]:.2f})")
    draw(axs[1, 2], field(lambda p: predict(snn["arctan"][0], p, "arctan")) > 0.5,
         f"arctan (acc {snn['arctan'][2]:.2f})")
    fig2.tight_layout()
    fig2.savefig("experiment-three/figures/experiment_3_boundaries.svg", format="svg")
    plt.close(fig2)

    print(f"{'model':<14}{'acc':>7}{'energy':>9}{'epochs':>8}{'latency':>9}")
    print(f"{'ANN':<14}{ann_a:>7.2f}{ann_mac:>9}{ann_conv:>8}{1:>9}")
    for kind in kinds:
        print(f"{kind:<14}{snn[kind][2]:>7.2f}{snn[kind][3]:>9}{snn[kind][4]:>8}{T:>9}")

    print("\nANN preds:", np.round(apred, 2))
    for kind in kinds:
        print(f"{kind} preds:", np.round([predict(snn[kind][0], x, kind) for x in X], 2))


if __name__ == "__main__":
    main()

"""Only place that writes PNGs: themed figures for the deck, from results/*.json.

No GPU. Output: results/figures/*.png (not results/*.png).
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results"


def savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print("wrote", path)


BG, INK, MUTED = "#F7F5F0", "#1F2A33", "#5C6570"
GRID, ACC, SOFT, LACC = "#E3DED2", "#D2622A", "#9AA0A6", "#E9A883"
W = 11.56  # slide content width, inches

for f in (Path(__file__).parent / "fonts").glob("Inter-*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 12,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.facecolor": BG,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.9,
    "axes.axisbelow": True,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.labelsize": 12,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "lines.solid_capstyle": "round",
})

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def comma(v, _=None):
    return f"{v:g}".replace(".", ",")


def pow10(v, _=None):
    if v <= 0:
        return ""
    e = math.log10(v)
    return f"10{str(round(e)).translate(SUP)}" if abs(e - round(e)) < 1e-6 else ""


def plain(v, _=None):
    return f"{v:,.0f}".replace(",", "\u202f")


def end_label(ax, x, y, text, color, dx=1.1, **kw):
    ax.text(x * dx, y, text, color=color, va="center", ha="left", fontsize=12,
            fontweight="semibold", clip_on=False, **kw)


def speedup_text(r):
    return f"{round(r, -1):.0f}×" if r >= 500 else f"{r:.0f}×"


def load(name, root):
    return json.loads((root / name).read_text())


def newton(root, out):
    data = load("newton.json", root)["curves"]
    styles = {"128": (ACC, "o"), "512": (INK, "s"), "2048": (SOFT, "^")}
    fig, axes = plt.subplots(1, 2, figsize=(W, 4.0), sharey=True)
    for ax, (cell, by_len) in zip(axes, data.items()):
        ax.axhline(2 ** -23, color=MUTED, lw=1, ls=(0, (4, 4)))
        for length, series in by_len.items():
            color, marker = styles[length]
            xs = [int(k) for k in series]
            ax.plot(xs, [series[str(k)] for k in xs], color=color, marker=marker, ms=7,
                    lw=2, label=f"L = {length}", zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(4e-8, 0.6)
        ax.yaxis.set_major_formatter(FuncFormatter(pow10))
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.8, 4.2)
        ax.set_xlabel("итерации Ньютона")
        ax.set_title(cell, pad=10)
        ax.axvline(3, color=ACC, lw=1, alpha=0.45, zorder=1)
        ax.minorticks_off()
    axes[0].set_ylabel("ошибка относительно цикла")
    axes[0].text(0.85, 2 ** -23 * 1.25, "точность float32", color=MUTED, fontsize=11, ha="left", va="bottom")
    for ax in axes:
        ax.text(3.06, 0.3, "3 итерации", color=ACC, fontsize=12, fontweight="semibold", va="center")
    axes[1].legend(loc="upper right", handlelength=1.4, bbox_to_anchor=(1.0, 0.82))
    fig.tight_layout(w_pad=2.5)
    savefig(fig, out / "newton.png")
    plt.close(fig)


def modes(root, out):
    rows = load("parity.json", root)["mode_losses"]
    steps = [r["step"] for r in rows]
    gap = max(abs(r["sequential"] - r["parallel_CUDA"]) for r in rows)
    fig, ax = plt.subplots(figsize=(W, 3.9))
    ax.plot(steps, [r["sequential"] for r in rows], color=INK, lw=4.5, alpha=0.9, label="sequential (цикл)")
    ax.plot(steps, [r["parallel_CUDA"] for r in rows], color=ACC, lw=1.8, ls=(0, (3, 2.2)),
            label="parallel_CUDA")
    ax.set_xlabel("шаг обучения")
    ax.set_ylabel("потеря на обучении")
    ax.yaxis.set_major_formatter(FuncFormatter(comma))
    ax.set_xlim(0, steps[-1] + 0.5)
    ax.legend(loc="upper right", handlelength=2.2)
    ax.text(0.99, 0.56, f"наибольшая разница между кривыми\n{gap * 1e7:.1f}·10⁻⁷".replace(".", ","),
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=12, linespacing=1.5)
    fig.tight_layout()
    savefig(fig, out / "modes.png")
    plt.close(fig)


def speed(root, out):
    cells = load("speed.json", root)["cells"]
    look = {
        "sequential": ("цикл", INK, 2.4, "o"),
        "parallel": ("PyTorch", SOFT, 1.8, "o"),
        "parallel_CUDA": ("CUDA", LACC, 2.0, "o"),
        "parallel_FUSED": ("fused", ACC, 2.8, "o"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(W, 4.2), sharey=True)
    for ax, (cell, rows) in zip(axes, cells.items()):
        last = {}
        for mode, (label, color, lw, marker) in look.items():
            pts = [(r["L"], r["ms"]) for r in rows if r["mode"] == mode and "ms" in r]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, lw=lw, marker=marker, ms=5.5, zorder=3)
            last[mode] = (xs[-1], ys[-1])
        # right-hand labels: push apart in log space so neighbours stay readable
        gap, placed = 0.2, []
        for mode in sorted(look, key=lambda m: last[m][1]):
            y = math.log10(last[mode][1])
            if placed and y - placed[-1] < gap:
                y = placed[-1] + gap
            placed.append(y)
            label, color = look[mode][0], look[mode][1]
            end_label(ax, last[mode][0], 10 ** y, label, color if color != LACC else "#C27B52")
        x_end = last["sequential"][0]
        ratio = last["sequential"][1] / last["parallel_FUSED"][1]
        ax.annotate("", xy=(x_end, last["parallel_FUSED"][1] * 1.12), xytext=(x_end, last["sequential"][1] * 0.9),
                    arrowprops=dict(arrowstyle="<->", color=ACC, lw=1.3, shrinkA=0, shrinkB=0))
        ax.text(x_end * 0.86, math.sqrt(last["sequential"][1] * last["parallel_FUSED"][1]), speedup_text(ratio),
                color=ACC, fontsize=17, fontweight="bold", ha="right", va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([64, 256, 1024, 4096])
        ax.xaxis.set_major_formatter(FuncFormatter(plain))
        ax.yaxis.set_major_formatter(FuncFormatter(plain))
        ax.set_xlim(55, 4096 * 2.4)
        ax.set_ylim(1, 6000)
        ax.minorticks_off()
        ax.set_xlabel("длина последовательности L")
        ax.set_title(cell, pad=10)
    axes[0].set_ylabel("прямой + обратный проход, мс")
    fig.tight_layout(w_pad=2.0)
    savefig(fig, out / "speed.png")
    plt.close(fig)


def reduction(root, out):
    rows = load("reduction.json", root)["diag_timings_ms"]
    xs = [r["L"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    for key, label, color in (("torch", "PyTorch", SOFT), ("cuda", "CUDA", ACC)):
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, color=color, lw=2.6, marker="o", ms=6.5, zorder=3)
        end_label(ax, xs[-1], ys[-1], label, color, dx=1.08)
    top, bottom = rows[-1]["torch"], rows[-1]["cuda"]
    ax.annotate("", xy=(xs[-1], bottom * 1.2), xytext=(xs[-1], top * 0.85),
                arrowprops=dict(arrowstyle="<->", color=ACC, lw=1.3, shrinkA=0, shrinkB=0))
    ax.text(xs[-1] * 0.88, math.sqrt(top * bottom), speedup_text(top / bottom), color=ACC, fontsize=17,
            fontweight="bold", ha="right", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    ax.yaxis.set_major_formatter(FuncFormatter(comma))
    ax.set_xlim(100, 4096 * 2.2)
    ax.set_ylim(0.025, 4)
    ax.minorticks_off()
    ax.set_xlabel("длина последовательности L")
    ax.set_ylabel("время, мс")
    fig.tight_layout()
    savefig(fig, out / "reduction.png")
    plt.close(fig)


PARITY = {"LinearRNN": ("линейная RNN", SOFT), "ParaGRU": ("ParaGRU", ACC), "ParaLSTM": ("ParaLSTM", INK)}


def parity_train(root, out):
    hist = load("parity.json", root)["histories"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 4.1))
    for ax, key in zip(axes, ("last_acc", "loss")):
        for name, (label, color) in PARITY.items():
            h = hist[name]
            xs, ys = [r["step"] for r in h], [r[key] for r in h]
            ax.plot(xs, ys, color=color, lw=2.6, zorder=3)
            ax.plot(xs[-1:], ys[-1:], "o", color=color, ms=6.5, zorder=4)
            # ParaLSTM finishes first: label it on the left so the two names do not collide
            left = name == "ParaLSTM"
            ax.text(xs[-1] + (-10 if left else 10), ys[-1], label, color=color if color != SOFT else MUTED,
                    fontsize=12, fontweight="semibold", va="center", ha="right" if left else "left")
        ax.set_xlim(0, 500)
        ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_xlabel("шаг обучения")
        ax.yaxis.set_major_formatter(FuncFormatter(comma))
    for ax, (y, text) in zip(axes, ((0.5, "случайное угадывание"), (math.log(2), "ln 2 — монетка"))):
        ax.axhline(y, color=MUTED, lw=1, ls=(0, (4, 4)), zorder=2)
        ax.text(262, y + (0.02 if ax is axes[0] else 0.012), text, color=MUTED, fontsize=11, va="bottom")
    axes[0].set_ylim(0.42, 1.04)
    axes[0].set_title("точность на всей строке", pad=10)
    axes[1].set_ylim(0.4, 0.84)
    axes[1].set_title("потеря на обучении", pad=10)
    fig.tight_layout(w_pad=2.5)
    savefig(fig, out / "parity_train.png")
    plt.close(fig)


def parity_trace(root, out):
    tr = load("parity.json", root)["trace"]
    labels = tr["labels"]
    n = len(labels)
    fig, axes = plt.subplots(3, 1, figsize=(W, 4.5), sharex=True)
    for ax, (name, (label, color)) in zip(axes, PARITY.items()):
        pred = tr["pred"][name]
        wrong = [i for i in range(n) if pred[i] != labels[i]]
        ax.step(range(n), labels, where="mid", color=INK, lw=5, alpha=0.16, zorder=2)
        ok = [i for i in range(n) if i not in wrong]
        ax.plot(ok, [pred[i] for i in ok], "o", color=INK, ms=4.8, zorder=3)
        ax.plot(wrong, [pred[i] for i in wrong], "o", color=ACC, ms=8, zorder=4)
        ax.set_ylim(-0.45, 1.45)
        ax.set_yticks([0, 1])
        ax.set_xlim(-1, n)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", labelsize=11, pad=2)
        ax.set_title(label, fontsize=13, pad=3)
        msg = f"ошибок: {len(wrong)} из {n}" if wrong else f"ошибок: 0 из {n}"
        ax.text(1.0, 1.0, msg, transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
                fontweight="semibold", color=ACC if wrong else MUTED)
    axes[-1].set_xlabel("позиция в строке")
    fig.tight_layout(h_pad=0.9)
    savefig(fig, out / "parity_trace.png")
    plt.close(fig)


def parity_position(root, out):
    pos = load("parity.json", root)["by_position"]
    fig, ax = plt.subplots(figsize=(W, 3.9))
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 4)))
    ax.text(26, 0.43, "случайное угадывание", color=MUTED, fontsize=11, va="top")
    # ParaGRU wide underneath, ParaLSTM thin on top: both stay visible where they coincide
    widths = {"LinearRNN": 2.4, "ParaGRU": 6.0, "ParaLSTM": 2.0}
    label_y = {"LinearRNN": None, "ParaGRU": 1.025, "ParaLSTM": 0.94}
    for name, (label, color) in PARITY.items():
        ys = pos[name]
        ax.plot(range(len(ys)), ys, color=color, lw=widths[name], alpha=0.55 if name == "ParaGRU" else 1, zorder=3)
        ax.text(len(ys) + 0.6, label_y[name] or ys[-1], label, color=color if color != SOFT else MUTED,
                fontsize=12, fontweight="semibold", va="center", clip_on=False)
    n = len(pos["ParaGRU"])
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0.25, 1.04)
    ax.yaxis.set_major_formatter(FuncFormatter(comma))
    ax.set_xlabel("позиция в строке")
    ax.set_ylabel("точность")
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    savefig(fig, out / "parity_position.png")
    plt.close(fig)


def main(root=DEFAULT_OUT):
    out = Path(root) / "figures"
    for fn in (newton, modes, speed, reduction, parity_train, parity_trace, parity_position):
        fn(Path(root), out)


if __name__ == "__main__":
    main()

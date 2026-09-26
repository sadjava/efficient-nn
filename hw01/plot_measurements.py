import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from equations import bytes_moved, energy, flops, latency, memory as memory_eq

HERE = Path(__file__).parent
CSV = HERE / "results" / "measurements.csv"
FIG = HERE / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
THETA = json.loads((HERE / "results" / "theta.json").read_text())
# This machine, from nvidia-smi. Axis values are MB (1e6), the label says MiB.
VRAM_MB = 6144 * 1024**2 / 1e6

raw = list(csv.DictReader(CSV.open()))
oom_rows = [r for r in raw if r["memory"] == "OOM"]
rows = [r for r in raw if r["memory"] != "OOM"]
S = np.array([int(r["S"]) for r in rows], dtype=float)
B = np.array([int(r["B"]) for r in rows], dtype=float)
flops_m = np.array([float(r["flops"]) for r in rows])
lat_m = np.array([float(r["latency"]) for r in rows])
mem_m = np.array([float(r["memory"]) for r in rows])
en_m = np.array([float(r["energy"]) for r in rows])
probe = B == 512
val = np.array([r["is_validation"] == "True" for r in rows]) & ~probe
train = ~val & ~probe
val_b = np.isin(B, (15, 24, 33))
val_s = val & ~val_b

F = flops(S, B)
Q = bytes_moved(S, B)
lat_p = latency(S, B, THETA)
mem_p = memory_eq(S, B)
en_p = energy(S, B, THETA)
Ss = np.array(sorted({int(r["S"]) for r in raw}), dtype=float)
s_line = np.geomspace(Ss.min(), Ss.max(), 200)
b_line = np.geomspace(1, 512, 200)

ZONE = {
    0: ("launch-bound", (0.75, 0.85, 0.95)),
    1: ("memory-bound", (0.97, 0.90, 0.72)),
    2: ("compute-bound", (0.75, 0.90, 0.78)),
}


def _scatter(ax, x, y):
    ax.scatter(x[train], y[train], s=22, c="C0", label="train (base grid)", zorder=3)
    ax.scatter(
        x[val_s], y[val_s], s=28, facecolors="none", edgecolors="C1",
        marker="s", label="val S (48, 192, 320, 448)", zorder=4,
    )
    ax.scatter(
        x[val_b], y[val_b], s=36, c="C3", marker="D",
        edgecolors="k", linewidths=0.4, label="val B (15, 24, 33)", zorder=5,
    )
    if probe.any():
        ax.scatter(
            x[probe], y[probe], s=40, c="k", marker="^",
            label="B=512 (not in fit)", zorder=6,
        )


def _xy(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.5)


def _loglog_yx(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")


def _winner(b, s):
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float)
    f = flops(s, b)
    q = bytes_moved(s, b)
    return np.argmax(
        np.stack(
            [
                np.full_like(f, THETA["launch_overhead"], dtype=float),
                q / THETA["bandwidth"],
                f / THETA["peak_flops"],
            ]
        ),
        axis=0,
    )


def _paint_s(ax, b, s0, s1):
    s = np.geomspace(s0, s1, 400)
    w = _winner(b, s)
    i0 = 0
    drawn = set()
    for i in range(1, len(s) + 1):
        if i == len(s) or w[i] != w[i0]:
            name, color = ZONE[int(w[i0])]
            label = name if name not in drawn else None
            ax.axvspan(s[i0], s[min(i, len(s) - 1)], color=color, lw=0, zorder=0, label=label)
            drawn.add(name)
            i0 = i


# 1. FLOPs. Closed form, no fit. The counter omits ReLU / pool / GAP.
fig, ax = plt.subplots(figsize=(5.6, 5.4))
_scatter(ax, F, flops_m)
_loglog_yx(ax, F.min(), F.max())
_xy(ax, "analytical FLOPs", "FlopCounterMode FLOPs", "FLOPs: equation vs counter")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "flops_vs_counter.png", dpi=140)
plt.close()

# 2. Memory. Left: parity, the 9.6 MB workspace lifts small points off y=x.
# Right: B=256 stays under the card; B=512 crosses it and OOMs.
fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
ax = axes[0]
_scatter(ax, mem_p / 1e6, mem_m / 1e6)
lo = min(mem_p.min(), mem_m.min()) / 1e6
hi = max(mem_p.max(), mem_m.max()) / 1e6
_loglog_yx(ax, lo, hi)
_xy(ax, "predicted memory (MB)", "measured peak (MB)", "Parity: small S sits above y = x")
ax.legend(fontsize=7)

ax = axes[1]
oom_labeled = False
for b, c in ((256, "C0"), (512, "C3")):
    ax.plot(s_line, memory_eq(s_line, b) / 1e6, "--", c=c, lw=1.2, label=f"predicted B={b}")
    m = B == b
    ax.scatter(S[m], mem_m[m] / 1e6, s=28, c=c, zorder=3, label=f"measured B={b}")
    for r in oom_rows:
        if int(r["B"]) != b:
            continue
        pred = memory_eq(int(r["S"]), b) / 1e6
        ax.scatter(
            int(r["S"]), pred, s=90, marker="X", c="k", zorder=5,
            label="OOM (at prediction)" if not oom_labeled else None,
        )
        oom_labeled = True
ax.axhline(VRAM_MB, color="k", ls=":", lw=1, label="GPU 6144 MiB")
ax.set_yscale("log")
_xy(ax, "S (px)", "memory (MB)", "OOM when the prediction crosses the card")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "memory_measured_vs_predicted.png", dpi=140)
plt.close()

# 3. Latency parity. Points off the line are kernel choices, not a bad S grid.
fig, ax = plt.subplots(figsize=(5.6, 5.4))
_scatter(ax, lat_p * 1e3, lat_m * 1e3)
lo = min(lat_p.min(), lat_m.min()) * 1e3
hi = max(lat_p.max(), lat_m.max()) * 1e3
_loglog_yx(ax, lo, hi)
_xy(ax, "predicted latency (ms)", "measured latency (ms)", "Latency: predicted vs measured")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "latency_measured_vs_predicted.png", dpi=140)
plt.close()

# 4. Latency vs S. Dashed line is the equation; markers are the GPU.
# B=1 is flat (launch). Larger B leave that floor.
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
for ax, b in zip(axes, (1, 32, 256)):
    _paint_s(ax, b, Ss.min(), Ss.max())
    m = B == b
    ax.plot(s_line, latency(s_line, b, THETA) * 1e3, "k--", lw=1.2, label="predicted", zorder=2)
    ax.scatter(S[m], lat_m[m] * 1e3, s=22, c="k", zorder=3, label="measured")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(Ss)
    ax.set_xticklabels([str(int(s)) for s in Ss], rotation=60, fontsize=7)
    ax.set_xlabel("S (px)")
    ax.set_title(f"B={b}")
    ax.grid(True, which="both", ls=":", lw=0.5)
axes[0].set_ylabel("latency (ms)")
zone_h = [Patch(facecolor=ZONE[k][1], label=ZONE[k][0]) for k in (0, 1, 2)]
axes[0].legend(zone_h + [
    Line2D([0], [0], color="k", ls="--", label="predicted"),
    Line2D([0], [0], color="k", marker="o", ls="none", label="measured"),
], [ZONE[k][0] for k in (0, 1, 2)] + ["predicted", "measured"], fontsize=7, loc="upper left")
fig.suptitle("Latency vs S — flat region is launch overhead")
fig.tight_layout()
fig.savefig(FIG / "latency_vs_S.png", dpi=140)
plt.close()

# 5. Latency vs B at sizes where the cuDNN heuristic is non-monotonic.
# Prediction is smooth; measured markers jump off it.
fig, ax = plt.subplots(figsize=(8.2, 5.0))
pred_labeled = False
for s, c in ((256, "C0"), (384, "C1"), (512, "C2")):
    m = S == s
    ax.scatter(B[m], lat_m[m] * 1e3, s=28, c=c, zorder=3, label=f"measured S={int(s)}")
    ax.plot(b_line, latency(s, b_line, THETA) * 1e3, "--", c=c, lw=1, label="predicted" if not pred_labeled else None)
    pred_labeled = True
    vb = m & val_b
    ax.scatter(B[vb], lat_m[vb] * 1e3, s=70, marker="D", facecolors="none", edgecolors=c, linewidths=1.2, zorder=4)
ax.scatter([], [], s=70, marker="D", facecolors="none", edgecolors="k", label="val B=15, 24, 33")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([1, 2, 4, 8, 15, 16, 24, 32, 33, 64, 128, 256, 512])
ticks = ax.set_xticklabels(
    ["1", "2", "4", "8", "15", "16", "24", "32", "33", "64", "128", "256", "512"], fontsize=8,
)
for tick, b in zip(ticks, (1, 2, 4, 8, 15, 16, 24, 32, 33, 64, 128, 256, 512)):
    if b in (15, 24, 33):
        tick.set_color("C3")
        tick.set_fontweight("bold")
_xy(ax, "batch size B", "latency (ms)", "Kernel choice: measured latency is not monotone in B")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "latency_vs_B.png", dpi=140)
plt.close()

# 6. Three terms. Large S is compute: F/π overtakes Q/β once AI clears the ridge.
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)
for ax, b in zip(axes, (1, 256)):
    _paint_s(ax, b, Ss.min(), Ss.max())
    f = flops(s_line, b)
    q = bytes_moved(s_line, b)
    ax.plot(s_line, np.full_like(s_line, THETA["launch_overhead"]) * 1e3, label="launch τ0")
    ax.plot(s_line, q / THETA["bandwidth"] * 1e3, label="memory Q/β")
    ax.plot(s_line, f / THETA["peak_flops"] * 1e3, ls="--", label="compute F/π")
    m = B == b
    ax.scatter(S[m], lat_m[m] * 1e3, s=22, c="k", zorder=3, label="measured")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("S (px)")
    ax.set_title(f"B={b}")
    ax.grid(True, which="both", ls=":", lw=0.5)
axes[0].set_ylabel("time (ms)")
axes[0].legend(fontsize=7)
fig.suptitle("Launch, then memory, then compute as the arithmetic intensity rises")
fig.tight_layout()
fig.savefig(FIG / "latency_decomposition.png", dpi=140)
plt.close()

# 7. Regime map on the assignment grid only (B=512 is the OOM probe, not a regime).
fit_b = np.array([1, 2, 4, 8, 15, 16, 24, 32, 33, 64, 128, 256], dtype=float)
grid = np.zeros((len(Ss), len(fit_b)))
for i, s in enumerate(Ss):
    grid[i] = _winner(fit_b, s)
fig, ax = plt.subplots(figsize=(8.2, 5.0))
cmap = ListedColormap([ZONE[k][1] for k in (0, 1, 2)])
ax.imshow(grid, origin="lower", aspect="auto", vmin=0, vmax=2, cmap=cmap)
ax.set_xticks(range(len(fit_b)), [str(int(b)) for b in fit_b])
for tick, b in zip(ax.get_xticklabels(), fit_b):
    if int(b) in (15, 24, 33):
        tick.set_color("C3")
        tick.set_fontweight("bold")
ax.set_yticks(range(len(Ss)), [str(int(s)) for s in Ss])
ax.set_xlabel("B")
ax.set_ylabel("S")
ax.set_title("Regime of each (S, B): launch, then memory, then compute")
ax.legend(
    [Patch(facecolor=ZONE[k][1], label=ZONE[k][0]) for k in (0, 1, 2)],
    [ZONE[k][0] for k in (0, 1, 2)],
    fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0,
)
for i in range(len(Ss)):
    for j in range(len(fit_b)):
        ax.text(j, i, "LMC"[int(grid[i, j])], ha="center", va="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "regime_map.png", dpi=140)
plt.close()

# 8. Energy parity. Same split as latency: θ was not fit on B=512.
fig, ax = plt.subplots(figsize=(5.6, 5.4))
_scatter(ax, en_p, en_m)
lo = min(en_p[en_p > 0].min(), en_m[en_m > 0].min())
hi = max(en_p.max(), en_m.max())
_loglog_yx(ax, lo, hi)
_xy(ax, "predicted energy (J)", "measured energy (J)", "Energy: predicted vs measured")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "energy_measured_vs_predicted.png", dpi=140)
plt.close()

stale = FIG / "latency_residuals.png"
if stale.exists():
    stale.unlink()

print("wrote", sorted(p.name for p in FIG.iterdir()))
print("OOM", [(r["S"], r["B"]) for r in oom_rows])

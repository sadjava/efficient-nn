# Homework 1 — analytical performance model

## Hardware and software

- GPU: NVIDIA GeForce RTX 3060 Laptop, 6144 MiB, driver 595.91.07
- PyTorch 2.9.0+cu130, Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.10
- FP32, `eval()` + `inference_mode()`. `cudnn.benchmark = False`, TF32 off (`allow_tf32 = False` and `fp32_precision = "ieee"`)

## Reproduce

From this directory:

```bash
python measure.py          # writes results/measurements.csv
python calibrate.py        # fits results/theta.json on is_validation=False
python plot_measurements.py
```

Grid: base $S \in \{32,64,128,224,256,384,512\}$, base $B \in \{1,2,4,8,16,32,64,128,256\}$, plus $S \in \{48,192,320,448\}$ and $B \in \{15,24,33\}$. Those 132 cells are the fit: extra $S$ or $B$ is `is_validation=True` (63 train / 69 val). $B=512$ is measured on top of that only to show OOM and is not used in $\theta$.

## Results

| | train MAPE | val MAPE |
|---|---:|---:|
| Latency | 35% | 38% |
| Energy | 25% | 32% |
| Memory | 31% | 25% |

`FlopCounterMode` matches conv+linear MACs exactly. The closed form also counts ReLU, MaxPool and GAP, about 0.2% more.

Fitted $\theta$: $\tau_0 = 0.418\,\text{ms}$, $\pi = 2.30\,\text{TFLOP/s}$, $\beta = 53.3\,\text{GB/s}$, ridge $\pi/\beta = 43.2$ FLOP/byte, $P_0 = 57.9\,\text{W}$, $e_Q = 5.71\times 10^{-10}\,\text{J/byte}$.

Of 132 cells: 22 launch-bound, 18 memory-bound, 92 compute-bound.

## Discussion

FLOPs are a closed form and do not depend on the GPU. The counter ignores elementwise ops, so the plot sits just off $y=x$. That gap is the ReLU/pool/GAP term, not a shape error.

Peak memory is weights plus the tensors alive inside the first MaxPool: input, conv 7×7 output, pool output, and the int64 index buffer (`max_pool2d_with_indices` allocates it even under `inference_mode`). That is $4\,161\,296 + 68\,B S^2$. On large cells ($BS^2 > 5\cdot 10^5$) the mean error is about 7%. Two things are missing from the formula, both visible on the log memory plot. A cuDNN workspace of 9.6 MB is present on most cells and does not scale with $BS^2$, so small images are predicted several times too low. On a few shapes the heuristic picks a fatter algorithm: $S=224, B=1$ measures 87 MB against a 7.6 MB prediction, and memory can drop when $B$ grows. Inside the 132-cell grid the largest point is 4.58 GB and fits. At $B=512$ the same formula crosses the 6144 MiB card between $S=384$ (measured 5.15 GB, prediction 5.14 GB, runs) and $S=448$ (prediction 7.0 GB, OOM). $S=512, B=512$ is OOM at a prediction of 9.13 GB. The crosses on the memory figure sit on that prediction, above the device line.

Latency is $\tau_0 + \max(F/\pi, Q/\beta)$. $F$ and $Q$ are both nearly proportional to $BS^2$, so a free pair $(\pi, \beta)$ slides to a corner where one ceiling never binds and the map is either all memory or all compute. The asymptotic intensity is $17751/380 \approx 47$ FLOP/byte; the weight term pulls it down at small $S$ (about 4 FLOP/byte at $S=32, B=1$). $\pi$ is the achieved FLOP rate. $\beta$ is then set so the ridge is the lowest value that still leaves both a memory band and a compute band on the training grid: 43.2 FLOP/byte. Below the ridge the cell is memory-bound, above it compute-bound, and if both terms are under $\tau_0$ it is launch-bound. That is the diagonal on the regime map: small $(S, B)$ launch, a yellow band as the work grows, green once the maps dominate the weights. Held-out latency MAPE is 38%. The remaining scatter is the cuDNN heuristic with `benchmark=False`. Examples: at $S=512$, $B=128$ takes 476 ms and $B=256$ takes 388 ms; at $S=256$, $B=4$ is 1.4 ms and $B=8$ is 7.6 ms. Validation batches 15, 24 and 33 fall in the same band as the powers of two.

Energy is the whole-GPU NVML counter over a ≥0.5 s burst, divided by the number of forwards. $E = P_0 T + e_Q Q$. About three quarters of the joules are $P_0 \approx 58\,\text{W}$: clocks, board and the laptop display. The rest follows bytes moved, $e_Q \approx 0.57\,\text{nJ/byte}$, not FLOPs. A fit with both $e_F F$ and $e_Q Q$ sets $e_F = 0$. On large $S$ the two formulas nearly agree, because $F \approx 47\,Q$. Short kernels still swing from ~20 W to ~150 W; that is boost and display noise.

import csv
import gc
import random
import statistics
import time
from pathlib import Path

import pynvml
import torch
from torch.utils.flop_counter import FlopCounterMode

from models import Model

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
# Torch 2.9 still honors the flags above and warns they are going away.
if hasattr(torch.backends.cudnn, "conv"):
    torch.backends.cudnn.conv.fp32_precision = "ieee"
if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
    torch.backends.cuda.matmul.fp32_precision = "ieee"

pynvml.nvmlInit()
_nvml = pynvml.nvmlDeviceGetHandleByIndex(0)

S_BASE_GRID = [32, 64, 128, 224, 256, 384, 512]
S_ADD_GRID = [48, 192, 320, 448]
B_BASE_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]
B_ADD_GRID = [15, 24, 33]
# Outside the 132-cell grid. Large S at this batch is how the 6 GB card OOMs.
# calibrate.py does not fit on it.
B_OOM_GRID = [512]


def _energy_j(model, x, min_s: float = 0.5) -> float:
    # Whole-GPU joules from the NVML energy counter (mJ), not a power sample
    # taken on the CPU timeline. Laptop display power sits in this number.
    torch.cuda.synchronize()
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(_nvml)
    n = 0
    t0 = time.perf_counter()
    while True:
        model(x)
        n += 1
        if n % 4 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() - t0 >= min_s:
                break
    torch.cuda.synchronize()
    e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(_nvml)
    return float((e1 - e0) / 1000.0 / n)


def _fmt(row: dict) -> str:
    if row["memory"] == "OOM":
        return f"S={row['S']:<3} B={row['B']:<3}  OOM"
    return (
        f"S={row['S']:<3} B={row['B']:<3}  "
        f"lat={row['latency'] * 1e3:7.2f} ms  "
        f"mem={int(row['memory']) / 1e6:7.1f} MB  "
        f"E={row['energy']:7.3f} J  "
        f"P={row['energy'] / row['latency']:5.1f} W  "
        f"FLOPs={row['flops']:.3e}"
    )


def measure(model: Model, image_size: int, batch_size: int, warmup: int = 5, repeats: int = 9):
    model = model.cuda().eval()
    torch.cuda.empty_cache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    try:
        with torch.inference_mode():
            x = torch.randn(batch_size, 3, image_size, image_size, device="cuda")
            for _ in range(warmup):
                model(x)
            torch.cuda.synchronize()
            with FlopCounterMode(display=False) as fc:
                model(x)
            flops = fc.get_total_flops()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(repeats):
                start.record()
                model(x)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) / 1e3)
            latency = statistics.median(times)
            memory = torch.cuda.max_memory_allocated()
            energy = _energy_j(model, x)
            return {
                "flops": flops,
                "latency": latency,
                "memory": memory,
                "energy": energy,
            }
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "flops": None,
            "latency": None,
            "memory": "OOM",
            "energy": None,
        }


def run_grid(out_path: str = "results/measurements.csv", seed: int = 0):
    model = Model().cuda().eval()
    configs = [
        (S, B)
        for S in S_BASE_GRID + S_ADD_GRID
        for B in B_BASE_GRID + B_ADD_GRID + B_OOM_GRID
    ]
    random.Random(seed).shuffle(configs)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["S", "B", "flops", "latency", "memory", "energy", "is_validation"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for S, B in configs:
            row = {
                "S": S,
                "B": B,
                **measure(model, S, B),
                "is_validation": B not in B_OOM_GRID and (S in S_ADD_GRID or B in B_ADD_GRID),
            }
            writer.writerow(row)
            f.flush()
            print(_fmt(row), flush=True)


if __name__ == "__main__":
    run_grid()

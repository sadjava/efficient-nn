"""Wall time versus sequence length (paper Fig. 1 left, Fig. 5-6), reduced width.

Writes speed.json only. Deck figures: make_figures.py
"""

import torch

from pararnn.rnn_cell.rnn_cell_application import RNNCellApplicationMode

from common import CELLS, DEFAULT_OUT, MODES, bench, device_info, make_model, require_cuda, write_json

LENGTHS = (64, 256, 1024, 4096)


def time_cell(name, cell, cfg, *, warmup, reps):
    torch.manual_seed(0)
    model = make_model(cell, cfg, mode="sequential", state_dim=64, input_dim=64, num_heads=4)
    rows = []
    for length in LENGTHS:
        x = torch.randn(2, length, 64, device="cuda")
        for mode in MODES:
            model.mode = RNNCellApplicationMode(mode)

            def step():
                y = model(x)
                torch.sum(y ** 2).backward()
                model.zero_grad(set_to_none=True)

            try:
                ms = bench(step, warmup=warmup, reps=reps)
            except Exception as exc:
                print(name, mode, "L", length, "FAILED", exc)
                rows.append({"mode": mode, "L": length, "failed": f"{type(exc).__name__}: {exc}"})
                continue
            print(f"{name:10s} {mode:16s} L={length:5d}  {ms:.3f} ms")
            rows.append({"mode": mode, "L": length, "ms": ms})
    return rows


def main(out=DEFAULT_OUT, warmup=3, reps=5):
    require_cuda()
    series = {
        name: time_cell(name, cell, cfg, warmup=warmup, reps=reps)
        for name, cell, cfg in CELLS
    }
    payload = {
        "hypothesis": "sequential time grows linearly in L; CUDA and fused stay flatter, so speedup grows with L",
        "device": device_info(),
        "state_dim": 64,
        "batch": 2,
        "num_heads": 4,
        "newton_iterations": 3,
        "warmup": warmup,
        "reps": reps,
        "timing": "median CUDA event, milliseconds, eager mode",
        "cells": series,
    }
    write_json(out / "speed.json", payload)
    return payload


if __name__ == "__main__":
    main()

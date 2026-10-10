"""Parallel modes match sequential forward and backward (paper Sec. 2 and 4)."""

import torch

from pararnn.rnn_cell.rnn_cell_application import RNNCellApplicationMode
from pararnn.rnn_cell.test import check_jacobians

from common import CELLS, DEFAULT_OUT, device_info, make_model, require_cuda, write_json


def _grads(model):
    return [
        None if p.grad is None else p.grad.detach().clone()
        for p in model.parameters()
    ]


def compare_cell(name, cell, cfg, *, seq_length, batch, state_dim, input_dim, num_heads):
    torch.manual_seed(0)
    model = make_model(
        cell, cfg, mode="sequential",
        state_dim=state_dim, input_dim=input_dim, num_heads=num_heads,
    )
    x = torch.randn(batch, seq_length, input_dim, device="cuda", requires_grad=True)
    names = [n for n, _ in model.named_parameters()]
    stored = {}
    for mode in ("sequential", "parallel", "parallel_CUDA", "parallel_FUSED"):
        try:
            model.mode = RNNCellApplicationMode(mode)
            model.zero_grad(set_to_none=True)
            x.grad = None
            y = model(x)
            torch.sum(y ** 2).backward()
            stored[mode] = {
                "y": y.detach(),
                "grads": _grads(model),
                "x_grad": x.grad.detach().clone(),
            }
        except Exception as exc:
            stored[mode] = {"failed": f"{type(exc).__name__}: {exc}"}
            print(name, mode, "FAILED", exc)

    ref = stored["sequential"]
    rows = []
    for mode, got in stored.items():
        if mode == "sequential" or "failed" in got or "failed" in ref:
            if "failed" in got:
                rows.append({"mode": mode, "failed": got["failed"]})
            continue
        grad_err = {}
        for pname, g_ref, g in zip(names, ref["grads"], got["grads"]):
            if g_ref is None or g is None:
                grad_err[pname] = None
            else:
                grad_err[pname] = torch.max(torch.abs(g - g_ref)).item()
        rows.append({
            "mode": mode,
            "output_inf": torch.max(torch.abs(got["y"] - ref["y"])).item(),
            "grad_inf": grad_err,
            "input_grad_inf": torch.max(torch.abs(got["x_grad"] - ref["x_grad"])).item(),
        })
        print(name, mode, "||y||_inf", rows[-1]["output_inf"], "||dx||_inf", rows[-1]["input_grad_inf"])
    return {"cell": name, "modes": rows}


def main(out=DEFAULT_OUT):
    require_cuda()
    spec = dict(seq_length=256, batch=3, state_dim=14, input_dim=6, num_heads=2)
    cells = [compare_cell(name, cell, cfg, **spec) for name, cell, cfg in CELLS]
    jacobians = {
        "ParaGRU": check_jacobians(CELLS[0][1], CELLS[0][2], structure_type="diag"),
        "ParaLSTM": check_jacobians(CELLS[1][1], CELLS[1][2], structure_type="block_diag"),
    }
    payload = {
        "hypothesis": "parallel, parallel_CUDA and parallel_FUSED match sequential fwd/bwd; analytic Jacobians match autograd",
        "device": device_info(),
        "spec": spec,
        "cells": cells,
        "jacobian_inf": jacobians,
    }
    write_json(out / "match.json", payload)
    return payload


if __name__ == "__main__":
    main()

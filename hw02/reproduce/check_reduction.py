"""CUDA parallel reduction matches the PyTorch scan and a sequential solve (Fig. 2 left).

Writes reduction.json only. Deck figures: make_figures.py
"""

import torch

from pararnn.parallel_reduction.parallel_reduction import ParallelSolve

from common import DEFAULT_OUT, bench, device_info, require_cuda, write_json

LENGTHS = (128, 1024, 4096)


def _diag_case(seq_length, batch, hidden):
    jac = torch.randint(0, 2, (batch, hidden, seq_length), device="cuda") * 2.0 - 1.0
    rhs = torch.zeros_like(jac)
    rhs[:, :, 0] = 1.0
    jac[:, :, 0] = 0.0
    sol_cuda = ParallelSolve.parallel_reduce_diag_cuda(jac.clone(), rhs.clone())
    sol_torch = ParallelSolve.parallel_reduce(
        jac.transpose(-1, -2).contiguous(),
        rhs.transpose(-1, -2).contiguous(),
        ParallelSolve._reduction_step_diag,
    ).transpose(-1, -2)
    sol_seq = torch.empty_like(rhs)
    sol_seq[:, :, 0] = rhs[:, :, 0]
    for i in range(1, seq_length):
        sol_seq[:, :, i] = rhs[:, :, i] - jac[:, :, i] * sol_seq[:, :, i - 1]
    return {
        "cuda_torch": torch.max(torch.abs(sol_cuda - sol_torch)).item(),
        "torch_seq": torch.max(torch.abs(sol_torch - sol_seq)).item(),
    }


def _block_case(seq_length, batch, hidden):
    # components_swap: |eigenvalues| = 1, so the state does not explode with L
    signs = torch.randint(0, 2, (batch, hidden, seq_length, 2), device="cuda") * 2.0 - 1.0
    jac = torch.zeros(batch, hidden, seq_length, 2, 2, device="cuda")
    jac[..., 0, 1] = signs[..., 0]
    jac[..., 1, 0] = signs[..., 1]
    rhs = torch.randint(0, 3, (batch, hidden, seq_length, 2), device="cuda") - 1.0
    jac[..., 0, :, :] = 0.0
    sol_cuda = ParallelSolve.parallel_reduce_block_diag_2x2_cuda(jac.clone(), rhs.clone())
    jac_torch = torch.cat(jac.transpose(-3, -4).unbind(dim=-2), dim=-2).contiguous()
    rhs_torch = rhs.permute(0, 2, 3, 1).flatten(-2).contiguous()
    sol_torch = ParallelSolve.parallel_reduce(
        jac_torch, rhs_torch, ParallelSolve._reduction_step_block_diag,
    )
    sol_torch = sol_torch.reshape(batch, seq_length, 2, hidden).transpose(-1, -2).transpose(1, 2)
    sol_seq = torch.empty_like(rhs)
    sol_seq[:, :, 0] = rhs[:, :, 0]
    for i in range(1, seq_length):
        sol_seq[:, :, i] = rhs[:, :, i] - torch.einsum("bnij,bnj->bni", jac[:, :, i], sol_seq[:, :, i - 1])
    return {
        "cuda_torch": torch.max(torch.abs(sol_cuda - sol_torch)).item(),
        "torch_seq": torch.max(torch.abs(sol_torch - sol_seq)).item(),
    }


def _time_diag(seq_length, batch, hidden):
    jac = (torch.randint(0, 2, (batch, hidden, seq_length), device="cuda") * 2.0 - 1.0)
    rhs = torch.zeros_like(jac)
    rhs[:, :, 0] = 1.0
    jac[:, :, 0] = 0.0
    jac_t = jac.transpose(-1, -2).contiguous()
    rhs_t = rhs.transpose(-1, -2).contiguous()
    return {
        "cuda": bench(lambda: ParallelSolve.parallel_reduce_diag_cuda(jac, rhs)),
        "torch": bench(lambda: ParallelSolve.parallel_reduce(jac_t, rhs_t, ParallelSolve._reduction_step_diag)),
    }


def main(out=DEFAULT_OUT):
    require_cuda()
    torch.manual_seed(0)
    errors = {"diag": [], "block_diag_2x2": []}
    for length in LENGTHS:
        diag = _diag_case(length, batch=4, hidden=64)
        block = _block_case(length, batch=2, hidden=32)
        errors["diag"].append({"L": length, **diag})
        errors["block_diag_2x2"].append({"L": length, **block})
        print("diag", length, diag)
        print("block", length, block)
    timings = []
    for length in LENGTHS:
        row = {"L": length, **_time_diag(length, batch=4, hidden=64)}
        timings.append(row)
        print("diag timing", row)

    payload = {
        "hypothesis": "CUDA parallel reduction matches the PyTorch reduction and a sequential linear recurrence",
        "device": device_info(),
        "errors": errors,
        "diag_timings_ms": timings,
    }
    write_json(out / "reduction.json", payload)
    return payload


if __name__ == "__main__":
    main()

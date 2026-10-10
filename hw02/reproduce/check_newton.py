"""Newton residual collapses by iteration 3, for several sequence lengths (App. A, Fig. 4 left).

Writes newton.json only. Deck figures: make_figures.py
"""

import torch

from pararnn.rnn_cell.rnn_cell_application import RNNCellApplicationMode

from common import CELLS, DEFAULT_OUT, device_info, make_model, require_cuda, write_json

LENGTHS = (128, 512, 2048)
ITERS = (1, 2, 3, 4)


def cell_curve(name, cell, cfg):
    torch.manual_seed(0)
    model = make_model(cell, cfg, mode="sequential", state_dim=64, input_dim=64, num_heads=4)
    curves = {}
    for length in LENGTHS:
        x = torch.randn(4, length, 64, device="cuda")
        model.mode = RNNCellApplicationMode.SEQUENTIAL
        with torch.no_grad():
            y_seq = model(x)
        model.mode = RNNCellApplicationMode.PARALLEL_CUDA
        curves[str(length)] = {}
        for nits in ITERS:
            model.newton_config.max_its = nits
            with torch.no_grad():
                y = model(x)
            err = torch.max(torch.abs(y - y_seq)).item()
            curves[str(length)][str(nits)] = err
            print(name, "L", length, "nits", nits, "err", err)
    return curves


def main(out=DEFAULT_OUT):
    require_cuda()
    curves = {name: cell_curve(name, cell, cfg) for name, cell, cfg in CELLS}
    payload = {
        "hypothesis": "3 Newton iterations match the sequential recurrence; extra length does not change that",
        "device": device_info(),
        "mode": "parallel_CUDA",
        "state_dim": 64,
        "batch": 4,
        "curves": curves,
    }
    write_json(out / "newton.json", payload)
    return payload


if __name__ == "__main__":
    main()

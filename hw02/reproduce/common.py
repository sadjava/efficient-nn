import json
from pathlib import Path

import torch

from pararnn.parallel_reduction.parallel_reduction import NewtonConfig
from pararnn.rnn_cell.gru_diag_mh import GRUDiagMH, GRUDiagMHConfig
from pararnn.rnn_cell.lstm_cifg_diag_mh import LSTMCIFGDiagMH, LSTMCIFGDiagMHConfig
from pararnn.rnn_cell.rnn_cell_application import RNNCellApplicationMode

CELLS = (
    ("ParaGRU", GRUDiagMH, GRUDiagMHConfig),
    ("ParaLSTM", LSTMCIFGDiagMH, LSTMCIFGDiagMHConfig),
)
MODES = ("sequential", "parallel", "parallel_CUDA", "parallel_FUSED")
HW02 = Path(__file__).resolve().parents[1]
DEFAULT_OUT = HW02 / "results"


def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")


def device_info():
    props = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_memory_gb": round(props.total_memory / 1024**3, 2),
        "torch": torch.__version__,
    }


def make_model(cell, cfg, *, mode, state_dim, input_dim, num_heads, max_its=3):
    model = cell(cfg(
        state_dim=state_dim,
        input_dim=input_dim,
        num_heads=num_heads,
        device="cuda",
        mode=mode,
        newton_config=NewtonConfig(max_its=max_its),
    ))
    model.mode = RNNCellApplicationMode(mode)
    return model


def bench(fn, warmup=3, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
    print("wrote", path)


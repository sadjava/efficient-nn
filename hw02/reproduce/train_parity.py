"""Prefix-parity training: the Table 1 expressivity check, at a size that fits one GPU.

The paper trains only the recurrent cell on Parity (L=100, up to 3000 epochs, 3 seeds)
and reports ParaGRU / ParaLSTM at 100%, with Mamba2 and a Transformer near chance.
Here L=16 and each position is labeled with the parity of its prefix, so the run
finishes in about a minute. The reported accuracy is the last position: parity of
the whole sequence, which is the paper's label. A diagonal linear RNN stands in for
the linear recurrence that cannot compute parity.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CELLS, DEFAULT_OUT, device_info, make_model, require_cuda, write_json
import make_figures


class LinearRNN(nn.Module):
    """h_t = a * h_{t-1} + B x_t. Affine in the inputs, so it cannot compute parity."""

    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.a = nn.Parameter(torch.full((state_dim,), 0.5))
        self.B = nn.Linear(input_dim, state_dim, bias=False)

    def forward(self, x):
        h = x.new_zeros(x.shape[0], self.a.shape[0])
        out = []
        for t in range(x.shape[1]):
            h = self.a * h + self.B(x[:, t])
            out.append(h)
        return torch.stack(out, 1)


class ParityNet(nn.Module):
    def __init__(self, mixer, input_dim, state_dim):
        super().__init__()
        self.embed = nn.Embedding(2, input_dim)
        self.mixer = mixer
        self.read = nn.Linear(state_dim, 2)

    def forward(self, tokens):
        return self.read(self.mixer(self.embed(tokens)))


def prefix_parity(tokens):
    return torch.cumsum(tokens, dim=1) % 2


def make_batch(batch, length, generator):
    tokens = torch.randint(0, 2, (batch, length), device="cuda", generator=generator)
    return tokens, prefix_parity(tokens)


def evaluate(model, tokens, labels):
    model.eval()
    correct = last = total = 0
    with torch.no_grad():
        for i in range(0, tokens.shape[0], 256):
            logits = model(tokens[i:i + 256])
            pred = logits.argmax(-1)
            y = labels[i:i + 256]
            correct += (pred == y).sum().item()
            last += (pred[:, -1] == y[:, -1]).sum().item()
            total += y.numel()
    model.train()
    n_seq = tokens.shape[0]
    return correct / total, last / n_seq


def train(model, *, steps, batch, length, lr, eval_every, test_tokens, test_labels, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    history = []
    model.train()
    for step in range(1, steps + 1):
        tokens, labels = make_batch(batch, length, gen)
        loss = F.cross_entropy(model(tokens).reshape(-1, 2), labels.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            acc, last = evaluate(model, test_tokens, test_labels)
            row = {"step": step, "loss": loss.item(), "acc": acc, "last_acc": last}
            history.append(row)
            print(f"  step {step:4d}  loss {row['loss']:.3f}  acc {acc:.3f}  last {last:.3f}")
            if acc >= 0.99:
                break
    return history


def build(kind, *, dim, heads, mode, max_its):
    if kind == "LinearRNN":
        mixer = LinearRNN(dim, dim).cuda()
    else:
        cell, cfg = {"ParaGRU": CELLS[0][1:], "ParaLSTM": CELLS[1][1:]}[kind]
        mixer = make_model(cell, cfg, mode=mode, state_dim=dim, input_dim=dim, num_heads=heads, max_its=max_its)
    return ParityNet(mixer, dim, dim).cuda()



def position_accuracy(model, tokens, labels):
    model.eval()
    with torch.no_grad():
        pred = model(tokens).argmax(-1)
    acc = (pred == labels).float().mean(0).tolist()
    model.train()
    return acc



def trace_rows(models, length):
    """One fixed string with every model's per-position prediction (for the figures)."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(2)
    tokens, labels = make_batch(1, length, gen)
    with torch.no_grad():
        pred = {n: m(tokens).argmax(-1)[0].tolist() for n, m in models.items()}
    return {"tokens": tokens[0].tolist(), "labels": labels[0].tolist(), "pred": pred}



def same_init_losses(dim, heads, length, steps, batch):
    """Same weights and batches, sequential versus parallel_CUDA. Losses should track."""
    torch.manual_seed(0)
    seq = build("ParaGRU", dim=dim, heads=heads, mode="sequential", max_its=3)
    par = build("ParaGRU", dim=dim, heads=heads, mode="parallel_CUDA", max_its=3)
    par.load_state_dict(seq.state_dict())
    opt_s = torch.optim.AdamW(seq.parameters(), lr=1e-3, weight_decay=1e-6)
    opt_p = torch.optim.AdamW(par.parameters(), lr=1e-3, weight_decay=1e-6)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(3)
    rows = []
    for step in range(1, steps + 1):
        tokens, labels = make_batch(batch, length, gen)
        losses = []
        for model, opt in ((seq, opt_s), (par, opt_p)):
            loss = F.cross_entropy(model(tokens).reshape(-1, 2), labels.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        rows.append({"step": step, "sequential": losses[0], "parallel_CUDA": losses[1]})
    print(f"  seq vs parallel final loss {rows[-1]['sequential']:.4f} vs {rows[-1]['parallel_CUDA']:.4f}")
    return rows



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=DEFAULT_OUT.__class__, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()
    require_cuda()
    args.out.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device="cuda")
    gen.manual_seed(1)
    test_tokens, test_labels = make_batch(2048, args.length, gen)

    histories = {}
    trained = {}
    for kind in ("LinearRNN", "ParaGRU", "ParaLSTM"):
        print("==", kind)
        torch.manual_seed(0)
        model = build(kind, dim=args.dim, heads=args.heads, mode="parallel_CUDA", max_its=3)
        histories[kind] = train(
            model, steps=args.steps, batch=args.batch, length=args.length, lr=args.lr,
            eval_every=25, test_tokens=test_tokens, test_labels=test_labels, seed=10,
        )
        trained[kind] = model

    print("== sequential vs parallel_CUDA")
    mode_rows = same_init_losses(args.dim, args.heads, length=32, steps=40, batch=32)

    finals = {name: hist[-1] for name, hist in histories.items()}
    payload = {
        "hypothesis": "ParaGRU and ParaLSTM learn sequence parity; a diagonal linear RNN stays at chance",
        "paper": "Table 1 Parity, cell only. This run uses prefix labels, one seed, and a diagonal linear RNN instead of Mamba2.",
        "device": device_info(),
        "length": args.length,
        "dim": args.dim,
        "heads": args.heads,
        "batch": args.batch,
        "lr": args.lr,
        "steps_budget": args.steps,
        "final": finals,
        "by_position": {name: position_accuracy(model, test_tokens, test_labels) for name, model in trained.items()},
        "trace": trace_rows(trained, args.length),
        "histories": histories,
        "mode_losses": mode_rows,
    }
    write_json(args.out / "parity.json", payload)
    make_figures.main(args.out)

    lines = [
        "Parity task (prefix labels, lightweight stand-in for Table 1)",
        f"device: {payload['device']['name']}",
        f"L={args.length} dim={args.dim} heads={args.heads} batch={args.batch} lr={args.lr}",
        "",
    ]
    for name, row in finals.items():
        lines.append(
            f"  {name:12s} sequence acc {row['last_acc']:.3f}  "
            f"mean-over-positions {row['acc']:.3f}  at step {row['step']}"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    summary = args.out / "summary.txt"
    prev = summary.read_text() if summary.exists() else ""
    marker = "Parity task"
    if marker in prev:
        prev = prev.split(marker)[0].rstrip() + "\n\n"
    summary.write_text(prev + text)


if __name__ == "__main__":
    main()

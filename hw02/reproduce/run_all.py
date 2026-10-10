"""Run every reproduction check and write hw02/results.

JSON from the checks; themed PNGs for the deck via make_figures.py.
"""

import argparse
from pathlib import Path

from common import DEFAULT_OUT

import check_match
import check_newton
import check_reduction
import check_speed
import make_figures


def _lines(payload):
    lines = [
        payload["hypothesis"],
        f"device: {payload['device']['name']}",
        "",
    ]
    return lines


def summary(out, match, newton, speed, reduction):
    lines = ["ParaRNN reproduction, lightweight GPU setup", ""]
    lines += _lines(match)
    for cell in match["cells"]:
        lines.append(cell["cell"])
        for row in cell["modes"]:
            if "failed" in row:
                lines.append(f"  {row['mode']}: FAILED {row['failed']}")
            else:
                lines.append(
                    f"  {row['mode']}: output {row['output_inf']:.3e}, "
                    f"input grad {row['input_grad_inf']:.3e}, "
                    f"param grads {row['grad_inf']}"
                )
    lines.append(f"  jacobian inf: {match['jacobian_inf']}")
    lines += ["", *_lines(newton)]
    for name, series in newton["curves"].items():
        for length, by_it in series.items():
            lines.append(f"  {name} L={length}: {by_it}")
    lines += ["", *_lines(speed)]
    for name, rows in speed["cells"].items():
        lines.append(name)
        for row in rows:
            if "ms" in row:
                lines.append(f"  {row['mode']:16s} L={row['L']:<5d} {row['ms']:.3f} ms")
            else:
                lines.append(f"  {row['mode']:16s} L={row['L']:<5d} FAILED")
    lines += ["", *_lines(reduction)]
    for kind, rows in reduction["errors"].items():
        for row in rows:
            lines.append(
                f"  {kind} L={row['L']}: cuda-torch {row['cuda_torch']:.3e}, torch-seq {row['torch_seq']:.3e}"
            )
    text = "\n".join(lines) + "\n"
    path = Path(out) / "summary.txt"
    path.write_text(text)
    print(text)
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-speed", action="store_true", help="skip the L sweep; it is the slow part")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    match = check_match.main(args.out)
    newton = check_newton.main(args.out)
    reduction = check_reduction.main(args.out)
    speed = {"hypothesis": "skipped", "device": match["device"], "cells": {}}
    if not args.skip_speed:
        speed = check_speed.main(args.out)
    summary(args.out, match, newton, speed, reduction)
    make_figures.main(args.out)


if __name__ == "__main__":
    main()

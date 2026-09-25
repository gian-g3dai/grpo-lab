"""Plot training curves from a TRL trainer_state.json.

Usage:
    python -m grpo_lab.plot --state outputs/x/final/trainer_state.json --out results/x/curves.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ema(xs, alpha=0.1):
    out, s = [], None
    for x in xs:
        s = x if s is None else alpha * x + (1 - alpha) * s
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None, help="also dump the raw log history as CSV")
    args = ap.parse_args()

    logs = [l for l in json.load(open(args.state))["log_history"] if "step" in l and "reward" in l]
    steps = [l["step"] for l in logs]

    if args.csv:
        keys = sorted({k for l in logs for k in l})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(logs)

    panels = [
        ("reward", "Total reward (mean over group)"),
        ("rewards/correctness_reward/mean", "Correctness reward"),
        ("rewards/format_reward/mean", "Format reward"),
        ("completions/mean_length", "Mean completion length (tokens)"),
    ]
    panels = [(k, t) for k, t in panels if any(k in l for l in logs)]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (key, title) in zip(axes, panels):
        ys = [l.get(key) for l in logs]
        xs = [s for s, y in zip(steps, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        ax.plot(xs, ys, color="#9ab", lw=0.8, alpha=0.6, label="raw")
        ax.plot(xs, _ema(ys), color="#c33", lw=1.8, label="EMA")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

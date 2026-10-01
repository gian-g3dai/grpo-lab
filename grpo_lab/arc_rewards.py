"""Reward functions for GRPO on ARC (direct grid output).

The dataset column ``answer`` holds the gold output grid as text (see ``arc.grid_to_text``).

* ``arc_exact``    1.0 if the parsed grid equals the gold grid. The real objective.
* ``arc_partial``  0.5 * fraction of matching cells, when the shape is right. Dense signal
                   so a group of 8 wrong answers still has a gradient (GRPO needs reward
                   variance inside a group). Capped well below ``arc_exact`` so an exact
                   solution always dominates.
* ``arc_format``   0.1 if the completion has exactly one code block and stops right after
                   it (no hedging with several grids, no rambling past the answer).
"""

from __future__ import annotations

from .arc import _FENCE_RE, cell_accuracy, grid_to_text, parse_grid


def _text(completion) -> str:
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


def _gold(answer: str) -> list[list[int]]:
    return [[int(ch) for ch in row] for row in answer.split("\n")]


def arc_exact(completions, answer, **kwargs) -> list[float]:
    out = []
    for c, gold in zip(completions, answer):
        pred = parse_grid(_text(c))
        out.append(1.0 if pred is not None and grid_to_text(pred) == gold else 0.0)
    return out


def arc_partial(completions, answer, **kwargs) -> list[float]:
    return [0.5 * cell_accuracy(parse_grid(_text(c)), _gold(gold)) for c, gold in zip(completions, answer)]


def arc_format(completions, **kwargs) -> list[float]:
    out = []
    for c in completions:
        text = _text(c)
        blocks = list(_FENCE_RE.finditer(text))
        ok = len(blocks) == 1 and len(text) - blocks[0].end() < 20 and parse_grid(text) is not None
        out.append(0.1 if ok else 0.0)
    return out


ARC_REWARD_FUNCS = {
    "arc_exact": arc_exact,
    "arc_partial": arc_partial,
    "arc_format": arc_format,
}

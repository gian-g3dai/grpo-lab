"""Reward functions for GRPO on GSM8K.

TRL calls each reward function with keyword arguments ``prompts``,
``completions`` and every extra dataset column (here: ``answer``). Each must
return one float per completion.

Two signals are used:

* ``correctness_reward`` -- 1.0 if the number in the last ``\\boxed{}`` equals the gold
  answer, else 0.0. This is the real objective.
* ``format_reward`` -- 0.2 if the completion contains exactly one ``\\boxed{}`` and ends
  shortly after it (i.e. the model stopped instead of rambling or being truncated).
  A small dense-ish signal that discourages hedging with several boxed answers and
  pushes toward terminating, which also makes rollouts cheaper.
"""

from __future__ import annotations

from .data import normalize_number


def _text(completion) -> str:
    # Conversational datasets yield a list of messages; plain datasets yield a str.
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


def find_boxed(text: str) -> list[tuple[str, int]]:
    """Return (content, end_index) for every \\boxed{...} with balanced braces."""
    out, i = [], 0
    while True:
        i = text.find("\\boxed{", i)
        if i < 0:
            return out
        j, depth = i + len("\\boxed{"), 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            out.append((text[i + len("\\boxed{") : j - 1], j))
        i = j


def extract_answer(text: str) -> str | None:
    boxes = find_boxed(text)
    if not boxes:
        return None
    return normalize_number(boxes[-1][0])


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    out = []
    for c, gold in zip(completions, answer):
        pred = extract_answer(_text(c))
        out.append(1.0 if pred is not None and pred == normalize_number(gold) else 0.0)
    return out


def format_reward(completions, **kwargs) -> list[float]:
    out = []
    for c in completions:
        text = _text(c)
        boxes = find_boxed(text)
        ok = len(boxes) == 1 and len(text) - boxes[0][1] < 80
        out.append(0.2 if ok else 0.0)
    return out


REWARD_FUNCS = {
    "correctness": correctness_reward,
    "format": format_reward,
}

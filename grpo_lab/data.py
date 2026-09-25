"""GSM8K loading and prompt construction.

The model is asked to reason step by step and put the final numeric answer in
``\\boxed{}``. Qwen-style instruct models already do this natively, so the
baseline is meaningful and GRPO has to improve *correctness*, not just teach a
format. The reward parses the last ``\\boxed{...}`` and compares numbers.
"""

from __future__ import annotations

import re

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = (
    "Solve the math problem step by step, showing your reasoning briefly. "
    "Finish with the final numeric answer inside \\boxed{}, e.g. \\boxed{42}."
)

_GOLD_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_gold(answer_field: str) -> str:
    """GSM8K stores the gold answer after '####' at the end of the solution."""
    m = _GOLD_RE.search(answer_field)
    if not m:
        raise ValueError(f"no gold answer in: {answer_field!r}")
    return normalize_number(m.group(1))


def normalize_number(s: str) -> str | None:
    """Pull the first number out of a string and canonicalise it (1,200 -> 1200; 3.0 -> 3)."""
    s = re.sub(r"\\text\{[^}]*\}", " ", s)  # \text{dollars}
    s = s.replace(",", "").replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\,", "").replace("\\!", "")
    m = _NUM_RE.search(s)
    if not m:
        return None
    f = float(m.group(0))
    return str(int(f)) if f.is_integer() else str(f)


def build_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def load_gsm8k(split: str, limit: int | None = None, seed: int = 0) -> Dataset:
    """Return a dataset with columns ``prompt`` (chat messages) and ``answer`` (gold string)."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None and limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
    return ds.map(
        lambda ex: {"prompt": build_prompt(ex["question"]), "answer": extract_gold(ex["answer"])},
        remove_columns=ds.column_names,
    )

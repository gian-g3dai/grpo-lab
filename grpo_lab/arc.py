"""ARC-AGI as a verifiable task: direct grid output.

Each prompt shows a few input/output grid pairs of one ARC task and one more
input; the model must produce the matching output grid. Grids are written as
rows of digits (Qwen tokenises every digit separately, so one cell = one token).

Three data sources, all plain JSON on disk (see ``scripts/get_arc_data.sh``):

* ``data/arc-agi-1/data/{training,evaluation}``   the official ARC-AGI-1 tasks
* ``data/arc-agi-2/data/{training,evaluation}``   the official ARC-AGI-2 tasks
* ``data/re_arc/tasks``                           re-arc: 1000 generated, verified
                                                  examples per ARC-1 training task

Training examples are built from re-arc: for one task, ``n_demos`` demo pairs plus
one query pair, all fresh samples. A fixed slice of every task's samples is held
out so "fresh samples of the trained tasks" can be scored after training.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from datasets import Dataset

Grid = list[list[int]]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_PROMPT = (
    "You are solving an ARC puzzle. Each grid is written as rows of digits 0-9, one row per line. "
    "The examples share one hidden transformation rule. Infer the rule from the examples and apply "
    "it to the test input. You may think briefly, then give the output grid, and nothing else, "
    "inside a ```grid code block."
)


# --------------------------------------------------------------------------- grids as text


def grid_to_text(grid: Grid) -> str:
    return "\n".join("".join(str(c) for c in row) for row in grid)


_FENCE_RE = re.compile(r"```(?:grid)?[ \t]*\n(.*?)```", re.DOTALL)
_ROW_RE = re.compile(r"^[0-9]+$")


def parse_grid(text: str) -> Grid | None:
    """Return the grid in the *last* fenced code block, or None if there is no valid one.

    Rows may be written with or without spaces/commas between digits. A grid is valid if it
    is non-empty, rectangular, and at most 30x30 (the ARC limit).
    """
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        return None
    rows = []
    for line in blocks[-1].strip().splitlines():
        line = re.sub(r"[ ,\[\]]", "", line.strip())
        if not line:
            continue
        if not _ROW_RE.match(line):
            return None
        rows.append([int(ch) for ch in line])
    if not rows or len(rows) > 30:
        return None
    w = len(rows[0])
    if w > 30 or any(len(r) != w for r in rows):
        return None
    return rows


def grid_blocks(text: str) -> int:
    """How many fenced code blocks the completion contains (format reward)."""
    return len(_FENCE_RE.findall(text))


def cell_accuracy(pred: Grid | None, gold: Grid) -> float:
    """Fraction of matching cells if the shapes agree, else 0."""
    if pred is None or len(pred) != len(gold) or len(pred[0]) != len(gold[0]):
        return 0.0
    total = len(gold) * len(gold[0])
    hits = sum(p == g for pr, gr in zip(pred, gold) for p, g in zip(pr, gr))
    return hits / total


# --------------------------------------------------------------------------- prompts


def build_prompt(demos: list[dict], test_input: Grid) -> list[dict[str, str]]:
    parts = []
    for i, pair in enumerate(demos, 1):
        parts.append(f"Example {i}\nInput:\n{grid_to_text(pair['input'])}\nOutput:\n{grid_to_text(pair['output'])}")
    parts.append(f"Test\nInput:\n{grid_to_text(test_input)}\nOutput:")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def n_cells(grid: Grid) -> int:
    return len(grid) * len(grid[0])


def prompt_cells(demos: list[dict], test_input: Grid) -> int:
    """Rough token count of a prompt: one token per cell (plus one per row, counted elsewhere)."""
    return sum(n_cells(p["input"]) + n_cells(p["output"]) for p in demos) + n_cells(test_input)


# --------------------------------------------------------------------------- official tasks


def load_arc_tasks(split: str, version: int = 1, data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """``split`` is 'training' or 'evaluation'. Returns {task_id: {'train': [...], 'test': [...]}}."""
    d = data_dir / f"arc-agi-{version}" / "data" / split
    if not d.is_dir():
        raise FileNotFoundError(f"{d} missing: run scripts/get_arc_data.sh")
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}


def select_task_ids(n_tasks: int | None, select: str = "smallest", data_dir: Path = DATA_DIR) -> list[str]:
    """Pick which ARC-1 training tasks to train on.

    'smallest' sorts the 400 tasks by the median size (input + output cells) of their re-arc
    samples, so the chosen tasks have short prompts *in the generated data* (the original
    task's grids are a poor proxy: re-arc varies grid sizes a lot). 'all' takes every task in
    id order.
    """
    ids = sorted(load_arc_tasks("training", 1, data_dir))
    if select == "smallest":

        def median_cells(task_id: str) -> int:
            sizes = sorted(n_cells(s["input"]) + n_cells(s["output"]) for s in load_rearc_task(task_id, data_dir))
            return sizes[len(sizes) // 2]

        ids.sort(key=lambda t: (median_cells(t), t))
    elif select != "all":
        raise ValueError(f"unknown select={select!r}")
    return ids if n_tasks is None else ids[:n_tasks]


# --------------------------------------------------------------------------- re-arc


def load_rearc_task(task_id: str, data_dir: Path = DATA_DIR) -> list[dict]:
    p = data_dir / "re_arc" / "tasks" / f"{task_id}.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing: run scripts/get_arc_data.sh")
    return json.loads(p.read_text())


def split_rearc(task_id: str, holdout_per_task: int, seed: int, data_dir: Path = DATA_DIR) -> tuple[list[dict], list[dict]]:
    """Deterministically split a task's re-arc samples into (train_pool, holdout)."""
    samples = load_rearc_task(task_id, data_dir)
    rng = random.Random(f"{seed}:{task_id}")
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    holdout = [samples[i] for i in idx[:holdout_per_task]]
    pool = [samples[i] for i in idx[holdout_per_task:]]
    return pool, holdout


def _example(task_id: str, demos: list[dict], query: dict) -> dict:
    return {
        "prompt": build_prompt(demos, query["input"]),
        "answer": grid_to_text(query["output"]),
        "task_id": task_id,
    }


def _fits(demos: list[dict], query: dict, max_prompt_cells: int, max_output_cells: int) -> bool:
    return prompt_cells(demos, query["input"]) <= max_prompt_cells and n_cells(query["output"]) <= max_output_cells


def build_rearc_dataset(
    task_ids: list[str],
    samples_per_task: int,
    n_demos: int = 3,
    holdout_per_task: int = 20,
    max_prompt_cells: int = 2000,
    max_output_cells: int = 400,
    seed: int = 0,
    data_dir: Path = DATA_DIR,
) -> Dataset:
    """Training set: ``samples_per_task`` prompts per task, each with fresh re-arc demos + query.

    Oversized candidates (long prompt or big output grid) are skipped, so a task with
    mostly large grids may contribute fewer than ``samples_per_task`` examples.
    """
    rows = []
    for task_id in task_ids:
        pool, _ = split_rearc(task_id, holdout_per_task, seed, data_dir)
        rng = random.Random(f"{seed}:{task_id}:train")
        made, tries = 0, 0
        while made < samples_per_task and tries < samples_per_task * 10:
            tries += 1
            picks = rng.sample(pool, n_demos + 1)
            demos, query = picks[:-1], picks[-1]
            if not _fits(demos, query, max_prompt_cells, max_output_cells):
                continue
            rows.append(_example(task_id, demos, query))
            made += 1
    rng = random.Random(seed)
    rng.shuffle(rows)
    return Dataset.from_list(rows)


def build_rearc_holdout(
    task_ids: list[str],
    n_demos: int = 3,
    holdout_per_task: int = 20,
    max_prompt_cells: int = 2000,
    max_output_cells: int = 400,
    seed: int = 0,
    data_dir: Path = DATA_DIR,
) -> list[dict]:
    """Eval set: every held-out sample of every task is a query, demos drawn from the train pool."""
    rows = []
    for task_id in task_ids:
        pool, holdout = split_rearc(task_id, holdout_per_task, seed, data_dir)
        rng = random.Random(f"{seed}:{task_id}:holdout")
        for query in holdout:
            demos = rng.sample(pool, n_demos)
            if not _fits(demos, query, max_prompt_cells, max_output_cells):
                continue
            rows.append(_example(task_id, demos, query))
    return rows


def build_official_eval(
    split: str,
    version: int = 1,
    task_ids: list[str] | None = None,
    max_prompt_cells: int | None = None,
    data_dir: Path = DATA_DIR,
) -> list[dict]:
    """One row per *test pair* of each official task (a few tasks have 2-3 test pairs)."""
    tasks = load_arc_tasks(split, version, data_dir)
    ids = task_ids if task_ids is not None else sorted(tasks)
    rows = []
    for task_id in ids:
        task = tasks[task_id]
        for k, pair in enumerate(task["test"]):
            if max_prompt_cells is not None and prompt_cells(task["train"], pair["input"]) > max_prompt_cells:
                continue
            row = _example(task_id, task["train"], pair)
            row["test_index"] = k
            rows.append(row)
    return rows


def load_arc_train(data_cfg: dict, seed: int) -> Dataset:
    """Entry point used by ``train.py`` when the config says ``task: arc``."""
    ids = select_task_ids(data_cfg.get("n_tasks"), data_cfg.get("select", "smallest"))
    return build_rearc_dataset(
        ids,
        samples_per_task=data_cfg.get("samples_per_task", 200),
        n_demos=data_cfg.get("n_demos", 3),
        holdout_per_task=data_cfg.get("holdout_per_task", 20),
        max_prompt_cells=data_cfg.get("max_prompt_cells", 2000),
        max_output_cells=data_cfg.get("max_output_cells", 400),
        seed=seed,
    )

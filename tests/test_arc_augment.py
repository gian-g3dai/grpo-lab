import json
import random

from grpo_lab.arc import answer_to_completion, build_rearc_dataset, build_rearc_holdout, parse_grid
from grpo_lab.arc_augment import (
    DIHEDRAL_NAMES,
    Augmentation,
    all_dihedral,
    inverse_color_perm,
    inverse_transform_grid,
    random_augmentation,
    random_color_perm,
    transform_grid,
)

G = [[1, 2, 3], [4, 5, 6]]  # 2x3, no symmetry


def test_dihedral_count_and_inverse():
    views = {json.dumps(transform_grid(G, n)) for n in DIHEDRAL_NAMES}
    assert len(views) == 8
    for n in DIHEDRAL_NAMES:
        assert inverse_transform_grid(transform_grid(G, n), n) == G


def test_dihedral_values():
    assert transform_grid(G, "rot90") == [[4, 1], [5, 2], [6, 3]]
    assert transform_grid(G, "rot180") == [[6, 5, 4], [3, 2, 1]]
    assert transform_grid(G, "rot270") == [[3, 6], [2, 5], [1, 4]]
    assert transform_grid(G, "flip_h") == [[3, 2, 1], [6, 5, 4]]
    assert transform_grid(G, "flip_v") == [[4, 5, 6], [1, 2, 3]]
    assert transform_grid(G, "transpose") == [[1, 4], [2, 5], [3, 6]]
    assert transform_grid(G, "anti_transpose") == [[6, 3], [5, 2], [4, 1]]
    assert transform_grid(G, "identity") == G and transform_grid(G, "identity") is not G


def test_color_perm():
    rng = random.Random(0)
    perm = random_color_perm(rng, keep_background=True)
    assert perm[0] == 0 and sorted(perm) == list(range(10))
    inv = inverse_color_perm(perm)
    assert all(inv[perm[c]] == c for c in range(10))
    perm_all = random_color_perm(random.Random(1), keep_background=False)
    assert sorted(perm_all) == list(range(10))


def test_augmentation_roundtrip_and_pairs():
    aug = Augmentation("rot90", random_color_perm(random.Random(3)))
    pairs = [{"input": G, "output": [[0, 7], [7, 0]]}]
    out = aug.apply_pairs(pairs)
    assert out[0]["input"] == aug.apply(G) and out[0]["output"] == aug.apply(pairs[0]["output"])
    assert aug.invert(out[0]["input"]) == G
    assert not aug.is_identity and Augmentation().is_identity
    assert len(all_dihedral()) == 8


def test_random_augmentation_respects_config():
    rng = random.Random(0)
    assert random_augmentation(rng, None).is_identity
    a = random_augmentation(rng, {"dihedral": False, "color_permutation": True})
    assert a.dihedral == "identity" and a.color_perm is not None
    b = random_augmentation(rng, {"dihedral": True, "color_permutation": False})
    assert b.color_perm is None
    names = {random_augmentation(rng, {"dihedral": True}).dihedral for _ in range(200)}
    assert names == set(DIHEDRAL_NAMES)


def test_completion_format_roundtrip():
    assert parse_grid(answer_to_completion("12\n34")) == [[1, 2], [3, 4]]


def _fake_rearc(tmp_path, task_id="t1", n=40):
    d = tmp_path / "re_arc" / "tasks"
    d.mkdir(parents=True)
    rng = random.Random(0)
    samples = []
    for _ in range(n):  # rule: output = input with 1 -> 2
        inp = [[rng.choice([0, 1]) for _ in range(3)] for _ in range(2)]
        samples.append({"input": inp, "output": [[2 if c == 1 else c for c in row] for row in inp]})
    (d / f"{task_id}.json").write_text(json.dumps(samples))
    return tmp_path


def test_build_rearc_dataset_augmented(tmp_path):
    data_dir = _fake_rearc(tmp_path)
    plain = build_rearc_dataset(["t1"], samples_per_task=30, n_demos=2, holdout_per_task=5, seed=0, data_dir=data_dir)
    aug = build_rearc_dataset(
        ["t1"], samples_per_task=30, n_demos=2, holdout_per_task=5, seed=0,
        augment={"dihedral": True, "color_permutation": True}, data_dir=data_dir,
    )
    assert len(plain) == len(aug) == 30
    assert set(plain.column_names) == {"prompt", "answer", "task_id"}
    # in the plain set every gold grid is 2x3 and uses colours {0, 2}; augmentation rotates and relabels
    assert all(len(a.split("\n")) == 2 for a in plain["answer"])
    shapes = {len(a.split("\n")) for a in aug["answer"]}
    colours = {c for a in aug["answer"] for c in a.replace("\n", "")}
    assert shapes == {2, 3} and colours - {"0", "2"}
    # the same view is applied to demos and query: colour 1 never appears next to colour 2 in a plain prompt,
    # and an augmented prompt's user message still shows its demo outputs with the query's colour mapping
    for row in aug:
        user = row["prompt"][1]["content"]
        assert user.count("Example") == 2 and user.endswith("Output:")


def test_n_demos_range(tmp_path):
    data_dir = _fake_rearc(tmp_path)
    ds = build_rearc_dataset(["t1"], samples_per_task=40, n_demos=[1, 3], holdout_per_task=5, seed=0, data_dir=data_dir)
    counts = {r["prompt"][1]["content"].count("Example ") for r in ds}
    assert counts == {1, 2, 3}
    hold = build_rearc_holdout(["t1"], n_demos=[1, 3], holdout_per_task=5, seed=0, data_dir=data_dir)
    assert len(hold) == 5
    assert {r["prompt"][1]["content"].count("Example ") for r in hold} <= {1, 2, 3}

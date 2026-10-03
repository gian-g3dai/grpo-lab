"""Augmentations for ARC grids.

An ARC rule is invariant to how the grid is drawn: rotate or mirror every grid of a task, or
relabel its colours consistently, and the same rule still maps each input to its output. Every
winning ARC-Prize entry trains on these views (the ARChitects, MindsAI, the MIT test-time-training
paper), because it is what turns "memorise these 400 tasks" into "learn rules that survive a
change of perspective". The same transforms also serve test-time training and inference-time
voting, which is why every transform here has an inverse.

* dihedral group D4: the 8 symmetries of a square (identity, 3 rotations, 2 mirrors, 2 diagonal
  transposes). Applied to every grid of an example (demos and query, inputs and outputs).
* colour permutation: a random relabelling of the digits 1-9. Colour 0 (the background in most
  tasks) stays fixed by default; ``keep_background: false`` permutes all 10.

Config block (``data.augment`` in the YAML)::

    augment:
      dihedral: true          # random one of the 8 symmetries per example
      color_permutation: true # random relabelling of colours per example
      keep_background: true   # colour 0 is never relabelled
"""

from __future__ import annotations

import random

Grid = list[list[int]]

# name -> (forward, inverse). Each takes a grid and returns a new grid.
_identity = lambda g: [row[:] for row in g]  # noqa: E731
_rot90 = lambda g: [list(r) for r in zip(*g[::-1])]  # noqa: E731  clockwise
_rot180 = lambda g: [row[::-1] for row in g[::-1]]  # noqa: E731
_rot270 = lambda g: [list(r) for r in zip(*g)][::-1]  # noqa: E731  counter-clockwise
_flip_h = lambda g: [row[::-1] for row in g]  # noqa: E731  mirror left-right
_flip_v = lambda g: [row[:] for row in g[::-1]]  # noqa: E731  mirror top-bottom
_transpose = lambda g: [list(r) for r in zip(*g)]  # noqa: E731
_anti_transpose = lambda g: [list(r) for r in zip(*g[::-1])][::-1]  # noqa: E731

DIHEDRAL: dict[str, tuple] = {
    "identity": (_identity, _identity),
    "rot90": (_rot90, _rot270),
    "rot180": (_rot180, _rot180),
    "rot270": (_rot270, _rot90),
    "flip_h": (_flip_h, _flip_h),
    "flip_v": (_flip_v, _flip_v),
    "transpose": (_transpose, _transpose),
    "anti_transpose": (_anti_transpose, _anti_transpose),
}
DIHEDRAL_NAMES = list(DIHEDRAL)


def transform_grid(grid: Grid, name: str) -> Grid:
    return DIHEDRAL[name][0](grid)


def inverse_transform_grid(grid: Grid, name: str) -> Grid:
    return DIHEDRAL[name][1](grid)


def random_color_perm(rng: random.Random, keep_background: bool = True) -> list[int]:
    """A permutation of the digits 0-9 as a lookup list: new = perm[old]."""
    if keep_background:
        colours = list(range(1, 10))
        rng.shuffle(colours)
        return [0] + colours
    colours = list(range(10))
    rng.shuffle(colours)
    return colours


def inverse_color_perm(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for old, new in enumerate(perm):
        inv[new] = old
    return inv


def permute_colors(grid: Grid, perm: list[int]) -> Grid:
    return [[perm[c] for c in row] for row in grid]


class Augmentation:
    """One concrete view: a dihedral symmetry plus a colour permutation, applied to every grid."""

    def __init__(self, dihedral: str = "identity", color_perm: list[int] | None = None):
        self.dihedral = dihedral
        self.color_perm = color_perm

    def __repr__(self) -> str:
        return f"Augmentation({self.dihedral}, {self.color_perm})"

    def apply(self, grid: Grid) -> Grid:
        g = transform_grid(grid, self.dihedral)
        return permute_colors(g, self.color_perm) if self.color_perm else g

    def invert(self, grid: Grid) -> Grid:
        """Map a grid in the augmented view back to the original frame (for test-time voting)."""
        g = permute_colors(grid, inverse_color_perm(self.color_perm)) if self.color_perm else grid
        return inverse_transform_grid(g, self.dihedral)

    def apply_pairs(self, pairs: list[dict]) -> list[dict]:
        return [{"input": self.apply(p["input"]), "output": self.apply(p["output"])} for p in pairs]

    @property
    def is_identity(self) -> bool:
        return self.dihedral == "identity" and (self.color_perm is None or self.color_perm == list(range(10)))


IDENTITY = Augmentation()


def random_augmentation(rng: random.Random, cfg: dict | None) -> Augmentation:
    """Draw one augmentation according to the ``data.augment`` config block (None/empty = identity)."""
    if not cfg:
        return IDENTITY
    name = rng.choice(DIHEDRAL_NAMES) if cfg.get("dihedral", True) else "identity"
    perm = random_color_perm(rng, cfg.get("keep_background", True)) if cfg.get("color_permutation", True) else None
    return Augmentation(name, perm)


def all_dihedral(color_perm: list[int] | None = None) -> list[Augmentation]:
    """The 8 symmetry views of a task, e.g. for augmentation voting at inference."""
    return [Augmentation(n, color_perm) for n in DIHEDRAL_NAMES]

from grpo_lab.arc import build_prompt, cell_accuracy, grid_to_text, parse_grid, prompt_cells
from grpo_lab.arc_rewards import arc_exact, arc_format, arc_partial


def test_grid_roundtrip():
    g = [[0, 7, 7], [7, 7, 7], [0, 7, 7]]
    assert grid_to_text(g) == "077\n777\n077"
    assert parse_grid("```grid\n077\n777\n077\n```") == g


def test_parse_grid_variants():
    assert parse_grid("```\n0 7 7\n7 7 7\n```") == [[0, 7, 7], [7, 7, 7]]
    assert parse_grid("thinking...\n```grid\n12\n34\n```\n") == [[1, 2], [3, 4]]
    assert parse_grid("first ```grid\n1\n``` then ```grid\n2\n```") == [[2]]  # last block wins
    assert parse_grid("no block here 123") is None
    assert parse_grid("```grid\n12\n345\n```") is None  # ragged
    assert parse_grid("```grid\n1a\n```") is None  # not digits
    assert parse_grid("```grid\n\n```") is None  # empty
    assert parse_grid("```grid\n" + "0" * 31 + "\n```") is None  # too wide


def test_cell_accuracy():
    gold = [[1, 2], [3, 4]]
    assert cell_accuracy([[1, 2], [3, 4]], gold) == 1.0
    assert cell_accuracy([[1, 2], [3, 0]], gold) == 0.75
    assert cell_accuracy([[1, 2]], gold) == 0.0  # wrong shape
    assert cell_accuracy(None, gold) == 0.0


def test_prompt():
    demos = [{"input": [[1]], "output": [[2]]}]
    msgs = build_prompt(demos, [[3]])
    assert msgs[0]["role"] == "system"
    assert "Example 1\nInput:\n1\nOutput:\n2" in msgs[1]["content"]
    assert msgs[1]["content"].endswith("Test\nInput:\n3\nOutput:")
    assert prompt_cells(demos, [[3]]) == 3


def test_rewards():
    gold = ["12\n34"]
    right = ["```grid\n12\n34\n```"]
    close = ["```grid\n12\n30\n```"]
    bad_shape = ["```grid\n1\n```"]
    none = ["I don't know"]
    assert arc_exact(right, gold) == [1.0]
    assert arc_exact(close, gold) == [0.0]
    assert arc_partial(right, gold) == [0.5]
    assert arc_partial(close, gold) == [0.375]
    assert arc_partial(bad_shape, gold) == [0.0]
    assert arc_partial(none, gold) == [0.0]
    assert arc_format(right) == [0.1]
    assert arc_format(["```grid\n1\n```\n```grid\n2\n```"]) == [0.0]  # two blocks
    assert arc_format(["```grid\n12\n34\n```" + " blah" * 10]) == [0.0]  # rambles on
    assert arc_format(none) == [0.0]

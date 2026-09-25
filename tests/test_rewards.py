from grpo_lab.data import extract_gold, normalize_number
from grpo_lab.rewards import correctness_reward, extract_answer, format_reward


def test_normalize():
    assert normalize_number("1,234") == "1234"
    assert normalize_number("$12.50") == "12.5"
    assert normalize_number(" 7. ") == "7"
    assert normalize_number("3.0") == "3"
    assert normalize_number("\\$400") == "400"
    assert normalize_number("22.5 \\text{ hours}") == "22.5"
    assert normalize_number("15\\%") == "15"
    assert normalize_number("-4") == "-4"
    assert normalize_number("no number") is None


def test_extract_gold():
    assert extract_gold("blah blah\n#### 42") == "42"
    assert extract_gold("x <<3*4=12>> y\n#### 1,200") == "1200"


def test_extract_answer():
    assert extract_answer("so \\boxed{3} no wait \\boxed{4}.") == "4"
    assert extract_answer("\\boxed{\\frac{1}{2}}") == "1"  # imperfect, but deterministic
    assert extract_answer("\\boxed{\\$1,050}") == "1050"
    assert extract_answer("the answer is 18") is None
    assert extract_answer("\\boxed{unclosed") is None


def test_rewards():
    comps = [
        [{"role": "assistant", "content": "reasoning...\nTherefore \\boxed{18}."}],
        [{"role": "assistant", "content": "reasoning...\n\\boxed{17}"}],
        [{"role": "assistant", "content": "\\boxed{18} or maybe \\boxed{18}"}],
        [{"role": "assistant", "content": "the answer is 18"}],
        [{"role": "assistant", "content": "\\boxed{18}" + " and then it keeps going " * 10}],
    ]
    gold = ["18"] * 5
    assert correctness_reward(comps, gold) == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert format_reward(comps) == [0.2, 0.2, 0.0, 0.0, 0.0]

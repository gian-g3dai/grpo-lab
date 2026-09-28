from grpo_lab.analyze import last_number, lenient_correct, mcnemar_exact, paired_bootstrap_ci


def test_last_number():
    assert last_number("60 / 3 = 20, so the total is $1,150.") == "1150"
    assert last_number("either 3, 4") == "4"
    assert last_number("no digits here") is None


def test_lenient_correct():
    boxed = {"pred": "17", "correct": False, "gold": "18", "completion": "18 apples... \\boxed{17}"}
    plain = {"pred": None, "correct": False, "gold": "18", "completion": "Therefore she has 18 apples."}
    assert not lenient_correct(boxed)  # a box is the answer even if an earlier number matches
    assert lenient_correct(plain)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert abs(mcnemar_exact(0, 5) - 2 / 32) < 1e-12


def test_paired_bootstrap_ci():
    before = [True] * 40 + [False] * 60
    after = [True] * 55 + [False] * 45
    lo, hi = paired_bootstrap_ci(before, after, n_resamples=2000)
    assert 0 < lo < 0.15 < hi

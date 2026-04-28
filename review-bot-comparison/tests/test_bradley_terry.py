"""Bradley-Terry MM fitter sanity checks."""

from review_bot_comparison.bradley_terry import fit_bradley_terry


def test_dominant_player_ranked_first():
    strengths = fit_bradley_terry(2, [(0, 1)] * 10)
    assert strengths[0] > strengths[1]


def test_equal_players_have_similar_strength():
    strengths = fit_bradley_terry(2, [(0, 1), (1, 0), (0, 1), (1, 0)])
    assert abs(strengths[0] - strengths[1]) < 0.1


def test_strengths_normalized_to_mean_one():
    strengths = fit_bradley_terry(3, [(0, 1), (1, 2), (2, 0)])
    assert abs(sum(strengths) / 3 - 1.0) < 1e-6


def test_no_outcomes_returns_unit_strengths():
    strengths = fit_bradley_terry(3, [])
    assert len(strengths) == 3
    assert all(abs(s - 1.0) < 1e-6 for s in strengths)


def test_three_players_strict_ordering():
    # 0 beats 1, 1 beats 2, 0 beats 2 — transitive ordering should hold.
    outcomes = [(0, 1)] * 5 + [(1, 2)] * 5 + [(0, 2)] * 5
    strengths = fit_bradley_terry(3, outcomes)
    assert strengths[0] > strengths[1] > strengths[2]


def test_winless_player_collapses_to_epsilon():
    # Player 2 never wins — strength should pin to the EPSILON floor (1e-8).
    outcomes = [(0, 2)] * 10 + [(1, 2)] * 10 + [(0, 1)] * 5
    strengths = fit_bradley_terry(3, outcomes)
    assert strengths[2] < 1e-6
    assert strengths[0] > strengths[2]
    assert strengths[1] > strengths[2]


def test_dominance_is_monotonic_in_win_share():
    # More wins for player 0 → bigger gap to player 1.
    s_close = fit_bradley_terry(2, [(0, 1)] * 6 + [(1, 0)] * 4)
    s_wide = fit_bradley_terry(2, [(0, 1)] * 9 + [(1, 0)] * 1)
    assert s_wide[0] / s_wide[1] > s_close[0] / s_close[1]


def test_ranking_is_invariant_to_outcome_order():
    base = [(0, 1)] * 5 + [(1, 2)] * 5 + [(0, 2)] * 5
    shuffled = base[::-1]
    s_base = fit_bradley_terry(3, base)
    s_shuf = fit_bradley_terry(3, shuffled)
    # Same set of outcomes must produce the same fixed-point strengths.
    for a, b in zip(s_base, s_shuf, strict=True):
        assert abs(a - b) < 1e-6


def test_early_convergence_does_not_explode_with_many_iters():
    # Trivial input should converge in a few iterations and stay normalized
    # even when allowed to iterate far past the convergence point.
    s = fit_bradley_terry(2, [(0, 1)] * 10, max_iter=1000)
    assert abs((s[0] + s[1]) / 2 - 1.0) < 1e-6

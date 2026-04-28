"""Bradley-Terry MM (minorization-maximization) fitter.

Given pairwise win/loss outcomes, returns a non-negative strength per player such
that P(i beats j) = strength[i] / (strength[i] + strength[j]). Strengths are
normalized to mean 1.

Reference: Hunter (2004), "MM algorithms for generalized Bradley-Terry models."
"""

from __future__ import annotations

EPSILON = 1e-8


def fit_bradley_terry(
    n_players: int,
    outcomes: list[tuple[int, int]],
    max_iter: int = 200,
    tol: float = 1e-6,
) -> list[float]:
    """Fit Bradley-Terry strengths via the MM iteration.

    Args:
        n_players: number of players (indices in `outcomes` must be in [0, n_players)).
        outcomes: list of (winner_idx, loser_idx) pairs.
        max_iter: cap on iterations.
        tol: stop when the max absolute change in strengths drops below this.

    Returns:
        strengths[i] for i in [0, n_players), normalized so the mean is 1.0.
    """
    wins = [0] * n_players
    games = [[0] * n_players for _ in range(n_players)]
    for winner, loser in outcomes:
        wins[winner] += 1
        games[winner][loser] += 1
        games[loser][winner] += 1

    strengths = [1.0] * n_players
    for _ in range(max_iter):
        prev = list(strengths)
        for i in range(n_players):
            if wins[i] == 0:
                strengths[i] = EPSILON
                continue
            denom = sum(
                games[i][j] / (strengths[i] + strengths[j])
                for j in range(n_players)
                if j != i and games[i][j] > 0
            )
            strengths[i] = wins[i] / denom if denom > 0 else EPSILON

        mean = sum(strengths) / n_players
        strengths = [s / mean for s in strengths]

        if max(abs(a - b) for a, b in zip(prev, strengths, strict=True)) < tol:
            break

    return strengths

from __future__ import annotations
from typing import Dict, Tuple, List
import math

def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def update_elo(r_a: float, r_b: float, s_a: float, k: float = 32.0) -> Tuple[float, float]:
    ea = expected_score(r_a, r_b)
    eb = 1.0 - ea
    new_a = r_a + k * (s_a - ea)
    new_b = r_b + k * ((1.0 - s_a) - eb)
    return new_a, new_b

def compute_pre_match_elos(
    teamA: List[str], teamB: List[str], outcomes: List[int], k: float = 32.0, base: float = 1500.0
) -> Tuple[List[float], List[float]]:
    """
    Given chronological sequences (teamA[i], teamB[i], outcome[i] where outcome is 1 if teamA wins, else 0),
    return (eloA_pre[i], eloB_pre[i]) before each match; updates after each result.
    """
    ratings: Dict[str, float] = {}
    preA, preB = [], []
    for a, b, y in zip(teamA, teamB, outcomes):
        ra = ratings.get(a, base)
        rb = ratings.get(b, base)
        preA.append(ra)
        preB.append(rb)
        na, nb = update_elo(ra, rb, float(y), k)
        ratings[a], ratings[b] = na, nb
    return preA, preB

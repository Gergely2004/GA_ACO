"""Held-Karp tesztek (B szemely): brute force-szal es a ket implementacio kozott."""
import itertools
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from distances import dist_matrix, tour_length  # noqa: E402
from held_karp import held_karp, held_karp_np  # noqa: E402


def _random_D(n, seed):
    """Veletlen magyarorszagi-kozepeuropai koordinatak -> haversine matrix (km)."""
    rng = random.Random(seed)
    coords = [(rng.uniform(45.0, 48.5), rng.uniform(17.0, 26.0)) for _ in range(n)]
    return dist_matrix(coords)


def _brute_force(D):
    n = len(D)
    best = math.inf
    for perm in itertools.permutations(range(1, n)):
        best = min(best, tour_length([0, *perm], D))
    return best


@pytest.mark.parametrize("solver", [held_karp, held_karp_np])
@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_held_karp_matches_brute_force(solver, n):
    D = _random_D(n, seed=100 + n)
    best, tour = solver(D)
    assert best == pytest.approx(_brute_force(D), abs=1e-9)
    # a visszaadott korut ervenyes permutacio, 0-val indul, hossza egyezik
    assert tour[0] == 0
    assert sorted(tour) == list(range(n))
    assert tour_length(tour, D) == pytest.approx(best, abs=1e-9)


@pytest.mark.parametrize("solver", [held_karp, held_karp_np])
def test_held_karp_unit_square(solver):
    # 4 sarokpont + a kozepso pont nelkul: az optimum a negyzet kerulete = 4
    pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
    D = [[math.dist(a, b) for b in pts] for a in pts]
    best, tour = solver(D)
    assert best == pytest.approx(4.0)
    assert sorted(tour) == [0, 1, 2, 3]


@pytest.mark.parametrize("solver", [held_karp, held_karp_np])
def test_held_karp_trivial_sizes(solver):
    assert solver([]) == (0.0, [])
    assert solver([[0.0]]) == (0.0, [0])
    best, tour = solver([[0.0, 5.0], [5.0, 0.0]])
    assert best == pytest.approx(10.0)
    assert tour == [0, 1]


def test_python_and_numpy_agree_on_12_cities():
    D = _random_D(12, seed=2024)
    b1, t1 = held_karp(D)
    b2, t2 = held_karp_np(D)
    assert b1 == pytest.approx(b2, abs=1e-9)
    # a ket korut hossza ugyanaz (az iranya/valasztasa kulonbozhet egyenlo hossz eseten)
    assert tour_length(t1, D) == pytest.approx(tour_length(t2, D), abs=1e-9)


def test_numpy_float32_close_to_float64():
    import numpy as np

    D = _random_D(13, seed=7)
    b64, _ = held_karp_np(D)
    b32, _ = held_karp_np(D, dtype=np.float32)
    assert b32 == pytest.approx(b64, abs=0.05)  # km-ben

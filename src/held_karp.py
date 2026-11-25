"""Held-Karp: a TSP pontos megoldasa dinamikus programozassal (bitmaszk).

Jeloles (a projekt kozos megallapodasa szerint):
    * varos = 0..n-1 index,
    * kor = permutacio lista (a 0-s varosbol indul, a visszaut nincs benne),
    * D = n x n szimmetrikus tavolsagmatrix km-ben.

A fuggveny csak egy D matrixot var, nem fugg a distances.py-tol.
"""


def held_karp(D):
    """Tiszta Python Held-Karp, O(2^n * n^2) ido, O(2^n * n) memoria.

    Visszateres: (optimalis korut hossza, korut mint lista; a lista 0-val kezdodik).
    Gyakorlati hatar: n <= ~14 (n = 10 masodpercen belul, n = 14 par perc).
    """
    n = len(D)
    if n == 0:
        return 0.0, []
    if n == 1:
        return 0.0, [0]

    INF = float("inf")
    # dp[mask][j]: legrovidebb ut a 0-s varosbol indulva, a mask varosain at, j-ben vegzodve
    dp = [[INF] * n for _ in range(1 << n)]
    par = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0.0

    for mask in range(1, 1 << n):
        if not mask & 1:  # a 0-s varos mindig a reszhalmaz tagja
            continue
        row = dp[mask]
        for j in range(n):
            cur = row[j]
            if cur == INF or not mask & (1 << j):
                continue
            Dj = D[j]
            for k in range(n):
                if mask & (1 << k):
                    continue
                nm = mask | (1 << k)
                c = cur + Dj[k]
                if c < dp[nm][k]:
                    dp[nm][k] = c
                    par[nm][k] = j

    full = (1 << n) - 1
    best, last = min((dp[full][j] + D[j][0], j) for j in range(1, n))

    # korut visszafejtese
    tour, mask = [], full
    while last != -1:
        tour.append(last)
        mask, last = mask ^ (1 << last), par[mask][last]
    return best, tour[::-1]
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


def held_karp_np(D, dtype=None):
    """Retegzett (popcount szerint haladó), numpy-vektorizalt Held-Karp, n <= 20-ig.

    Visszateres: (optimalis hossz, korut lista 0-val kezdve) -- mint held_karp().
    dtype: alapertelmezett float64; float32-vel felere csokken a memoria,
    de km-ben ~1e-3 pontatlansag johet.
    """
    import numpy as np

    n = len(D)
    if n == 0:
        return 0.0, []
    if n == 1:
        return 0.0, [0]
    if n > 22:
        raise ValueError("held_karp_np: n > 22 mar nem fer el a memoriaban")
    if dtype is None:
        dtype = np.float64

    D = np.asarray(D, dtype=dtype)
    m = n - 1                      # a 0-s varos nelkuli varosok szama
    Dm = D[1:, 1:]                 # m x m: a varosok kozotti tavolsagok (1..n-1)
    d0 = D[0, 1:]                  # 0 -> k tavolsagok
    INF = np.array(np.inf, dtype=dtype)

    size = 1 << m
    # dp[mask, k]: 0-bol indulva, a mask (1..n-1 varosok bitmaszkja) varosain at, k-ban vegzodve
    dp = np.full((size, m), INF, dtype=dtype)
    for k in range(m):
        dp[1 << k, k] = d0[k]

    # popcount minden maszkra
    masks = np.arange(size, dtype=np.int64)
    pc = np.zeros(size, dtype=np.int8)
    for b in range(m):
        pc += ((masks >> b) & 1).astype(np.int8)

    for s in range(2, m + 1):
        layer = masks[pc == s]
        for k in range(m):
            sel = layer[(layer >> k) & 1 == 1]     # k-t tartalmazo s elemu maszkok
            if sel.size == 0:
                continue
            prev = sel ^ (1 << k)
            # dp[prev, j] = INF, ha j nincs prev-ben -> a min automatikusan jo
            dp[sel, k] = (dp[prev] + Dm[:, k]).min(axis=1)

    full = size - 1
    final = dp[full] + D[1:, 0]
    last = int(np.argmin(final))
    best = float(final[last])

    # korut visszafejtese (argmin ujraszamolasa)
    tour_rev = [last + 1]
    mask = full
    while mask & (mask - 1):       # amig 1-nel tobb bit van
        prev = mask ^ (1 << last)
        j = int(np.argmin(dp[prev] + Dm[:, last]))
        tour_rev.append(j + 1)
        mask, last = prev, j
    return best, [0] + tour_rev[::-1]

"""Segedszkript: melyik 10 varos adja a megadott Held-Karp optimumot (pl. 443,3 km)?

A feladat nem mondja meg, melyik 10 varosrol van szo, ezert a megadott
CSV-k varosaibol kivalasztott 10 elemu reszhalmazokat probalja vegig, es
kilistazza azokat, amelyeknek a Held-Karp optimuma a celertek +-tol km-en belul van.

Hasznalat (a src/ mappabol):
    python find_subset.py ../data/cities_tr14.csv
    python find_subset.py ../data/cities_hu20.csv --target 443.3 --tol 0.05
    python find_subset.py ../data/cities_tr14.csv ../data/cities_kp16.csv ../data/cities_hu20.csv
    python find_subset.py ../data/cities_hu20.csv --mode consecutive
    python find_subset.py ../data/cities_tr14.csv --radii 6371 6371.0088 6378.137 6366.707

Modok:
    first        : csak a lista elso k varosa
    consecutive  : a lista osszes folyamatos k hosszu ablaka
    all          : az osszes C(n, k) reszhalmaz (alapertelmezett; MST-alsokorlat
                   szuressel gyorsitva, 20 varosbol is percek alatt vegez)
Tobb CSV megadasakor a kulonallo listak mellett az egyesitett lista (nev szerint
duplikatum nelkul) is vizsgalhato az --union kapcsoloval.
"""
import argparse
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distances import dist_matrix, load_cities  # noqa: E402
from held_karp import held_karp_np  # noqa: E402


def mst_weight(D, idx):
    """Prim-algoritmus a idx varosokra. A TSP-koru hossza >= MST sulya (alsokorlat)."""
    k = len(idx)
    in_tree = [False] * k
    key = [math.inf] * k
    key[0] = 0.0
    total = 0.0
    for _ in range(k):
        u = min((i for i in range(k) if not in_tree[i]), key=key.__getitem__)
        in_tree[u] = True
        total += key[u]
        du = D[idx[u]]
        for v in range(k):
            if not in_tree[v] and du[idx[v]] < key[v]:
                key[v] = du[idx[v]]
    return total


def candidate_subsets(n, k, mode):
    if mode == "first":
        yield tuple(range(min(k, n)))
    elif mode == "consecutive":
        for s in range(n - k + 1):
            yield tuple(range(s, s + k))
    elif mode == "all":
        yield from itertools.combinations(range(n), k)
    else:
        raise ValueError(f"ismeretlen mod: {mode}")


def search(names, coords, k, target, tol, mode="all", R=6371.0):
    """Visszaadja a talalatok listajat: [(hossz, [indexek], [nevek], [korut indexei])]."""
    if k > len(names):
        return []
    D = dist_matrix(coords, R=R)
    hits = []
    checked = pruned = 0
    for idx in candidate_subsets(len(names), k, mode):
        # opt >= MST  es  opt <= 2*MST  -> ha a celertek ezen kivul esik, kihagyhato
        mst = mst_weight(D, idx)
        if mst > target + tol or 2 * mst < target - tol:
            pruned += 1
            continue
        sub = [[D[i][j] for j in idx] for i in idx]
        best, tour = held_karp_np(sub)
        checked += 1
        if abs(best - target) <= tol:
            hits.append((best, list(idx), [names[i] for i in idx], [idx[t] for t in tour]))
    return hits, checked, pruned


def union_lists(lists):
    """Tobb (names, coords) egyesitese nev szerint, a sorrendet megtartva."""
    seen, names, coords = set(), [], []
    for ns, cs in lists:
        for nm, c in zip(ns, cs):
            if nm not in seen:
                seen.add(nm)
                names.append(nm)
                coords.append(c)
    return names, coords


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="varoslista CSV(k): name,lat,lon")
    ap.add_argument("--k", type=int, default=10, help="reszhalmaz merete (alap: 10)")
    ap.add_argument("--target", type=float, default=443.3, help="celertek km-ben (alap: 443.3)")
    ap.add_argument("--tol", type=float, default=0.05, help="megengedett eltérés km-ben (alap: 0.05)")
    ap.add_argument("--mode", choices=["first", "consecutive", "all"], default="all")
    ap.add_argument("--radii", type=float, nargs="+", default=[6371.0],
                    help="kiprobalando foldsugarak km-ben (alap: 6371)")
    ap.add_argument("--union", action="store_true", help="az egyesitett listat is vizsgalja")
    ap.add_argument("--max-hits", type=int, default=20, help="legfeljebb ennyi talalat kiirasa listankent")
    args = ap.parse_args()

    lists = [(os.path.basename(p), *load_cities(p)) for p in args.csv]
    if args.union and len(lists) > 1:
        un, uc = union_lists([(n, c) for _, n, c in lists])
        lists.append(("UNION", un, uc))

    total = 0
    for label, names, coords in lists:
        for R in args.radii:
            res = search(names, coords, args.k, args.target, args.tol, args.mode, R)
            if not res:
                print(f"[{label}, R={R}] csak {len(names)} varos van, k={args.k} nem lehetseges")
                continue
            hits, checked, pruned = res
            print(f"[{label}, R={R}] {len(names)} varos, mod={args.mode}: "
                  f"{checked} Held-Karp futas, {pruned} reszhalmaz kiszurve, {len(hits)} talalat")
            for best, idx, nm, tour in hits[: args.max_hits]:
                print(f"   {best:.3f} km  indexek={idx}")
                print(f"      varosok: {', '.join(nm)}")
                print(f"      korut:   {' -> '.join(names[t] for t in tour)} -> {names[tour[0]]}")
            total += len(hits)
    if total == 0:
        print("Nincs talalat. Probald: nagyobb --tol, mas --radii, --union, vagy kerdezd meg az oktatot.")


if __name__ == "__main__":
    main()

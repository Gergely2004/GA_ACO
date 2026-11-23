import csv
import math

R_EARH_KM = 6371.0


def haversine(lat1, lon1, lat2, lon2, R=R_EARH_KM):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def dist_matrix(coords, R=R_EARH_KM):
    n = len(coords)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            D[i][j] = D[j][i] = haversine(*coords[i], *coords[j], R=R)
    return D


def load_cities(path):
    names, coords = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            names.append(row["name"])
            coords.append((float(row["lat"]), float(row["lon"])))
    return names, coords


def tour_length(tour, D):
    n = len(tour)
    return sum(D[tour[k]][tour[(k + 1) % n]] for k in range(n))

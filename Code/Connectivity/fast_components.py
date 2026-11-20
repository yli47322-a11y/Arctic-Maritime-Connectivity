"""Incremental four-neighbor component counts for nested POLARIS passability."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "全区海上联系机会_1979-2025/.runtime"))
import numpy as np
from numba import njit


@njit(cache=True)
def pass_rank(c, h, cat, unknown, riv):
    n = len(c)
    rank_low = np.empty(n, np.int8)
    rank_high = np.empty(n, np.int8)
    for i in range(n):
        if not unknown[i] and c[i] == 0.0:
            rank_low[i] = rank_high[i] = 11
            continue
        low_c = c[i] if np.isfinite(c[i]) else 1.0
        high_c = c[i] if np.isfinite(c[i]) else 0.0
        low = high = 11
        if unknown[i]:
            while low > 0 and 30.0 + 10.0*low_c*(riv[low, -1]-3) <= -1e-6:
                low -= 1
            while high > 0 and 30.0 + 10.0*high_c*(riv[high, 0]-3) <= -1e-6:
                high -= 1
        else:
            while low > 0 and 30.0 + 10.0*c[i]*(riv[low, cat[i]]-3) <= -1e-6:
                low -= 1
            high = low
        rank_low[i] = low
        rank_high[i] = high
    return rank_low, rank_high


@njit(cache=True, inline="always")
def root(parent, cell):
    while parent[cell] >= 0:
        p = parent[cell]
        if parent[p] >= 0:
            parent[cell] = parent[p]
        cell = p
    return cell


@njit(cache=True, inline="always")
def merge(parent, mass, a, b):
    ra, rb = root(parent, a), root(parent, b)
    if ra == rb:
        return 0
    if parent[ra] > parent[rb]:
        ra, rb = rb, ra
    add = mass[ra] * mass[rb]
    parent[ra] += parent[rb]
    parent[rb] = ra
    mass[ra] += mass[rb]
    return add


@njit(cache=True)
def connected_pairs(grid_rank, domain_flat, mass_template, nx, n_classes, departures):
    parent = np.zeros(grid_rank.size, np.int32)
    # Unvisited cells have parent equal to themselves plus one.
    inactive = np.int32(2147483647)
    parent[:] = inactive
    mass = mass_template.copy()
    sizes = np.zeros(n_classes, np.int32)
    for cell in domain_flat:
        sizes[grid_rank[cell]] += 1
    edges = np.zeros(n_classes + 1, np.int32)
    for k in range(n_classes):
        edges[k+1] = edges[k] + sizes[k]
    cursor = edges[:-1].copy()
    order = np.empty(len(domain_flat), np.int32)
    for cell in domain_flat:
        k = grid_rank[cell]
        order[cursor[k]] = cell
        cursor[k] += 1
    total = np.int64(0)
    counts = np.empty(n_classes, np.int64)
    labels = np.empty((n_classes, len(departures)), np.int32)
    for k in range(n_classes-1, -1, -1):
        for j in range(edges[k], edges[k+1]):
            cell = order[j]
            parent[cell] = -1
            if cell % nx > 0 and parent[cell-1] != inactive:
                total += merge(parent, mass, cell, cell-1)
            if (cell+1) % nx > 0 and parent[cell+1] != inactive:
                total += merge(parent, mass, cell, cell+1)
            if cell >= nx and parent[cell-nx] != inactive:
                total += merge(parent, mass, cell, cell-nx)
            if cell+nx < parent.size and parent[cell+nx] != inactive:
                total += merge(parent, mass, cell, cell+nx)
        counts[k] = total
        for q in range(len(departures)):
            labels[k, q] = root(parent, departures[q])
    return counts, labels


def init(g):
    riv = g["riv"]
    # Positive RIV classes always pass. The remaining pass masks are nested.
    severity = np.minimum(riv, 0)
    if (np.diff(severity, axis=0) > 0).any() or not np.array_equal(riv.min(axis=1), riv[:, -1]) or not np.array_equal(riv.max(axis=1), riv[:, 0]):
        raise ValueError("The RIV matrix does not satisfy nested passability")
    g["domain_flat"] = np.flatnonzero(g["domain"]).astype("int32")
    g["mass_template"] = np.zeros(g["domain"].size, "int32")
    g["mass_template"][g["departures"]] = g["weights"]


def process_day(raw_c, heff, g, edges):
    c = np.clip(raw_c.astype("float64"), 0, 1)
    c[c <= 1e-6] = 0
    unknown = ~np.isfinite(c) | ~np.isfinite(heff) | (heff < 0) | ((c > 0) & (heff == 0))
    h = np.divide(heff, c, out=np.zeros_like(c), where=c > 0)
    cat = np.searchsorted(edges, h, side="right") + 1
    cat[c == 0] = 0
    if unknown.any():
        raise ValueError("Unresolved risk input after paired sea-ice correction")
    lo, _ = pass_rank(c, h, cat, unknown, g["riv"])
    rank = np.full(g["domain"].size, 11, np.int8)
    rank[g["active"]] = lo
    return connected_pairs(rank, g["domain_flat"], g["mass_template"], g["domain"].shape[1],
                           len(g["riv"]), g["departures"])

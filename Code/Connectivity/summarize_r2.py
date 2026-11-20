"""Summarize the corrected R2 daily graph with exact settlement-pair weights."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from fast_components import njit
from run_r2 import rows, DEPS, OUT, YEARS, write_csv

MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_BOUNDS = np.r_[0, np.cumsum(MONTH_LENGTHS)]


@njit(cache=True)
def histograms(traces, weight):
    days = traces.shape[1]
    out = np.zeros((3, days + 1), np.int64)
    for a in range(len(weight)):
        wa = weight[a]
        for k in range(2):
            out[k, days] += wa * (wa - 1) // 2
        out[2, 1] += wa * (wa - 1) // 2
        for b in range(a):
            count, run, longest, windows = 0, 0, 0, 0
            for t in range(days):
                if traces[a,t] == traces[b,t]:
                    if run == 0:
                        windows += 1
                    run += 1
                    count += 1
                    if run > longest:
                        longest = run
                else:
                    run = 0
            ab = wa * weight[b]
            out[0, count] += ab
            out[1, longest] += ab
            out[2, windows] += ab
    return out


def group_sizes(labels, weight):
    _, g = np.unique(labels, return_inverse=True)
    sizes = np.bincount(g, weights=weight).astype("int64")
    return sizes, g


def summarize_year(year):
    output = OUT / "year_summary" / f"{year}.npz"
    if output.exists():
        return year
    weight = np.array([int(r["settlement_record_count"]) for r in rows(DEPS)], dtype="int64")
    arrays, pairs = [], []
    for month in range(1, 13):
        with np.load(OUT/"monthly_counts"/f"{year}-{month:02}.npz") as f:
            arrays.append(f["labels"])
            pairs.append(f["pairs"])
    labels, pairs = np.concatenate(arrays), np.concatenate(pairs)
    if len(labels) != 365:
        raise ValueError(f"Incomplete year: {year}")
    largest = np.zeros(pairs.shape, dtype="int16")
    retention = np.full(pairs.shape, np.nan)
    node_total = np.zeros(labels.shape[1:], dtype="int64")
    hist = np.empty((pairs.shape[1],3,366), "int64")
    prev_labels, prev_pairs = None, None
    if year > YEARS[0]:
        with np.load(OUT/"monthly_counts"/f"{year-1}-12.npz") as f:
            prev_labels, prev_pairs = f["labels"][-1], f["pairs"][-1]
    candidate = weight.sum() * (weight.sum()-1)//2
    for s in range(labels.shape[1]):
        for t, ll in enumerate(labels[:,s,:]):
            sizes, g = group_sizes(ll, weight)
            largest[t,s] = sizes.max()
            node_total[s] += sizes[g] - 1
            if t or prev_labels is not None:
                old = labels[t-1,s] if t else prev_labels[s]
                n = pairs[t-1,s] if t else prev_pairs[s]
                if n:
                    key = (old.astype("int64") << 32) | ll
                    common, _ = group_sizes(key, weight)
                    retention[t,s] = np.dot(common, common-1) / (2*n)
        trace, inverse = np.unique(labels[:,s,:].T, axis=0, return_inverse=True)
        combined = np.bincount(inverse, weights=weight).astype("int64")
        hist[s] = histograms(trace, combined)
        if (hist[s].sum(axis=1) != candidate).any() or np.dot(hist[s,0], np.arange(366)) != pairs[:,s].sum():
            raise ValueError(f"Inconsistent exact pair aggregation: {year} class {s}")
    tmp = output.with_suffix(".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(f, pairs=pairs, largest=largest, retention=retention, node_total=node_total, hist=hist)
    tmp.replace(output)
    return year


def combine():
    state=json.loads((OUT/"运行状态.json").read_text(encoding="utf-8"))
    n, candidate = state["settlement_records"], state["candidate_settlement_pairs"]
    ships=state["ship_classes"]
    ys=[]
    for y in YEARS:
        with np.load(OUT/"year_summary"/f"{y}.npz") as d:
            ys.append({key:d[key] for key in d.files})
    pairs, largest, persistence, nodes, hist = [np.stack([d[k] for d in ys]) for k in
                                                 ("pairs","largest","retention","node_total","hist")]
    daily=pairs/candidate
    annual=daily.mean(axis=1)
    monthly=np.stack([daily[:,a:b].mean(axis=1) for a,b in zip(MONTH_BOUNDS[:-1],MONTH_BOUNDS[1:])],axis=1)
    change=100*(monthly[-10:].mean(axis=0)-monthly[:10].mean(axis=0))
    contribution=change*np.array(MONTH_LENGTHS)[:,None]/365
    x=YEARS-YEARS.mean()
    co=1000*x/np.dot(x,x)
    trend=np.einsum("y,ys->s",co,annual)
    node_year=nodes/(365*(n-1))
    node_slope=np.einsum("y,ysd->sd",co,node_year)
    q=np.array([.25,.50,.75])
    pair_quantiles=np.empty((len(YEARS),len(ships),3,3),dtype="float64")
    for y in range(len(YEARS)):
        for s in range(len(ships)):
            for k in range(3):
                cum=hist[y,s,k].cumsum()
                rank=q*(candidate-1)
                a=np.searchsorted(cum,np.floor(rank),side="right")
                b=np.searchsorted(cum,np.ceil(rank),side="right")
                pair_quantiles[y,s,k]=a+(rank-np.floor(rank))*(b-a)
    np.savez_compressed(OUT/"R2_完整分析结果.npz", years=YEARS, ships=ships, daily=daily, annual=annual, monthly=monthly,
                        monthly_delta_pp=change, monthly_contribution_pp=contribution, trend=trend, largest=largest/n,
                        persistence=persistence, hist=hist, pair_quantiles=pair_quantiles,
                        node_year=node_year, node_slope=node_slope, pairs=pairs, candidate=candidate)
    write_csv(OUT/"全区年度与月度机会.csv",
              [dict(year=int(y),ship_class=s,annual=float(annual[i,k]),**{
               f"month_{j+1:02}":float(monthly[i,j,k]) for j in range(12)})
               for i,y in enumerate(YEARS) for k,s in enumerate(ships)])
    write_csv(OUT/"全区趋势与首末十年对比.csv",
              [dict(ship_class=s,mean_1979_1988_pct=100*annual[:10,k].mean(),mean_2016_2025_pct=100*annual[-10:,k].mean(),
                    change_pp=100*(annual[-10:,k].mean()-annual[:10,k].mean()),ols_pp_per_decade=trend[k])
               for k,s in enumerate(ships)])
    deps={r["departure_id"]:i for i,r in enumerate(rows(DEPS))}
    relation=rows(DEPS.with_name("聚居点与出发格对应.csv"))
    write_csv(OUT/"4258聚居地_12船级机会趋势.csv",
              [dict(entity_id=r["entity_id"],name=r["name"],departure_id=r["departure_id"],longitude=r["settlement_longitude"],
                    latitude=r["settlement_latitude"],**{s:node_slope[k,deps[r["departure_id"]]] for k,s in enumerate(ships)})
               for r in relation])


def main():
    from time import sleep

    (OUT/"year_summary").mkdir(exist_ok=True)
    histograms(np.zeros((1,2),dtype="int32"),np.ones(1,dtype="int64"))
    pending=set(int(y) for y in YEARS if not (OUT/"year_summary"/f"{y}.npz").is_file())
    with ProcessPoolExecutor(max_workers=2) as pool:
        while pending:
            state=json.loads((OUT/"运行状态.json").read_text(encoding="utf-8"))
            if state["status"] == "failed":
                raise ValueError("Daily components failed")
            ready=[y for y in sorted(pending) if all(
                (OUT/"monthly_counts"/f"{y}-{m:02}.npz").is_file() for m in range(1,13))
                and (y == YEARS[0] or (OUT/"monthly_counts"/f"{y-1}-12.npz").is_file())]
            if not ready:
                sleep(12)
                continue
            for y in pool.map(summarize_year, ready):
                pending.remove(y)
                print(f"Pair distributions: {len(YEARS)-len(pending)}/{len(YEARS)}, {y}",flush=True)
    while True:
        state=json.loads((OUT/"运行状态.json").read_text(encoding="utf-8"))
        if state["status"]=="complete":
            break
        if state["status"]=="failed":
            raise ValueError("Daily components failed")
        sleep(2)
    combine()


if __name__ == "__main__":
    main()

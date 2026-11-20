"""Compute pan-Arctic daily pair counts and annual opportunity trends."""
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5netcdf
import numpy as np
from scipy.ndimage import generate_binary_structure, label

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "正式数据/PIOMAS_5km_预处理_v1/processed"
RIV = ROOT / "参数资料/POLARIS_12船级_2026-11-01/POLARIS_RIV_12classes_standard.csv"
DEPS = ROOT / "预处理/PIOMAS_5km_标准流程_v1/固定点位/出发格.csv"
EDGES = np.array([.10, .15, .30, .50, .70, .95, 1.20, 2., 2.5, 3.])
YEARS = np.arange(1979, 2026)
CROSS = generate_binary_structure(2, 1)
G = {}
import fast_components


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_state(**state):
    tmp = OUT / "运行状态.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT / "运行状态.json")
    print(json.dumps(state, ensure_ascii=False), flush=True)


def init(g):
    G.update(g)
    fast_components.init(G)


def job(month):
    path = SOURCE / f"PIOMAS_{month}_逐日_5km.nc"
    out = OUT / "monthly_counts" / f"{month}.npz"
    pairs, labels = [], []
    with h5netcdf.File(path, "r") as f:
        if f.attrs.get("ice_pair_correction_version") != "joint_nearest_sea_ring_v1":
            raise ValueError(f"Missing corrected ice inputs: {path}")
        day = f["source_day_index"][:]
        c, h = f["siconc"], f["sithick_effective"]
        for k in range(len(day)):
            pp, ll = fast_components.process_day(c[k].ravel()[G["active"]], h[k].ravel()[G["active"]], G, EDGES)
            pairs.append(pp + G["shared"])
            labels.append(ll)
    tmp = out.with_suffix(".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(f, pairs=pairs, labels=labels, day=day)
    tmp.replace(out)
    return month


def main():
    (OUT / "monthly_counts").mkdir(exist_ok=True)
    months = [f"{y:04d}-{m:02d}" for y in YEARS for m in range(1, 13)]
    riv_rows, dep_rows = rows(RIV), rows(DEPS)
    ships = [r["ship_class"] for r in riv_rows]
    riv = np.array([[int(r[k]) for k in r if k != "ship_class"] for r in riv_rows])
    weights = np.array([int(r["settlement_record_count"]) for r in dep_rows], dtype="int64")
    coordinates = tuple([int(r[key]) for r in dep_rows] for key in ("row", "col"))
    with h5netcdf.File(SOURCE / "PIOMAS_1979-01_逐日_5km.nc", "r") as f:
        domain = f["marine_domain_mask"][:].astype(bool)
        active = np.flatnonzero((domain & f["AMAP_mask"][:].astype(bool)).ravel())
    departures = np.ravel_multi_index(coordinates, domain.shape)
    n = int(weights.sum())
    assert (n, len(weights)) == (4258, 1248)
    denominator = n * (n - 1) // 2
    shared = int(np.dot(weights, weights - 1) // 2)
    geometry = dict(domain=domain, active=active, departures=departures,
                    weights=weights, shared=shared, ships=ships, riv=riv)
    remaining = [month for month in months if not (OUT / "monthly_counts" / f"{month}.npz").exists()]
    done = len(months) - len(remaining)
    try:
        write_state(status="running", completed_months=done, total_months=len(months))
        if remaining:
            with ProcessPoolExecutor(max_workers=8, initializer=init, initargs=(geometry,)) as executor:
                for future in as_completed([executor.submit(job, m) for m in remaining]):
                    latest = future.result()
                    done += 1
                    if done % 12 == 0 or done == len(months):
                        write_state(status="running", completed_months=done, total_months=len(months), latest=latest)
        write_state(status="complete", completed_months=done, total_months=len(months),
                    complete_source_years=[1979, 2025], source_days=47*365, settlement_records=n,
                    departures=len(weights), candidate_settlement_pairs=denominator,
                    same_departure_pairs_included=shared, ship_classes=ships,
                    input_correction="joint_nearest_sea_ring_v1")
    except Exception as exc:
        write_state(status="failed", completed_months=done, error=repr(exc))
        raise


if __name__ == "__main__":
    import msvcrt
    with (OUT / ".process.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()



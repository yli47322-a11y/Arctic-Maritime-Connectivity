"""Apply the approved joint 5 km ice-field correction to existing PIOMAS files."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "正式数据/PIOMAS_5km_预处理_v1"
OUT = BASE
sys.path.insert(0, str(ROOT / "正式成果/全区海上联系机会_1979-2025/.runtime"))
from numba import njit

LIMIT = 5
TOL = 1e-6


@njit(cache=True)
def neighbors(c, h, donor, sea, rr, cc, limit):
    count = len(rr)
    values_c = np.zeros(count, np.float32)
    values_h = np.zeros(count, np.float32)
    radius = np.zeros(count, np.uint8)
    side = 2 * limit + 1
    qr = np.empty(side * side, np.int32)
    qc = np.empty_like(qr)
    used = np.zeros((side, side), np.int32)
    dr = (-1, 1, 0, 0)
    dc = (0, 0, -1, 1)
    for k in range(count):
        r0, c0 = rr[k], cc[k]
        qr[0], qc[0] = r0, c0
        used[limit, limit] = k + 1
        start, end = 0, 1
        for step in range(1, limit + 1):
            tail = end
            sum_c, sum_h, n = 0.0, 0.0, 0
            for i in range(start, end):
                for d in range(4):
                    r, x = qr[i] + dr[d], qc[i] + dc[d]
                    if r < 0 or r >= sea.shape[0] or x < 0 or x >= sea.shape[1] or not sea[r, x]:
                        continue
                    a, b = r - r0 + limit, x - c0 + limit
                    if used[a, b] == k + 1:
                        continue
                    used[a, b] = k + 1
                    qr[tail], qc[tail] = r, x
                    tail += 1
                    if donor[r, x]:
                        sum_c += c[r, x]
                        sum_h += h[r, x]
                        n += 1
            if n > 0:
                if sum_c / n > TOL:
                    values_c[k], values_h[k] = sum_c / n, sum_h / n
                radius[k] = step
                break
            start, end = end, tail
            if start == end:
                break
    return values_c, values_h, radius


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def amend(src):
    report = OUT / "修正统计" / (src.stem + ".json")
    with h5py.File(src, "r") as f:
        if "ice_pair_correction_summary" in f.attrs:
            item = json.loads(f.attrs["ice_pair_correction_summary"])
            item["output_bytes"] = src.stat().st_size
            write_json(report, item)
            return item
    tmp = src.with_suffix(".correction.partial")
    shutil.copyfile(src, tmp)
    anomaly_count = filled_count = days = 0
    hist = np.zeros(LIMIT + 1, dtype="int64")
    with h5py.File(tmp, "r+") as f:
        sea = (f["marine_domain_mask"][:] == 1) & (f["AMAP_mask"][:] == 1)
        fixed = f["departure_zero_mask"][:] == 1
        c_file = f["siconc"]
        h_file = f["sithick_effective"]
        cf_file = f["siconc_provenance_flag"]
        hf_file = f["sithick_effective_provenance_flag"]
        for day in range(c_file.shape[0]):
            c, h = c_file[day], h_file[day]
            anomaly = sea & ~fixed & (~np.isfinite(c) | ~np.isfinite(h) | (h < 0)
                                     | ((c > TOL) & (h == 0)))
            rr, cc = np.where(anomaly)
            if not len(rr):
                continue
            days += 1
            cf, hf = cf_file[day], hf_file[day]
            c[anomaly], h[anomaly] = 0, 0
            donor = (sea & ~fixed & ~anomaly & (((cf | hf) & 2) == 0)
                     & np.isfinite(c) & np.isfinite(h) & (c >= 0) & (c <= 1)
                     & (((c <= TOL) & (h == 0)) | ((c > TOL) & (h > 0))))
            vc, vh, radius = neighbors(c, h, donor, sea, rr, cc, LIMIT)
            c[rr, cc], h[rr, cc] = vc, vh
            bit = np.where(radius > 0, 4, 8).astype("u1")
            cf[rr, cc] |= bit
            hf[rr, cc] |= bit
            n = int(np.count_nonzero(radius))
            anomaly_count += len(rr)
            filled_count += n
            hist += np.bincount(radius, minlength=LIMIT + 1)
            # Only rewrite chunks that contain a corrected value.
            dy, dx = c_file.chunks[1:]
            for j, i in np.unique(np.column_stack((rr // dy, cc // dx)), axis=0):
                ys, xs = slice(j * dy, (j + 1) * dy), slice(i * dx, (i + 1) * dx)
                c_file[day, ys, xs] = c[ys, xs]
                h_file[day, ys, xs] = h[ys, xs]
                cf_file[day, ys, xs] = cf[ys, xs]
                hf_file[day, ys, xs] = hf[ys, xs]
        for flag in (cf_file, hf_file):
            flag.attrs["flag_masks"] = np.array([1, 2, 4, 8], dtype="u1")
            flag.attrs["flag_meanings"] = ("influenced_by_native_gap_interpolation residual_missing_filled_zero "
                                          "anomaly_joint_neighborhood_interpolation anomaly_fallback_ice_free")
        f.attrs["ice_pair_correction_version"] = "joint_nearest_sea_ring_v1"
        f.attrs["correction_rule"] = ("AMAP marine anomalies jointly corrected by the average of valid paired donors "
                                      "at the nearest four-neighbor sea distance, max 25 km; fallback zero; "
                                      "fixed departure and assumed outside ice-free cells excluded as donors")
        item = dict(month=src.name[7:14], days=c_file.shape[0], affected_days=days, anomaly_cell_days=anomaly_count,
                    interpolated_cell_days=filled_count, ice_free_fallback_cell_days=anomaly_count - filled_count,
                    sea_distance_cell_days=hist.tolist())
        f.attrs["ice_pair_correction_summary"] = json.dumps(item)
    tmp.replace(src)
    item["output_bytes"] = src.stat().st_size
    write_json(report, item)
    return item


def main():
    source = sorted((BASE / "processed").glob("*.nc"))
    if len(source) != 566:
        raise ValueError(f"Unexpected number of input files: {len(source)}")
    for sub in ("processed", "修正统计"):
        (OUT / sub).mkdir(exist_ok=True)
    g = np.zeros((3, 3), dtype="f4")
    neighbors(g, g, g == 0, g == 0, np.array([1]), np.array([1]), LIMIT)
    metadata = dict(status="running", total_months=len(source), completed_months=0)
    totals = dict(days=0, affected_days=0, anomaly_cell_days=0, interpolated_cell_days=0,
                  ice_free_fallback_cell_days=0, output_bytes=0)
    hist = np.zeros(LIMIT + 1, dtype="int64")
    try:
        write_json(OUT / "异常格修正状态.json", metadata)
        with ProcessPoolExecutor(max_workers=6) as pool:
            for f in as_completed([pool.submit(amend, src) for src in source]):
                result = f.result()
                metadata["completed_months"] += 1
                for key in totals:
                    totals[key] += result[key]
                hist += result["sea_distance_cell_days"]
                if metadata["completed_months"] % 12 == 0:
                    write_json(OUT / "异常格修正状态.json", metadata | totals)
                    print(metadata | totals, flush=True)
        metadata["status"] = "complete"
        write_json(OUT / "异常格修正状态.json", metadata | totals | {"sea_distance_cell_days": hist.tolist()})
        print(metadata | totals, flush=True)
    except Exception as e:
        metadata.update(status="failed", error=repr(e))
        write_json(OUT / "异常格修正状态.json", metadata | totals)
        raise


if __name__ == "__main__":
    import msvcrt

    OUT.mkdir(exist_ok=True)
    with (OUT / ".process.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()

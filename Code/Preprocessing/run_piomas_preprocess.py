"""Process locally downloaded PIOMAS using the approved 5 km workflow."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import date
import json
from pathlib import Path
from zipfile import BadZipFile

import h5netcdf
import numpy as np
from pyproj import Transformer
import rasterio
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "PIOMAS_v2.1_1979-2026"
OUT = ROOT / "正式数据/PIOMAS_5km_预处理_v1"
NATIVE = (120, 360)
MONTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
FIELDS = (("siconc", "aiday", "area", "1"), ("sithick_effective", "hiday", "hiday", "m"))
VERSION = "PIOMAS_5km_preprocessing_v1"
GEOM = {}


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def fixed_width(path, width):
    return np.array([float(line[i:i + width]) for line in path.read_text().splitlines()
                     for i in range(0, len(line), width) if line[i:i + width].strip()])


def stencil(source, targets):
    tri = Delaunay(source)
    simplices = tri.find_simplex(targets)
    keep = np.flatnonzero(simplices >= 0)
    ids = simplices[keep]
    transform = tri.transform[ids]
    b = np.einsum("ijk,ik->ij", transform[:, :2], targets[keep] - transform[:, 2])
    return keep, tri.simplices[ids], np.column_stack((b, 1 - b.sum(axis=1))).astype("f4")


def interpolate(values, vertices, weights):
    return np.sum(values[vertices] * weights, axis=1)


def spatial():
    cfg = json.loads((HERE / "流程参数.json").read_text(encoding="utf-8"))
    with rasterio.open(ROOT / cfg["grid"]["land_mask"]) as src:
        land, tf, crs = src.read(1) == 1, src.transform, src.crs
    with rasterio.open(ROOT / cfg["grid"]["AMAP_mask"]) as src:
        amap = src.read(1) == 1
    x = tf.c + (np.arange(land.shape[1]) + .5) * tf.a
    y = tf.f + (np.arange(land.shape[0]) + .5) * tf.e
    xx, yy = np.meshgrid(x, y)
    _, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(xx, yy)
    outside = ~land & ~amap & (lat <= 70)
    inside = amap & ~land
    domain = inside | outside
    with (HERE / "固定点位/出发格.csv").open(encoding="utf-8-sig") as f:
        nodes = list(csv.DictReader(f))
    rr = np.array([int(node["row"]) for node in nodes])
    cc = np.array([int(node["col"]) for node in nodes])
    dep = np.ravel_multi_index((rr, cc), land.shape)
    if len(set(dep)) != cfg["nodes"]["departure_cells"] or not domain.ravel()[dep].all():
        raise ValueError("Frozen departure grid does not match the marine domain")
    lon, native_lat = fixed_width(SOURCE / "static/grid.dat", 8).reshape(2, *NATIVE)
    wet = fixed_width(SOURCE / "static/io.dat_360_120.output", 2).reshape(NATIVE) > 0
    sx, sy = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, native_lat)
    native_xy = np.column_stack((sx.ravel(), sy.ravel()))
    ocean = np.flatnonzero(wet.ravel())
    fidx, fv, fw = stencil(native_xy[ocean], native_xy)
    amap_ids = np.flatnonzero(amap.ravel())
    keep, sv, sw = stencil(native_xy, np.column_stack((xx.ravel()[amap_ids], yy.ravel()[amap_ids])))
    sample_ids = amap_ids[keep]
    on_sea = inside.ravel()[sample_ids]
    np.savez(OUT / "spatial/5km_线性采样权重.npz", target_flat_index=sample_ids,
             native_flat_vertices=sv, weights=sw, target_shape=land.shape, native_shape=NATIVE)
    g = dict(land=land, amap=amap, outside=outside, inside=inside, domain=domain, dep=dep,
             rr=rr, cc=cc, x=x, y=y, lon=lon, native_lat=native_lat, wet=wet,
             native_xy=native_xy, ocean=ocean, fill_index=fidx, fill_vertices=ocean[fv],
             fill_weights=fw, sample_index=sample_ids[on_sea], sample_vertices=sv[on_sea],
             sample_weights=sw[on_sea], crs=crs.to_wkt())
    return g


def initialize(geometry):
    GEOM.update(geometry)


def reconstruct(raw):
    g = GEOM
    values = raw.ravel()
    original = g["wet"].ravel() & np.isfinite(values) & (np.abs(values) <= 1e15)
    if original[g["ocean"]].all():
        target, vertices, weights = g["fill_index"], g["fill_vertices"], g["fill_weights"]
    else:
        support = np.flatnonzero(original)
        target, indices, weights = stencil(g["native_xy"][support], g["native_xy"])
        vertices = support[indices]
    filled = np.full(values.shape, np.nan, dtype="f4")
    filled[target] = interpolate(values, vertices, weights)
    filled[original] = values[original]
    return filled, original


def process_field(raw):
    g = GEOM
    native, original = reconstruct(raw)
    idx, vertices = g["sample_index"], g["sample_vertices"]
    final = np.full(g["land"].size, np.nan, dtype="f4")
    final[idx] = interpolate(native, vertices, g["sample_weights"])
    flag = np.zeros(final.size, dtype="u1")
    influenced = np.any(~original[vertices], axis=1) & np.isfinite(final[idx])
    flag[idx[influenced]] = 1
    gap = g["inside"].ravel() & ~np.isfinite(final)
    flag[gap] |= 2
    final[gap | g["outside"].ravel()] = 0
    before = final[g["dep"]].copy()
    final[g["dep"]] = 0
    return native.reshape(NATIVE), final.reshape(g["land"].shape), flag.reshape(g["land"].shape), before


def output_path(year, month):
    return OUT / "processed" / f"PIOMAS_{year:04d}-{month:02d}_逐日_5km.nc"


def finished(job):
    year, month, start, stop = job
    path = output_path(year, month)
    if not path.is_file():
        return False
    with h5netcdf.File(path, "r") as nc:
        if nc.attrs.get("workflow") != VERSION or nc.attrs.get("complete") != 1 or len(nc.dimensions["time"]) != stop - start:
            raise ValueError(f"Unexpected existing output: {path}")
    return True


def run_month(job):
    year, month, start, stop = job
    g = GEOM
    shape = g["land"].shape
    path = output_path(year, month)
    tmp = path.with_suffix(".partial")
    raw = {}
    for name, var, folder, _ in FIELDS:
        raw[name] = np.memmap(SOURCE / f"decoded/{folder}/{var}.H{year}", dtype="<f4", mode="r",
                              offset=start * np.prod(NATIVE) * 4, shape=(stop - start, *NATIVE))
    with h5netcdf.File(tmp, "w") as nc:
        nc.dimensions = dict(time=stop - start, y=shape[0], x=shape[1],
                             native_y=NATIVE[0], native_x=NATIVE[1], departure=len(g["dep"]))
        time = nc.create_variable("time", ("time",), data=np.arange(start, stop, dtype="i4"))
        time.attrs.update(units=f"days since {year}-01-01 00:00:00", calendar="365_day")
        nc.create_variable("source_day_index", ("time",), data=np.arange(start + 1, stop + 1, dtype="i4"))
        for axis in ("x", "y"):
            nc.create_variable(axis, (axis,), data=g[axis]).attrs["units"] = "m"
        for name, arr in (("native_longitude", g["lon"]), ("native_latitude", g["native_lat"]),
                          ("native_ocean_mask", g["wet"].astype("u1"))):
            nc.create_variable(name, ("native_y", "native_x"), data=arr, compression="gzip")
        for name, key in (("land_mask", "land"), ("AMAP_mask", "amap"), ("marine_domain_mask", "domain"),
                          ("outside_AMAP_assumed_ice_free", "outside")):
            nc.create_variable(name, ("y", "x"), data=g[key].astype("u1"), compression="gzip")
        departure = np.zeros(shape, dtype="u1")
        departure.ravel()[g["dep"]] = 1
        nc.create_variable("departure_zero_mask", ("y", "x"), data=departure, compression="gzip")
        nc.create_variable("departure_row", ("departure",), data=g["rr"])
        nc.create_variable("departure_col", ("departure",), data=g["cc"])
        nc.attrs.update(workflow=VERSION, source="PIOMAS v2.1 aiday and hiday", crs_wkt=g["crs"],
                        thickness_definition="ice volume per total native grid-cell area",
                        calendar_policy="Fixed 365 source days per complete year; monthly groups follow project R1. No extra records.",
                        domain_rule="Formal ocean in AMAP plus formal ocean outside AMAP at latitude <=70N; no buffer",
                        stage_reconstruction="Native interpolated fields plus spatial/5km_线性采样权重.npz reproduce pre-mask AMAP samples",
                        zero_rule="Residual missing marine values filled independently; outside-AMAP marine domain and all departures forced zero")
        for name, _, _, unit in FIELDS:
            options = dict(compression="gzip", compression_opts=1, shuffle=True)
            field = nc.create_variable(name, ("time", "y", "x"), dtype="f4", fillvalue=np.nan,
                                       chunks=(1, 256, 256), **options)
            field.attrs["units"] = unit
            native = nc.create_variable(name + "_native_interpolated", ("time", "native_y", "native_x"),
                                        dtype="f4", fillvalue=np.nan, chunks=(1, *NATIVE), **options)
            native.attrs["units"] = unit
            flag = nc.create_variable(name + "_provenance_flag", ("time", "y", "x"), dtype="u1",
                                      chunks=(1, 256, 256), **options)
            flag.attrs.update(flag_masks=np.array([1, 2], dtype="u1"),
                              flag_meanings="influenced_by_native_gap_interpolation residual_missing_filled_zero")
            before = nc.create_variable(name + "_before_departure_override", ("time", "departure"),
                                        dtype="f4", fillvalue=np.nan, **options)
            for day, values in enumerate(raw[name]):
                native[day], field[day], flag[day], before[day] = process_field(values)
        nc.attrs["complete"] = 1
    tmp.replace(path)
    return stop - start, path.stat().st_size


def source_jobs():
    state = json.loads((SOURCE / "download_state.json").read_text(encoding="utf-8"))["files"]
    jobs = []
    for year in range(1979, 2027):
        lengths = []
        for _, var, folder, _ in FIELDS:
            item = state[f"{var}_{year}"]
            days = int(item["source_days"])
            path = SOURCE / f"decoded/{folder}/{var}.H{year}"
            if item["status"] != "complete" or path.stat().st_size != days * np.prod(NATIVE) * 4:
                raise ValueError(f"Incomplete source: {path}")
            lengths.append(days)
        if lengths[0] != lengths[1] or (year < 2026 and lengths[0] != 365):
            raise ValueError(f"Unmatched source lengths: {year}, {lengths}")
        start = 0
        for month, length in enumerate(MONTHS, 1):
            stop = min(start + length, lengths[0])
            if stop == start:
                break
            jobs.append((year, month, start, stop))
            start = stop
    return jobs


def run(workers):
    for sub in ("processed", "spatial"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    jobs = source_jobs()
    complete = [job for job in jobs if finished(job)]
    completed = set(complete)
    remaining = [job for job in jobs if job not in completed]
    report = dict(status="running", total_months=len(jobs), total_days=sum(j[3] - j[2] for j in jobs),
                  completed_months=len(complete), completed_days=sum(j[3] - j[2] for j in complete),
                  output_bytes=sum(output_path(*job[:2]).stat().st_size for job in complete))

    def update():
        write_json(OUT / "运行状态.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)

    try:
        update()
        if remaining:
            geometry = spatial()
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(geometry,)) as pool:
                futures = {pool.submit(run_month, job): job for job in remaining}
                for future in as_completed(futures):
                    days, size = future.result()
                    year, month, _, _ = futures[future]
                    report["completed_months"] += 1
                    report["completed_days"] += days
                    report["output_bytes"] += size
                    report["last_completed_month"] = f"{year}-{month:02d}"
                    update()
        report["status"] = "complete"
        update()
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        update()
        raise


if __name__ == "__main__":
    import msvcrt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".process.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        run(args.workers)

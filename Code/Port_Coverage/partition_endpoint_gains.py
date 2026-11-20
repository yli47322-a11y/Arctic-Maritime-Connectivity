"""Partition the existing daily connection results by endpoint port evidence."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "正式成果/R2_全区海上联系机会_完整分析"
POINTS = ROOT / "预处理/PIOMAS_5km_标准流程_v1/固定点位"
OUT = ROOT / "正式成果/现有港口支撑的潜在联系机会增量"
GROUPS = ["两端有港口", "仅一端有港口", "两端均未记录港口"]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(name, records):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    OUT.mkdir(exist_ok=True)
    start = time.monotonic()
    settlements = read(POINTS / "聚居点与出发格对应.csv")
    departures = read(POINTS / "出发格.csv")
    departure_index = {row["departure_id"]: index for index, row in enumerate(departures)}
    point_index = np.array([departure_index[row["departure_id"]] for row in settlements])
    is_port = np.array([row["category"] == "港口聚居点" for row in settlements])
    port_mass = np.bincount(point_index, weights=is_port.astype(int), minlength=len(departures))
    other_mass = np.bincount(point_index, weights=(~is_port).astype(int), minlength=len(departures))
    n, p = len(settlements), int(is_port.sum())
    assert (n, p, len(departures)) == (4258, 272, 1248)
    group_denominator = np.array([p * (p - 1) // 2, p * (n - p), (n - p) * (n - p - 1) // 2])
    denominator = n * (n - 1) // 2
    assert group_denominator.sum() == denominator
    with np.load(SOURCE / "R2_完整分析结果.npz") as reference:
        years = reference["years"].copy()
        ships = reference["ships"].copy()
        annual_reference = reference["annual"].copy()
    totals = np.zeros((len(years), len(ships), 3), np.int64)
    # Endpoint-specific yearly linked partner-day counts for the two five-year epochs.
    node_epochs = np.zeros((2, len(ships), len(departures), 2), np.int64)
    annual_rows, monthly_rows = [], []
    checked_days = 0
    for yi, year in enumerate(years):
        epoch = 0 if yi < 5 else 1 if yi >= len(years) - 5 else None
        day_count = 0
        for month in range(1, 13):
            with np.load(SOURCE / "monthly_counts" / f"{year}-{month:02}.npz") as cache:
                labels, expected = cache["labels"], cache["pairs"]
            month_totals = np.zeros((len(ships), 3), np.int64)
            for day in range(len(labels)):
                for si in range(len(ships)):
                    _, inverse = np.unique(labels[day, si], return_inverse=True)
                    port_components = np.bincount(inverse, weights=port_mass)
                    other_components = np.bincount(inverse, weights=other_mass)
                    counts = np.array([
                        np.dot(port_components, port_components - 1) / 2,
                        np.dot(port_components, other_components),
                        np.dot(other_components, other_components - 1) / 2,
                    ], dtype=np.int64)
                    assert counts.sum() == expected[day, si], (year, month, day, si)
                    month_totals[si] += counts
                    if epoch is not None:
                        node_epochs[epoch, si, :, 0] += port_components[inverse].astype(np.int64)
                        node_epochs[epoch, si, :, 1] += other_components[inverse].astype(np.int64)
                checked_days += 1
            totals[yi] += month_totals
            day_count += len(labels)
            for si, ship in enumerate(ships):
                for gi, group in enumerate(GROUPS):
                    monthly_rows.append(dict(year=int(year), month=month, ship_class=str(ship), group=group,
                                             connected_pair_days=int(month_totals[si, gi]), source_days=len(labels),
                                             possible_pairs=int(group_denominator[gi])))
        assert day_count == 365
        assert np.allclose(totals[yi].sum(1) / denominator / 365, annual_reference[yi],
                           rtol=0, atol=1e-12)
        for si, ship in enumerate(ships):
            for gi, group in enumerate(GROUPS):
                annual_rows.append(dict(year=int(year), ship_class=str(ship), group=group,
                                        connected_pair_days=int(totals[yi, si, gi]),
                                        common_denominator_opportunity_pct=totals[yi, si, gi] / denominator / 365 * 100,
                                        within_group_opportunity_pct=totals[yi, si, gi] / group_denominator[gi] / 365 * 100))
        print(f"{year} completed", flush=True)
    early = totals[:5].mean(0)
    late = totals[-5:].mean(0)
    delta = late - early
    delta_all = delta.sum(1)
    summaries, point_rows = [], []
    for si, ship in enumerate(ships):
        for gi, group in enumerate(GROUPS):
            summaries.append(dict(
                ship_class=str(ship), group=group, possible_pairs=int(group_denominator[gi]),
                possible_pair_share_pct=group_denominator[gi] / denominator * 100,
                early_mean_pair_days=float(early[si, gi]), late_mean_pair_days=float(late[si, gi]),
                increase_pair_days_per_year=float(delta[si, gi]),
                increase_common_denominator_pp=float(delta[si, gi] / denominator / 365 * 100),
                share_of_total_net_increase_pct=float(delta[si, gi] / delta_all[si] * 100) if delta_all[si] != 0 else "",
                total_increase_pair_days_per_year=float(delta_all[si])))
        point_delta = (node_epochs[1, si] - node_epochs[0, si])[:, :][point_index] / 5
        # Self-pair subtraction is identical in both epochs, so it cancels from the change.
        for index, settlement in enumerate(settlements):
            blocked = point_delta[index, 1] if is_port[index] else point_delta[index].sum()
            supported = point_delta[index, 0] if is_port[index] else 0.0
            point_rows.append(dict(ship_class=str(ship), entity_id=settlement["entity_id"],
                                   name=settlement["name"], country_code=settlement["country_code"],
                                   category=settlement["category"],
                                   total_increase_partner_days=float(point_delta[index].sum()),
                                   both_ports_increase_partner_days=float(supported),
                                   at_least_one_endpoint_without_documented_port_increase_partner_days=float(blocked)))
    write("12船级_三类点对增量及贡献比例.csv", summaries)
    write("年度三类联系机会_1979-2025.csv", annual_rows)
    write("月度三类联通点对天数_1979-2025.csv", monthly_rows)
    write("聚居地新增机会的港口支撑.csv", point_rows)
    affected = []
    for ship in ships:
        selected = [r for r in point_rows if r["ship_class"] == ship]
        positive = [r for r in selected if r["total_increase_partner_days"] > 0]
        exposed = [r for r in selected if r["at_least_one_endpoint_without_documented_port_increase_partner_days"] > 0]
        fully = [r for r in positive if r["both_ports_increase_partner_days"] <= 0]
        affected.append(dict(ship_class=str(ship), settlements_with_net_gain=len(positive),
                             settlements_with_positive_gain_in_links_lacking_port_evidence=len(exposed),
                             net_gain_settlements_without_positive_both_port_gain=len(fully)))
    write("12船级_涉及聚居地数量.csv", affected)
    np.savez_compressed(OUT / "port_endpoint_partition.npz", years=years, ships=ships,
                        groups=np.array(GROUPS), totals=totals, group_denominator=group_denominator,
                        early=early, late=late, delta=delta, node_epochs=node_epochs)
    metadata = dict(status="complete", early_period="1979-1983", late_period="2021-2025",
                    endpoint_rule="fixed_documented_port_settlement_status_at_both_endpoints",
                    historical_port_construction_dates_used=False, shared_departure_pairs_included=True,
                    self_pairs_excluded=True, settlement_records=n, documented_port_records=p,
                    other_coastal_records=n-p, departure_cells=len(departures),
                    documented_port_departure_cells=len(np.flatnonzero(port_mass)),
                    possible_pairs=denominator, group_denominator=group_denominator.tolist(),
                    source_days=checked_days, checked_against_existing_daily_pair_totals=True,
                    source_table_sha256=hashlib.sha256((POINTS / "聚居点与出发格对应.csv").read_bytes()).hexdigest(),
                    elapsed_seconds=round(time.monotonic()-start, 2))
    (OUT / "计算记录.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

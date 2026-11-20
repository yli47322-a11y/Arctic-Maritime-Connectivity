import csv
import json
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / '正式成果/R2_全区海上联系机会_完整分析'
POINTS = ROOT / '预处理/PIOMAS_5km_标准流程_v1/固定点位'


def read(name):
    with (POINTS / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


@njit
def pair_sums(trace, mass):
    regions = mass.shape[1]
    total = np.zeros((regions, regions), np.int64)
    longest = np.zeros_like(total)
    counts = np.zeros_like(total)
    full = np.zeros_like(total)
    zero = np.zeros_like(total)
    for i in range(len(trace)):
        for j in range(i + 1):
            days, run, best = 0, 0, 0
            if i == j:
                days = best = trace.shape[1]
            else:
                for k in range(trace.shape[1]):
                    if trace[i,k] == trace[j,k]:
                        days += 1
                        run += 1
                        if run > best:
                            best = run
                    else:
                        run = 0
            for a in range(regions):
                ma = mass[i,a]
                if not ma:
                    continue
                for b in range(regions):
                    pairs = ma * mass[j,b]
                    if not pairs:
                        continue
                    if i == j and a == b:
                        pairs -= ma
                    counts[a,b] += pairs
                    total[a,b] += days * pairs
                    longest[a,b] += best * pairs
                    if days == trace.shape[1]:
                        full[a,b] += pairs
                    elif days == 0:
                        zero[a,b] += pairs
    # Directed counts. Identical traces already contribute both orientations.
    return total, longest, full, zero, counts


def metrics(year, membership, size):
    chunks = []
    for month in range(1, 13):
        with np.load(SOURCE / 'monthly_counts' / f'{year}-{month:02}.npz') as cache:
            chunks.append(cache['labels'])
    labels = np.concatenate(chunks)
    days, ships, departures = labels.shape
    assert days == 365 and departures == len(membership)
    region_count = len(size)
    sums = np.zeros((ships, 4, region_count, region_count), np.int64)
    node = np.zeros((ships, departures, region_count), np.int64)
    for c in range(ships):
        trace, inverse = np.unique(labels[:,c].T, return_inverse=True, axis=0)
        mass = np.zeros((len(trace), region_count), np.int64)
        np.add.at(mass, inverse, membership)
        # Include the reverse orientation for pairs from distinct traces only.
        a, b, f, z, count = pair_sums(trace, mass)
        same = mass.T @ mass - np.diag(size)
        count = count + count.T - same
        assert np.array_equal(count, np.outer(size, size)-np.diag(size))
        sums[c] = (a + a.T - same*days, b + b.T - same*days, f + f.T - same, z + z.T)
        trace_node = np.zeros_like(mass)
        for day in range(days):
            _, comp = np.unique(trace[:,day], return_inverse=True)
            by_comp = np.zeros((comp.max()+1, region_count), np.int64)
            np.add.at(by_comp, comp, mass)
            trace_node += by_comp[comp]
        node[c] = trace_node[inverse]
    return sums, node


def main():
    mapping = json.loads((HERE/'assignment.json').read_text(encoding='utf-8'))
    regions = mapping['regions']
    assignment = mapping['assignments']
    dep = {r['departure_id']:i for i,r in enumerate(read('出发格.csv'))}
    records = read('聚居点与出发格对应.csv')
    region = np.array([assignment[r['entity_id']] for r in records])
    index = np.array([dep[r['departure_id']] for r in records])
    membership = np.zeros((len(dep), len(regions)), np.int64)
    np.add.at(membership, (index, region), 1)
    size = membership.sum(0)
    with np.load(SOURCE/'R2_完整分析结果.npz') as ref:
        years, ships, annual, histogram = (ref[k] for k in ('years','ships','annual','hist'))
    shape = (len(years), len(ships), len(size))
    sums = np.zeros((*shape[:2],4,len(size),len(size)), np.int64)
    nodes = np.zeros((*shape[:2],3,len(records)), np.float32)
    for i,year in enumerate(years):
        sums[i], node = metrics(year, membership, size)
        exact = sums[i,:,0].sum(axis=(1,2))
        assert np.allclose(exact/(len(records)*(len(records)-1)*365), annual[i], atol=1e-12)
        assert np.array_equal(sums[i,:,1].sum(axis=(1,2)), 2*(histogram[i,:,1] @ np.arange(366)))
        all_dest = node[:,index,:].sum(-1)-365
        within = node[:,index,region]-365
        nodes[i,:,0] = 100 * all_dest/(365*(len(records)-1))
        nodes[i,:,1] = np.divide(100*within,365*(size[region]-1),out=np.full_like(within,np.nan,dtype=float),where=size[region]>1)
        nodes[i,:,2] = 100*(all_dest-within)/(365*(len(records)-size[region]))
        print(int(year), flush=True)
    np.savez_compressed(HERE/'regional.npz', years=years, ships=ships, size=size, sums=sums, nodes=nodes, region=region)


if __name__ == '__main__':
    main()

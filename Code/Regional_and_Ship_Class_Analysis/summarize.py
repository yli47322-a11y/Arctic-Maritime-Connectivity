import json

import numpy as np

from analyze import HERE


def main():
    f = np.load(HERE/'regional.npz')
    a = json.loads((HERE/'assignment.json').read_text(encoding='utf-8'))
    years, ships, size, sums, nodes, region = [f[k] for k in ('years','ships','size','sums','nodes','region')]
    n = size.sum()
    r = len(size)
    den = np.outer(size,size)-np.diag(size)
    assert np.min(den)>0
    selectors = {
        'all': lambda x: x.sum(-1),
        'within': lambda x: np.diagonal(x,axis1=-2,axis2=-1),
        'cross': lambda x: x.sum(-1)-np.diagonal(x,axis1=-2,axis2=-1),
    }
    num = sums[:,:,0]
    pair = num/den*100/365
    windows = sums[:,:,1]
    report = {k:a[k] for k in ('regions','summary','nearest_count','max_km')}
    report.update(years=years.tolist(),ships=ships.tolist(),modes={})
    base, end = slice(0,5), slice(-5,None)
    slope = (years-years.mean())*10/np.sum((years-years.mean())**2)
    decomp = {}
    focus = list(range(2,12))
    for j,(mode,sel) in enumerate(selectors.items()):
        rates = sel(num)/sel(den)*100/365
        longest = sel(windows)/sel(den)
        gap = rates[:,:,:].max(1)-rates[:,:,:].min(1)
        mean_r = rates.mean(2)
        sd_r = rates.std(2)
        sd_ship = rates.std(1)
        resid = rates - rates.mean(2,keepdims=True) - rates.mean(1,keepdims=True) + rates.mean((1,2),keepdims=True)
        interaction = np.sqrt((resid**2).mean((1,2)))
        within_sd = np.stack([nodes[:,:,j,region==k].std(-1) for k in range(r)],axis=-1)
        early_node = nodes[base,:,j].mean(0)
        late_node = nodes[end,:,j].mean(0)
        low = early_node <= np.quantile(early_node,.2,axis=1,keepdims=True)
        avg_early, avg_late = rates[base].mean(0),rates[end].mean(0)
        four_types = np.stack([sel(sums[:,:,k])/sel(den) for k in (2,3)],axis=2)*100
        report['modes'][mode] = {
            'annual':rates.tolist(), 'early':avg_early.tolist(), 'late':avg_late.tolist(),
            'trend':np.einsum('y,ycr->cr',slope,rates).tolist(),
            'regional_sd':sd_r.tolist(),'regional_mean':mean_r.tolist(),
            'ship_sd_by_region':sd_ship.tolist(),'ship_span':gap.tolist(),
            'interaction':interaction.tolist(),
            'interaction_change':(resid[end].mean(0)-resid[base].mean(0)).tolist(),
            'within_node_sd_early':within_sd[base].mean(0).tolist(),
            'within_node_sd_late':within_sd[end].mean(0).tolist(),
            'low_cohort':[[int(low[c].sum()),int((~low[c]).sum()),float((late_node[c]-early_node[c])[low[c]].mean()),
                           float((late_node[c]-early_node[c])[~low[c]].mean()) if (~low[c]).any() else None] for c in range(len(ships))],
            'longest_early':longest[base].mean(0).tolist(), 'longest_late':longest[end].mean(0).tolist(),
            'full_early':four_types[base,:,0].mean(0).tolist(),'full_late':four_types[end,:,0].mean(0).tolist(),
            'zero_early':four_types[base,:,1].mean(0).tolist(),'zero_late':four_types[end,:,1].mean(0).tolist(),
        }
        # Separate the spatial variance of settlement opportunities by the fixed geographic groups.
        if mode=='all':
            for c in focus:
                def partition(x):
                    group = np.array([x[region==k].mean() for k in range(r)])
                    return [float(x.var()),float(np.average((group-x.mean())**2,weights=size)),
                            float(np.average([x[region==k].var() for k in range(r)],weights=size))]
                decomp[str(ships[c])] = [partition(early_node[c]),partition(late_node[c])]
    report['node_variance'] = decomp
    report['pair_early'] = pair[base].mean(0).tolist()
    report['pair_late'] = pair[end].mean(0).tolist()
    report['pair_trend'] = np.einsum('y,ycrs->crs',slope,pair).tolist()
    report['same_dest'] = []
    for c in (3,4,5,11):
        low, high = np.argmin(report['modes']['cross']['early'][c]),np.argmax(report['modes']['cross']['early'][c])
        valid = [k for k in range(r) if k not in (low,high)]
        before = pair[base,c].mean(0)
        after = pair[end,c].mean(0)
        report['same_dest'].append({'ship':int(c),'low':int(low),'high':int(high),'valid':valid,
            'early':(before[high,valid]-before[low,valid]).tolist(),'late':(after[high,valid]-after[low,valid]).tolist()})
    (HERE/'report.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for m in ('all','within','cross'):
        d=report['modes'][m]
        v=np.array(d['regional_sd'])
        print(m,'regional SD early/late',np.round(v[base].mean(0),1),np.round(v[end].mean(0),1))
        q=np.array(d['ship_sd_by_region'])
        print(m,'ship SD avg early/late',q[base].mean(),q[end].mean())
        z=np.array(d['interaction'])
        print(m,'interaction RMS early/late',z[base].mean(),z[end].mean())


if __name__ == '__main__':
    main()

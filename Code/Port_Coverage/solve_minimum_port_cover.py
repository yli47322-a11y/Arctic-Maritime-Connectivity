"""Minimum additional settlement-sited ports under the fixed 100-km/land rule."""
from pathlib import Path
import csv
import json
from collections import Counter
import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize import milp, Bounds, LinearConstraint
from pyproj import Geod

ROOT=Path(__file__).resolve().parent.parent
IN=ROOT/'正式成果/港口100km陆地连通支撑的潜在联系机会增量/4258聚居地_港口100km覆盖与最近港口.csv'
OUT=ROOT/'正式成果/港口100km陆地连通_新增港口最小覆盖'
OUT.mkdir(exist_ok=True)
with IN.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
target=np.array([i for i,r in enumerate(rows) if r['port_supported_100km']=='0'])
candidates=np.array([i for i,r in enumerate(rows) if r['original_category']!='港口聚居点'])
lon=np.array([float(r['longitude']) for r in rows]); lat=np.array([float(r['latitude']) for r in rows])
comp=np.array([int(r['land_component_1km']) for r in rows])
geod=Geod(ellps='WGS84')
ir=[]; jc=[]
for j,i in enumerate(candidates):
    _,_,d=geod.inv(np.full(len(target),lon[i]),np.full(len(target),lat[i]),lon[target],lat[target])
    cover=np.flatnonzero((d<=100000+1e-6)&(comp[target]==comp[i]))
    ir.extend(cover.tolist()); jc.extend([j]*len(cover))
A=csc_matrix((np.ones(len(ir)),(ir,jc)),shape=(len(target),len(candidates)))
assert np.all(np.asarray(A.sum(axis=1)).ravel()>0)
print(f'Targets={len(target)} candidates={len(candidates)} links={A.nnz}',flush=True)
result=milp(np.ones(len(candidates)),integrality=np.ones(len(candidates)),
            bounds=Bounds(0,1),constraints=LinearConstraint(A,1,np.inf),
            options={'mip_rel_gap':0.0,'disp':True})
if result.x is None: raise RuntimeError(str(result.message))
selected=candidates[result.x>0.5]
assert np.all(A @ (result.x>0.5) >= 1)
print(f'Selected={len(selected)} status={result.status} gap={result.mip_gap}',flush=True)
def write(name,records):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
selected_rows=[]
for rank,i in enumerate(selected,1):
    _,_,d=geod.inv(np.full(len(target),lon[i]),np.full(len(target),lat[i]),lon[target],lat[target])
    covered=target[(d<=100000+1e-6)&(comp[target]==comp[i])]
    selected_rows.append(dict(new_port_id=f'NEW{rank:04}',entity_id=rows[i]['entity_id'],name=rows[i]['name'],
                              country_code=rows[i]['country_code'],longitude=lon[i],latitude=lat[i],
                              land_component_1km=comp[i],unserved_settlements_within_coverage=len(covered)))
write('新增港口_最小覆盖选址.csv',selected_rows)
assignments=[]
for i in target:
    _,_,d=geod.inv(np.full(len(selected),lon[i]),np.full(len(selected),lat[i]),lon[selected],lat[selected])
    d=np.where((comp[selected]==comp[i])&(d<=100000+1e-6),d,np.inf)
    k=int(np.argmin(d));assert np.isfinite(d[k])
    assignments.append(dict(entity_id=rows[i]['entity_id'],name=rows[i]['name'],country_code=rows[i]['country_code'],
                            new_port_id=selected_rows[k]['new_port_id'],new_port_name=selected_rows[k]['name'],
                            new_port_country=selected_rows[k]['country_code'],distance_km=float(d[k]/1000)))
write('未覆盖聚居地_新增港口服务分配.csv',assignments)
country=[]
for code in sorted({r['country_code'] for r in rows}):
    group=[r for r in rows if r['country_code']==code]
    country.append(dict(country_code=code,coastal_settlements=len(group),
                        existing_port_records=sum(r['original_category']=='港口聚居点' for r in group),
                        already_supported=sum(r['port_supported_100km']=='1' for r in group),
                        currently_unserved=sum(r['port_supported_100km']=='0' for r in group),
                        new_ports=sum(r['country_code']==code for r in selected_rows),
                        new_port_share_pct=sum(r['country_code']==code for r in selected_rows)/len(selected)*100))
write('国家地区_新增港口需求.csv',country)
meta=dict(status='complete',solver_message=result.message,optimal=result.status==0,mip_gap=result.mip_gap,
          optimal_lower_bound=result.mip_dual_bound,additional_ports=len(selected),
          existing_port_records=272,existing_supported_settlements=len(rows)-len(target),unserved_settlements=len(target),
          candidates=len(candidates),candidate_rule='all non-port settlement points, including already supported points',
          coverage_rule='WGS84 geodesic <=100km and identical existing 1km four-neighbour land component',
          target_rule=f'all {len(target)} currently unsupported settlement records',
          geographic_scope='existing 4258 AMAP coastal settlement records; not latitude>=66.56 subset',
          cross_country_coverage_allowed=True,solution_ties_possible=True,
          capacities_considered=False,actual_construction_feasibility_considered=False)
(OUT/'计算记录.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(country,ensure_ascii=False),flush=True)

"""Rebuild all canonical records from final land cells with AMAP-constrained nearest ocean."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,hashlib,shutil
import numpy as np
import rasterio
from scipy.spatial import cKDTree
from scipy.ndimage import label,generate_binary_structure
from pyproj import Transformer,Geod

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT/'正式成果/空间基准_5km_最终正式版_v2_外海连通_2026-11-04'
OUT=ROOT/'正式成果/空间基准_5km_最终正式版_v3_全量最近出海_2026-11-04'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rr):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def geo(p,rr):
 p.write_text(json.dumps(dict(type='FeatureCollection',features=[dict(type='Feature',geometry=dict(type='Point',coordinates=[float(r['longitude']),float(r['latitude'])]),properties=r) for r in rr]),ensure_ascii=False),encoding='utf-8')
def main():
 if OUT.exists():raise RuntimeError('v3 exists; do not overwrite silently')
 for n in ['01_最终点位与对应','02_原始清单与筛选溯源','03_正式5km栅格','04_规则与处理记录','05_展示图']:(OUT/n).mkdir(parents=True,exist_ok=True)
 ss=read(OLD/'01_最终点位与对应/最终聚居点_4258条.csv')
 oldrel={r['entity_id']:r for r in read(OLD/'01_最终点位与对应/出发点与聚居点对应_4258条.csv')}
 old16=read(OLD/'01_最终点位与对应/16条陆地定位调整记录.csv')
 stale16=[r for r in old16 if r['departure_id']!=oldrel[r['entity_id']]['departure_id']]
 with rasterio.open(OLD/'03_正式5km栅格/land_mask_5km.tif') as d:land=d.read(1);tf=d.transform;crs=d.crs;profile=d.profile
 with rasterio.open(OLD/'03_正式5km栅格/AMAP_mask_5km.tif') as d:amap=d.read(1)
 c,n=label(land==0,generate_binary_structure(2,1));j0,i0=rasterio.transform.rowcol(tf,0,0);ocean=c==c[j0,i0]
 jj,ii=np.where(ocean&(amap==1));tree=cKDTree(np.column_stack((jj,ii)))
 cells=sorted(set((int(s['land_grid_row']),int(s['land_grid_col'])) for s in ss));cache={}
 for j,i in cells:
  d,k=tree.query([j,i]);ix=np.asarray(tree.query_ball_point([j,i],float(d)+1e-8),int)
  ds=(jj[ix]-j)**2+(ii[ix]-i)**2;ix=ix[ds==ds.min()];q=min(ix,key=lambda k:(jj[k],ii[k]))
  cache[j,i]=(int(jj[q]),int(ii[q]),len(ix),float(d))
 # Boundary diagnostic only. All actual mappings remain AMAP-constrained.
 uj,ui=np.where(ocean);utree=cKDTree(np.column_stack((uj,ui)))
 uq=np.array([[int(s['land_grid_row']),int(s['land_grid_col'])] for s in ss]);ud,_=utree.query(uq)
 toll=Transformer.from_crs(crs,4326,always_xy=True);toxy=Transformer.from_crs(4326,crs,always_xy=True);geod=Geod(ellps='WGS84')
 relations=[];changes=[];impact=[];group=defaultdict(list);nonminimal=[]
 for ix,s in enumerate(ss):
  lj,li=int(s['land_grid_row']),int(s['land_grid_col']);j,i,nt,dist=cache[lj,li]
  o=oldrel[s['entity_id']];oj,oi=int(o['departure_row']),int(o['departure_col']);od=float(np.hypot(oj-lj,oi-li))
  strictbad=od>dist+1e-8
  if strictbad:nonminimal.append(o)
  did=f'DEP5K_R{j:04d}_C{i:04d}';x,y=rasterio.transform.xy(tf,j,i);lo,la=toll.transform(x,y)
  px,py=toxy.transform(float(s['longitude']),float(s['latitude']));boundary=dist>ud[ix]+1e-8
  r=dict(entity_id=s['entity_id'],name=s['name'],country_code=s['country_code'],category=s['category'],
    settlement_longitude=float(s['longitude']),settlement_latitude=float(s['latitude']),settlement_land_row=lj,settlement_land_col=li,
    departure_id=did,departure_row=j,departure_col=i,departure_center_x_m=float(x),departure_center_y_m=float(y),
    departure_longitude=float(lo),departure_latitude=float(la),
    settlement_land_cell_to_departure_km=round(dist*5,6),
    representative_to_departure_projected_km=round(float(np.hypot(x-px,y-py)/1000),6),
    representative_to_departure_geodesic_km=round(float(geod.inv(float(s['longitude']),float(s['latitude']),lo,la)[2]/1000),6),
    equidistant_candidates=nt,AMAP_constrained=True,connected_to_outer_ocean_4=True,
    nearest_within_AMAP_verified=True,AMAP_boundary_increases_minimum_distance=bool(boundary))
  relations.append(r);group[j,i].append(r)
  if (j,i)!=(oj,oi):changes.append(dict(r,previous_departure_id=o['departure_id'],previous_departure_row=oj,previous_departure_col=oi,
   previous_landcell_distance_km=round(od*5,6),change_reason='原记录非最近格' if strictbad else '同距离候选统一按行列号排序'))
  if boundary:impact.append(dict(r,previous_departure_id=o['departure_id'],unconstrained_nearest_landcell_distance_km=round(float(ud[ix]*5),6),
    AMAP_constraint_extra_landcell_distance_km=round(float((dist-ud[ix])*5),6)))
  s.pop('previous_departure_id',None);s['departure_id']=did
 dep=[]
 for ind,((j,i),rr) in enumerate(sorted(group.items()),1):
  r=rr[0];dep.append(dict(departure_id=r['departure_id'],departure_index=ind,row=j,col=i,center_x_m=r['departure_center_x_m'],center_y_m=r['departure_center_y_m'],
   longitude=r['departure_longitude'],latitude=r['departure_latitude'],settlement_record_count=len(rr),
   port_settlement_count=sum(x['category']=='港口聚居点' for x in rr),coastal_settlement_count=sum(x['category']=='沿海聚居点' for x in rr),
   countries=';'.join(sorted(set(x['country_code'] for x in rr))),settlement_names=';'.join(x['name'] for x in rr),settlement_entity_ids=';'.join(x['entity_id'] for x in rr),
   max_land_cell_to_departure_km=max(x['settlement_land_cell_to_departure_km'] for x in rr),connected_to_outer_ocean_4=True))
 tab=OUT/'01_最终点位与对应';rec=OUT/'04_规则与处理记录'
 write(tab/'最终聚居点_4258条.csv',ss);geo(tab/'最终聚居点_4258条.geojson',ss)
 write(tab/f'最终出发点_{len(dep)}个.csv',dep);geo(tab/f'最终出发点_{len(dep)}个.geojson',dep)
 write(tab/'出发点与聚居点对应_4258条.csv',relations)
 write(tab/'16条陆地定位调整记录.csv',[s for s in ss if float(s['position_shift_projected_km'])>0])
 if changes:write(rec/'v2至v3对应关系变化.csv',changes)
 if impact:write(rec/'AMAP边界限制影响_逐条.csv',impact)
 if nonminimal:write(rec/'v2非最近出海记录.csv',nonminimal)
 if stale16:write(rec/'v2残留旧出发编号_定位明细.csv',stale16)
 shutil.copytree(OLD/'02_原始清单与筛选溯源',OUT/'02_原始清单与筛选溯源',dirs_exist_ok=True)
 coast=ROOT/'正式成果/空间基准_5km_最终正式版_2026-11-04/04_规则与处理记录/海岸线与AMAP来源'
 shutil.copytree(coast,rec/'海岸线与AMAP来源',dirs_exist_ok=True)
 for name in ['land_mask_5km.tif','AMAP_mask_5km.tif','marine_geometry_mask_5km.tif','open_ocean_connected_mask_5km.tif']:
  shutil.copy2(OLD/'03_正式5km栅格'/name,OUT/'03_正式5km栅格'/name)
 index=np.zeros(land.shape,dtype='int32')
 for r in dep:index[r['row'],r['col']]=r['departure_index']
 profile.update(dtype='int32',nodata=0)
 with rasterio.open(OUT/'03_正式5km栅格/departure_index_5km.tif','w',**profile) as d:
  d.write(index,1);d.update_tags(index_definition='joins v3 final departure table departure_index')
 # Audit saved canonical records and geometry against one authoritative mapping.
 canon={r['entity_id']:r for r in relations};depmap={r['departure_id']:r for r in dep};inconsistencies=[]
 for name in ['最终聚居点_4258条.csv','16条陆地定位调整记录.csv']:
  for r in read(tab/name):
   if r['departure_id']!=canon[r['entity_id']]['departure_id']:inconsistencies.append([name,r['entity_id']])
 for name in ['最终聚居点_4258条.geojson',f'最终出发点_{len(dep)}个.geojson']:
  g=json.loads((tab/name).read_text(encoding='utf-8'))
  for f in g['features']:
   r=f['properties']
   if 'entity_id' in r and r['departure_id']!=canon[r['entity_id']]['departure_id']:inconsistencies.append([name,r['entity_id']])
 for r in relations:
  j,i=r['departure_row'],r['departure_col'];lj,li=r['settlement_land_row'],r['settlement_land_col']
  assert land[lj,li]==1 and ocean[j,i] and amap[j,i]==1
  assert abs(np.hypot(j-lj,i-li)-cache[lj,li][3])<1e-8
  assert index[j,i]==depmap[r['departure_id']]['departure_index']
 assert len(relations)==4258 and len(canon)==4258 and not inconsistencies
 assert sum(r['settlement_record_count'] for r in dep)==4258
 oldset={r['departure_id'] for r in oldrel.values()};newset=set(depmap)
 def stats(rr,key):
  a=np.array([float(r[key]) for r in rr]);return dict(n=len(a),min_km=float(a.min()),median_km=float(np.median(a)),mean_km=float(a.mean()),p95_km=float(np.percentile(a,95)),max_km=float(a.max()))
 summary=dict(version='v3_all_records_nearest_within_AMAP',settlement_records=4258,settlement_land_cells=2748,
  previous_departures=len(oldset),final_departures=len(dep),new_departure_cells=len(newset-oldset),removed_departure_cells=len(oldset-newset),
  previous_not_nearest_records=len(nonminimal),previous_not_nearest_departures=len(set(r['departure_id'] for r in nonminimal)),
  changed_correspondence_records=len(changes),changed_due_distance=sum(r['change_reason']=='原记录非最近格' for r in changes),changed_due_tie_rule=sum(r['change_reason']!='原记录非最近格' for r in changes),
  previous_stale_departure_references_in_16row_table=len(stale16),current_stale_departure_references=0,
  current_not_nearest_records=0,current_blocked_departures=0,current_departures_outside_AMAP=0,
  AMAP_constraint_affected_records=len(impact),AMAP_constraint_affected_previous_departures=len(set(r['previous_departure_id'] for r in impact)),AMAP_constraint_affected_current_departures=len(set(r['departure_id'] for r in impact)),
  previous_distance_stats=stats(list(oldrel.values()),'representative_to_departure_geodesic_km'),
  current_distance_stats=stats(relations,'representative_to_departure_geodesic_km'),
  max_current_distance_record=max(relations,key=lambda r:r['representative_to_departure_geodesic_km']),
  grid_spacing_m=5000,grid_shape=list(land.shape),nearest_rule='final land-cell center to nearest AMAP-inside sea-cell center belonging to full-grid four-neighbor Arctic Ocean component; Euclidean; ties row,col ascending',
  land_mask_changed=False,settlement_coordinates_changed=False,TOPAZ_processed=False,
  water_paths_use_full_raster=True,departure_candidates_AMAP_only=True,
  land_mask_sha256=hashlib.sha256((OUT/'03_正式5km栅格/land_mask_5km.tif').read_bytes()).hexdigest())
 (OUT/'版本摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 (rec/'正式记录一致性检查.json').write_text(json.dumps({k:v for k,v in summary.items() if any(t in k for t in ['stale','nearest_records','blocked','outside_AMAP'])},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in summary.items() if k!='max_current_distance_record'},ensure_ascii=False,indent=2));print('CHANGES',json.dumps(changes,ensure_ascii=False))

if __name__=='__main__':main()

"""User-authorized remapping to the 4-connected open-ocean component; mask unchanged."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,hashlib,shutil,zipfile
import numpy as np
import rasterio
from scipy.ndimage import label,generate_binary_structure
from scipy.spatial import cKDTree
from pyproj import Transformer,Geod

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT/'正式成果/空间基准_5km_最终正式版_2026-11-04'
OUT=ROOT/'正式成果/空间基准_5km_最终正式版_v2_外海连通_2026-11-04'

def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def geo(p,rows):
 p.write_text(json.dumps(dict(type='FeatureCollection',features=[dict(type='Feature',
  geometry=dict(type='Point',coordinates=[float(r['longitude']),float(r['latitude'])]),properties=r) for r in rows]),ensure_ascii=False),encoding='utf-8')

def main():
 if OUT.exists():raise RuntimeError('v2 directory exists; do not overwrite silently.')
 for n in ['01_最终点位与对应','02_原始清单与筛选溯源','03_正式5km栅格','04_规则与处理记录','05_展示图']:(OUT/n).mkdir(parents=True,exist_ok=True)
 mask=OLD/'03_正式5km栅格/land_mask_5km.tif';maskhash=hashlib.sha256(mask.read_bytes()).hexdigest()
 with rasterio.open(mask) as d:land=d.read(1);tf=d.transform;crs=d.crs;profile=d.profile
 with rasterio.open(OLD/'03_正式5km栅格/AMAP_mask_5km.tif') as d:amap=d.read(1)
 sea=land==0;c,n=label(sea,generate_binary_structure(2,1));pj,pi=rasterio.transform.rowcol(tf,0,0);ocean_id=int(c[pj,pi]);ocean=c==ocean_id
 jj,ii=np.where(ocean);tree=cKDTree(np.column_stack((jj,ii)))
 # AMAP continues to define output/source nodes; the water path may cross its boundary.
 # Candidate departure cells stay inside AMAP, while their ocean connectivity uses full footprint.
 candidate=ocean&(amap==1);cj,ci=np.where(candidate);tree=cKDTree(np.column_stack((cj,ci)))
 settlements=read(OLD/'01_最终点位与对应/最终聚居点_4258条.csv')
 olddeps={r['departure_id']:r for r in read(OLD/'01_最终点位与对应/最终出发点_1331个.csv')}
 oldrelations={r['entity_id']:r for r in read(OLD/'01_最终点位与对应/出发点与聚居点对应_4258条.csv')}
 blocked={did for did,r in olddeps.items() if not ocean[int(r['row']),int(r['col'])]}
 assert len(blocked)==188
 cache={};groups=defaultdict(list);relations=[];adjustments=[]
 toll=Transformer.from_crs(crs,4326,always_xy=True);toxy=Transformer.from_crs(4326,crs,always_xy=True);geod=Geod(ellps='WGS84')
 for s in settlements:
  oldid=s['departure_id'];o=olddeps[oldid];oj,oi=int(o['row']),int(o['col'])
  lj,li=int(s['land_grid_row']),int(s['land_grid_col']);tie=1
  if oldid not in blocked:
   j,i=oj,oi;method='保留原已连外海出发格'
  else:
   if (lj,li) not in cache:
    dist,k=tree.query([lj,li]);inds=np.asarray(tree.query_ball_point([lj,li],float(dist)+1e-8),dtype=int)
    ds=(cj[inds]-lj)**2+(ci[inds]-li)**2;inds=inds[ds==ds.min()]
    q=min(inds,key=lambda k:(cj[k],ci[k]));cache[(lj,li)]=(int(cj[q]),int(ci[q]),len(inds))
   j,i,tie=cache[(lj,li)];method='原封闭出发格_从最终陆格重映射至最近外海连通海格'
  did=f'DEP5K_R{j:04d}_C{i:04d}';x,y=rasterio.transform.xy(tf,j,i);lo,la=toll.transform(x,y)
  ox,oy=rasterio.transform.xy(tf,oj,oi);lx,ly=rasterio.transform.xy(tf,lj,li)
  px,py=toxy.transform(float(s['longitude']),float(s['latitude']))
  z=dict(entity_id=s['entity_id'],name=s['name'],country_code=s['country_code'],category=s['category'],
   settlement_longitude=float(s['longitude']),settlement_latitude=float(s['latitude']),
   settlement_land_row=lj,settlement_land_col=li,
   departure_id=did,departure_row=j,departure_col=i,departure_longitude=float(lo),departure_latitude=float(la),
   departure_center_x_m=float(x),departure_center_y_m=float(y),previous_departure_id=oldid,
   previous_departure_row=oj,previous_departure_col=oi,previous_departure_blocked=int(oldid in blocked),
   departure_mapping_method=method,equidistant_candidates=tie,
   settlement_land_cell_to_departure_km=round(float(np.hypot(x-lx,y-ly)/1000),6),
   representative_to_departure_projected_km=round(float(np.hypot(x-px,y-py)/1000),6),
   representative_to_departure_geodesic_km=round(float(geod.inv(float(s['longitude']),float(s['latitude']),lo,la)[2]/1000),6),
   departure_relocation_projected_km=round(float(np.hypot(x-ox,y-oy)/1000),6),
   connected_to_outer_ocean_4=True)
  relations.append(z);groups[(j,i)].append(z)
  if oldid in blocked:adjustments.append(z)
  s['previous_departure_id']=oldid;s['departure_id']=did
 dep=[]
 for index,((j,i),rr) in enumerate(sorted(groups.items()),1):
  r=rr[0]
  dep.append(dict(departure_id=r['departure_id'],departure_index=index,row=j,col=i,
   center_x_m=r['departure_center_x_m'],center_y_m=r['departure_center_y_m'],
   longitude=r['departure_longitude'],latitude=r['departure_latitude'],settlement_record_count=len(rr),
   port_settlement_count=sum(x['category']=='港口聚居点' for x in rr),
   coastal_settlement_count=sum(x['category']=='沿海聚居点' for x in rr),
   countries=';'.join(sorted(set(x['country_code'] for x in rr))),
   settlement_names=';'.join(x['name'] for x in rr),settlement_entity_ids=';'.join(x['entity_id'] for x in rr),
   connected_to_outer_ocean_4=True,
   max_land_cell_to_departure_km=max(x['settlement_land_cell_to_departure_km'] for x in rr)))
 tab=OUT/'01_最终点位与对应'
 write(tab/'最终聚居点_4258条.csv',settlements);geo(tab/'最终聚居点_4258条.geojson',settlements)
 write(tab/f'最终出发点_{len(dep)}个.csv',dep);geo(tab/f'最终出发点_{len(dep)}个.geojson',dep)
 write(tab/'出发点与聚居点对应_4258条.csv',relations)
 write(tab/'封闭出发点重映射_1497条.csv',adjustments)
 shutil.copy2(OLD/'01_最终点位与对应/16条陆地定位调整记录.csv',tab/'16条陆地定位调整记录.csv')
 shutil.copytree(OLD/'02_原始清单与筛选溯源',OUT/'02_原始清单与筛选溯源',dirs_exist_ok=True)
 shutil.copytree(OLD/'04_规则与处理记录',OUT/'04_规则与处理记录/v1历史规则与来源',dirs_exist_ok=True)
 for name in ['land_mask_5km.tif','AMAP_mask_5km.tif','marine_geometry_mask_5km.tif']:
  shutil.copy2(OLD/'03_正式5km栅格'/name,OUT/'03_正式5km栅格'/name)
 tags=dict(role='static_land_connectivity_not_ice_passability',ocean_reference='component containing North Pole, four neighbors',AMAP_applied='false; full grid water paths retained')
 profile.update(dtype='uint8',nodata=255)
 with rasterio.open(OUT/'03_正式5km栅格/open_ocean_connected_mask_5km.tif','w',**profile) as d:
  d.write(np.where(land==1,0,np.where(ocean,1,2)).astype('uint8'),1)
  d.update_tags(**tags,values='0=land,1=outer_ocean_connected,2=disconnected_marine')
 labels=np.zeros(land.shape,dtype='int32')
 for r in dep:labels[r['row'],r['col']]=r['departure_index']
 profile.update(dtype='int32',nodata=0)
 with rasterio.open(OUT/'03_正式5km栅格/departure_index_5km.tif','w',**profile) as d:
  d.write(labels,1);d.update_tags(index_definition='joins departure_index in v2 departure table')
 # Recheck all final points against the unchanged full-extent four-neighbor labeling.
 assert len(settlements)==4258 and len(set(s['entity_id'] for s in settlements))==4258
 assert all(ocean[r['row'],r['col']] for r in dep)
 assert all(land[int(s['land_grid_row']),int(s['land_grid_col'])]==1 for s in settlements)
 assert all(s['departure_id'] in {r['departure_id'] for r in dep} for s in settlements)
 assert len(adjustments)==1497 and sum(r['settlement_record_count'] for r in dep)==4258
 assert all(s['longitude']==read_s['longitude'] and s['latitude']==read_s['latitude'] for s,read_s in zip(settlements,read(OLD/'01_最终点位与对应/最终聚居点_4258条.csv')))
 assert hashlib.sha256((OUT/'03_正式5km栅格/land_mask_5km.tif').read_bytes()).hexdigest()==maskhash
 split=defaultdict(set)
 for r in adjustments:split[r['previous_departure_id']].add(r['departure_id'])
 oldids=set(olddeps);newids={r['departure_id'] for r in dep}
 summary=dict(version='v2_outer_ocean_connected',date='2026-11-04',settlement_records=4258,
   previous_departure_count=1331,previous_open_departures_unchanged=1143,
   previous_blocked_departures_remapped=188,affected_settlement_records=1497,
   final_departure_count=len(dep),final_blocked_departures=0,
   newly_created_departure_cells=len(newids-oldids),removed_departure_cells=len(oldids-newids),
   remapped_previous_departures_split=sum(len(v)>1 for v in split.values()),
   distinct_targets_for_remapped_records=len(set(r['departure_id'] for r in adjustments)),
   relocation_distance_km={k:round(float(np.percentile([r['departure_relocation_projected_km'] for r in adjustments],q)),6) for k,q in [('min',0),('median',50),('p95',95),('max',100)]},
   affected_landcell_to_departure_km={k:round(float(np.percentile([r['settlement_land_cell_to_departure_km'] for r in adjustments],q)),6) for k,q in [('min',0),('median',50),('p95',95),('max',100)]},
   max_relocation_record=max(adjustments,key=lambda r:r['departure_relocation_projected_km']),
   grid_spacing_m=5000,grid_shape=list(land.shape),settlement_land_cells=2748,
   four_neighbor_paths_use_full_raster=True,AMAP_limits_nodes_and_reporting_not_water_paths=True,
   nearest_candidates_restricted_to_AMAP=True,tie_break='minimum center Euclidean distance, then row,col ascending',
   land_mask_changed=False,land_mask_sha256=maskhash,settlement_positions_changed=False,
   TOPAZ_processed=False,water_depth_used=False,previous_version_preserved=str(OLD.relative_to(ROOT)))
 (OUT/'版本摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 write(OUT/'04_规则与处理记录/预处理前_修复后出发点连通检查.csv',
  [dict(departure_id=r['departure_id'],row=r['row'],col=r['col'],land_value=int(land[r['row'],r['col']]),
   component_4=int(c[r['row'],r['col']]),connected_to_outer_ocean_4=True,settlement_record_count=r['settlement_record_count']) for r in dep])
 print(json.dumps({k:v for k,v in summary.items() if k!='max_relocation_record'},ensure_ascii=False,indent=2))
 print('MAX',json.dumps(summary['max_relocation_record'],ensure_ascii=False))

if __name__=='__main__':main()

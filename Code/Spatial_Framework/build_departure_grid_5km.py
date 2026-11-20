"""Map every retained record to a 5km geometric sea departure cell, per user rule."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,hashlib,shutil,zipfile
import numpy as np
import rasterio
from scipy.spatial import cKDTree
from pyproj import Transformer

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
SRC=ROOT/'海陆几何_5km对照_2026-11-02';OUT=ROOT/'正式空间基准_5km'

def write(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 OUT.mkdir(exist_ok=True)
 mask=SRC/'GSHHG_land_alltouched_5km.tif';boundary=SRC/'AMAP_center_mask_5km.tif'
 with rasterio.open(mask) as d:land=d.read(1);profile=d.profile;tf=d.transform;crs=d.crs
 with rasterio.open(boundary) as d:amap=d.read(1)
 marine=(land==0)&(amap==1)
 jj,ii=np.where(marine);tree=cKDTree(np.column_stack((jj,ii)))
 rows=list(csv.DictReader((SRC/'4258点_5km海陆位置.csv').open(encoding='utf-8-sig')))
 cells=sorted(set((int(r['grid_row_5km']),int(r['grid_col_5km'])) for r in rows))
 mapped={};ties=0
 for j,i in cells:
  if land[j,i]==0:
   dest=(j,i);method='海格_保留所在格';n_ties=1
  else:
   d,k=tree.query([j,i]);ix=np.asarray(tree.query_ball_point([j,i],float(d)+1e-8),dtype=int)
   ds=(jj[ix]-j)**2+(ii[ix]-i)**2;ix=ix[ds==ds.min()]
   chosen=min(ix,key=lambda k:(jj[k],ii[k]));dest=(int(jj[chosen]),int(ii[chosen]))
   method='陆格_最近海格';n_ties=len(ix);ties+=int(n_ties>1)
  mapped[(j,i)]=(dest,method,n_ties)
 departures=sorted(set(v[0] for v in mapped.values()))
 ids={c:f'DEP5K_R{c[0]:04d}_C{c[1]:04d}' for c in departures}
 to_ll=Transformer.from_crs(crs,4326,always_xy=True)
 to_xy=Transformer.from_crs(4326,crs,always_xy=True)
 bydep=defaultdict(list)
 for r in rows:
  j,i=int(r['grid_row_5km']),int(r['grid_col_5km']);(dj,di),method,nt=mapped[(j,i)]
  dx,dy=rasterio.transform.xy(tf,dj,di);sx,sy=rasterio.transform.xy(tf,j,i)
  lon,lat=to_ll.transform(dx,dy);px,py=to_xy.transform(float(r['longitude']),float(r['latitude']))
  r.update(departure_id=ids[(dj,di)],departure_row=dj,departure_col=di,
   departure_center_x_m=float(dx),departure_center_y_m=float(dy),
   departure_longitude=float(lon),departure_latitude=float(lat),
   mapping_method=method,grid_center_mapping_distance_km=round(float(np.hypot(dx-sx,dy-sy)/1000),6),
   settlement_to_departure_center_projected_km=round(float(np.hypot(dx-px,dy-py)/1000),6),
   equidistant_nearest_sea_candidates=nt,tie_break='row_ascending_then_col_ascending',
   departure_in_AMAP_center_mask=int(amap[dj,di]),entrance_assignment='assigned_by_user_5km_rule',
   marine_grid_mapping='assigned_to_final_5km_geometric_departure',coordinate_changed=0)
  bydep[(dj,di)].append(r)
 dep=[];features=[]
 for (j,i),rr in sorted(bydep.items()):
  r=rr[0];z=dict(departure_id=r['departure_id'],departure_index=len(dep)+1,row=j,col=i,
   center_x_m=r['departure_center_x_m'],center_y_m=r['departure_center_y_m'],
   longitude=r['departure_longitude'],latitude=r['departure_latitude'],
   settlement_record_count=len(rr),port_settlement_count=sum(x['category']=='港口聚居点' for x in rr),
   coastal_settlement_count=sum(x['category']=='沿海聚居点' for x in rr),
   countries=';'.join(sorted(set(x['country_code'] for x in rr))),
   settlement_names=';'.join(x['name'] for x in rr),
   settlement_entity_ids=';'.join(x['entity_id'] for x in rr),
   max_grid_center_mapping_distance_km=max(x['grid_center_mapping_distance_km'] for x in rr))
  dep.append(z);features.append(dict(type='Feature',geometry=dict(type='Point',coordinates=[z['longitude'],z['latitude']]),properties=z))
  for x in rr:x['departure_shared_by_n_settlement_records']=len(rr)
 write('4258条聚居点_出发点对应表.csv',rows)
 write('船舶出发点_去重清单.csv',dep)
 write('出发点对应聚居点_长表.csv',[{k:r[k] for k in ['departure_id','departure_row','departure_col','entity_id','name','country_code','category','longitude','latitude','mapping_method','grid_center_mapping_distance_km']} for r in rows])
 (OUT/'船舶出发点_WGS84.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=features),ensure_ascii=False),encoding='utf-8')
 shutil.copy2(mask,OUT/'land_mask_5km.tif');shutil.copy2(boundary,OUT/'AMAP_mask_5km.tif')
 geography=np.where(amap==0,255,np.where(land==1,0,1)).astype('uint8')
 profile.update(nodata=255)
 with rasterio.open(OUT/'marine_geometry_mask_5km.tif','w',**profile) as d:
  d.write(geography,1);d.update_tags(values='0=land,1=geometric_marine,255=outside_AMAP',role='geometry_only_no_ice_or_depth')
 labels=np.zeros(land.shape,dtype='int32')
 for n,(j,i) in enumerate(departures,1):labels[j,i]=n
 profile.update(dtype='int32',nodata=0)
 with rasterio.open(OUT/'departure_index_5km.tif','w',**profile) as d:
  d.write(labels,1);d.update_tags(index_definition='1-based order of departure rows sorted by row,col; 0=no_departure')
 dist=np.array([r['grid_center_mapping_distance_km'] for r in rows])
 summary=dict(settlement_records=len(rows),settlement_occupied_cells=len(cells),
   departure_cells=len(dep),land_records=sum(r['mapping_method']=='陆格_最近海格' for r in rows),
   marine_records=sum(r['mapping_method']=='海格_保留所在格' for r in rows),
   one_record_departures=sum(len(rr)==1 for rr in bydep.values()),
   shared_departures=sum(len(rr)>1 for rr in bydep.values()),
   max_records_one_departure=max(len(rr) for rr in bydep.values()),
   tied_land_cells=ties,grid_spacing_m=5000,grid_shape=list(land.shape),
   projection_wkt=crs.to_wkt(),transform=list(tf),
   nearest_rule='minimum Euclidean distance between occupied land-cell center and marine-cell center; sea candidates within AMAP center mask',
   tie_break='smallest row then smallest column',
   original_sea_cell_rule='retain original occupied sea cell',
   land_rasterization='GSHHG 2.3.7 high L1 all_touched; same as 5km comparison',
   mapping_distance_km=dict(median=float(np.median(dist)),p95=float(np.percentile(dist,95)),max=float(dist.max())),
   all_4258_records_mapped=True,nodes_deleted=0,original_coordinates_changed=0,
   TOPAZ_processed=False,water_depth_used=False,RIO_computed=False,
   input_points_sha256=hashlib.sha256((SRC/'4258点_5km海陆位置.csv').read_bytes()).hexdigest(),
   land_mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest())
 assert len(rows)==4258 and len({r['entity_id'] for r in rows})==4258
 assert sum(z['settlement_record_count'] for z in dep)==4258
 assert all(land[j,i]==0 for j,i in departures)
 (OUT/'空间基准与映射统计.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

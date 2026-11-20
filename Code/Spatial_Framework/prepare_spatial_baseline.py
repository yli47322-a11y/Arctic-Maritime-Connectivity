"""Finish static land/sea geometry only, without ice, bathymetry or node snapping."""
from pathlib import Path
from collections import Counter
import csv,json,hashlib
import numpy as np
import rasterio
from pyproj import Transformer

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
OUT=ROOT/'高分辨率静态基准_2026-11-02'

def write(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 with rasterio.open(OUT/'GSHHG_land_alltouched_1km.tif') as d:
  land=d.read(1);profile=d.profile;tf=d.transform;crs=d.crs
 with rasterio.open(OUT/'AMAP_center_mask_1km.tif') as d:amap=d.read(1)
 # Distinguish outside study area (255), land (0), geometric marine (1).
 sea=np.where(amap==0,255,np.where(land==1,0,1)).astype('uint8')
 profile.update(nodata=255)
 with rasterio.open(OUT/'AMAP_geometric_marine_mask_1km.tif','w',**profile) as d:
  d.write(sea,1);d.update_tags(role='geometric_marine_not_ice_passability',
    values='0=land,1=geometric_marine,255=outside_AMAP',TOPAZ_processed='false',
    depth_condition='none',land_rasterization='all_touched')
 source=PROJECT/'人类聚居点/港口与沿海聚居点_规则结果/保留点位_港口与沿海.csv'
 with source.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 oldpath=PROJECT/'人类聚居点/TOPAZ网格排查_4258点_2026-11-02/4258点逐点网格排查.csv'
 with oldpath.open(encoding='utf-8-sig') as f:old={r['entity_id']:r for r in csv.DictReader(f)}
 tr=Transformer.from_crs(4326,crs,always_xy=True)
 coords=[tr.transform(float(r['longitude']),float(r['latitude'])) for r in rows]
 for r,(xx,yy) in zip(rows,coords):
  j,i=rasterio.transform.rowcol(tf,xx,yy);ins=0<=j<land.shape[0] and 0<=i<land.shape[1]
  n4=n8=0
  if ins:
   for dj in [-1,0,1]:
    for di in [-1,0,1]:
     if not (di or dj):continue
     # Pure geometric marine independent of AMAP; border does not become land.
     if 0<=j+dj<land.shape[0] and 0<=i+di<land.shape[1] and land[j+dj,i+di]==0:
      n8+=1
      if abs(di)+abs(dj)==1:n4+=1
  state='网格外' if not ins else '所在格为几何海格' if land[j,i]==0 else '所在格为陆格_四邻域有海格' if n4 else '所在格为陆格_仅对角有海格' if n8 else '所在格为陆格_八邻域无海格'
  o=old[r['entity_id']]
  r.update(geometry_1km_row=j,geometry_1km_col=i,geometry_1km_status=state,
   geometry_1km_land_alltouched=int(land[j,i]) if ins else '',
   geometry_1km_AMAP_center=int(amap[j,i]) if ins else '',
   geometry_1km_sea_neighbors_4=n4,geometry_1km_sea_neighbors_8=n8,
   previous_TOPAZ_native_status=o['native_cell_status'],
   previous_TOPAZ_native_wet=o['native_cell_wet'],
   previous_TOPAZ_nearest_wet_distance_km=o['nearest_wet_center_distance_km'],
   point_GSHHG_land=o['point_GSHHG_land'],
   final_marine_entrance='未指定',node_coordinates_changed=0)
 groups=Counter((r['geometry_1km_row'],r['geometry_1km_col']) for r in rows)
 for r in rows:r['same_geometry_1km_cell_point_count']=groups[(r['geometry_1km_row'],r['geometry_1km_col'])]
 counts=Counter(r['geometry_1km_status'] for r in rows)
 cross=Counter((r['previous_TOPAZ_native_wet'],r['geometry_1km_land_alltouched']) for r in rows)
 countries=[]
 for c in sorted(set(r['country_code'] for r in rows)):
  rr=[r for r in rows if r['country_code']==c]
  countries.append(dict(country_code=c,n_points=len(rr),**{s:sum(r['geometry_1km_status']==s for r in rr) for s in counts}))
 summary=dict(points=len(rows),working_spacing_m=1000,working_spacing_final=False,
   geometry_status_counts=dict(counts),
   previous_TOPAZ_wet322_on_highres_land=sum(r['previous_TOPAZ_native_wet']=='1' and r['geometry_1km_land_alltouched']==1 for r in rows),
   previous_TOPAZ_wet322_on_highres_marine=sum(r['previous_TOPAZ_native_wet']=='1' and r['geometry_1km_land_alltouched']==0 for r in rows),
   point_cells_with_AMAP_center_outside=sum(r['geometry_1km_AMAP_center']==0 for r in rows),
   distinct_point_cells=len(groups),multiple_point_cells=sum(n>1 for n in groups.values()),
   points_sharing_cells=sum(n for n in groups.values() if n>1),max_points_one_cell=max(groups.values()),
   AMAP_land_cells=int(((sea==0)).sum()),AMAP_geometric_marine_cells=int((sea==1).sum()),
   shape=list(land.shape),projection_wkt=crs.to_wkt(),transform=list(tf),
   TOPAZ_processed=False,water_depth_used=False,RIO_computed=False,nodes_moved=0,nodes_deleted=0,final_marine_assignments=0,
   land_method='GSHHG 2.3.7 high L1, all_touched=True; lakes remain land',
   source_points_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
   source_land_shp_sha256=hashlib.sha256((PROJECT/'人类聚居点/原始数据/GSHHG_2.3.7_high_L1/GSHHS_h_L1.shp').read_bytes()).hexdigest())
 assert len(rows)==4258 and sum(counts.values())==4258
 write('4258点_1km海陆几何与原生网格对照.csv',rows)
 write('按国家地区_1km几何位置.csv',countries)
 write('原生海格与高分辨率陆地对照.csv',[dict(TOPAZ_native_wet=k[0],highres_land_alltouched=k[1],n_points=n) for k,n in cross.items()])
 (OUT/'海陆基准统计.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2));print('COUNTRIES',json.dumps(countries,ensure_ascii=False))

if __name__=='__main__':main()

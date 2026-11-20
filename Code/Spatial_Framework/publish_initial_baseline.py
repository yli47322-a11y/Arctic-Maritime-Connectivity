"""Freeze the agreed final spatial baseline with explicit source/final coordinates."""
from pathlib import Path
from collections import Counter
import csv,json,hashlib,shutil,zipfile
import rasterio

PROJECT=Path(__file__).resolve().parent.parent
BASE=PROJECT/'空间基准/正式空间基准_5km'
INV=PROJECT/'人类聚居点/港口与沿海聚居点_规则结果'
OUT=PROJECT/'正式成果/空间基准_5km_最终正式版_2026-11-04'

def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def geo(p,rows,lon='longitude',lat='latitude'):
 features=[dict(type='Feature',geometry=dict(type='Point',coordinates=[float(r[lon]),float(r[lat])]),properties=r) for r in rows]
 p.write_text(json.dumps(dict(type='FeatureCollection',features=features),ensure_ascii=False),encoding='utf-8')

def main():
 if OUT.exists():raise RuntimeError('Formal snapshot already exists; do not overwrite it silently.')
 for name in ['01_最终点位与对应','02_原始清单与筛选溯源','03_正式5km栅格','04_规则与处理记录','05_展示图']:(OUT/name).mkdir(parents=True,exist_ok=True)
 original=read(BASE/'聚居点陆地定位/4258条聚居点_陆地定位与出发点.csv')
 departures=read(BASE/'船舶出发点_去重清单.csv')
 settlements=[];relations=[]
 for r in original:
  z=dict(entity_id=r['entity_id'],name=r['name'],country_code=r['country_code'],category=r['category'],
   longitude=float(r['settlement_representative_longitude']),latitude=float(r['settlement_representative_latitude']),
   land_grid_row=int(r['settlement_land_row']),land_grid_col=int(r['settlement_land_col']),
   original_longitude=float(r['longitude']),original_latitude=float(r['latitude']),
   original_grid_row=int(r['grid_row_5km']),original_grid_col=int(r['grid_col_5km']),
   representative_position_method=r['land_position_method'],
   position_shift_projected_km=float(r['land_position_shift_projected_km']),
   position_shift_geodesic_km=float(r['land_position_shift_geodesic_km']),
   original_cell_to_land_cell_distance_km=float(r['original_to_land_cell_center_distance_km']),
   source_record_ids=r['source_record_ids'],source_layer=r['source_layer'],source_url=r['source_url'],
   original_type=r['original_type'],coordinate_status=r['coordinate_status'],identity_status=r['identity_status'],
   original_coast_distance_km=float(r['coast_distance_km']),port_evidence=r['port_evidence'],
   historical_residence_1991_2025=r['historical_residence_1991_2025'],departure_id=r['departure_id'])
  settlements.append(z)
  relations.append(dict(entity_id=r['entity_id'],name=r['name'],country_code=r['country_code'],category=r['category'],
   settlement_longitude=z['longitude'],settlement_latitude=z['latitude'],
   settlement_land_row=z['land_grid_row'],settlement_land_col=z['land_grid_col'],
   departure_id=r['departure_id'],departure_row=int(r['departure_row']),departure_col=int(r['departure_col']),
   departure_longitude=float(r['departure_longitude']),departure_latitude=float(r['departure_latitude']),
   departure_center_x_m=float(r['departure_center_x_m']),departure_center_y_m=float(r['departure_center_y_m']),
   original_mapping_method=r['mapping_method'],original_grid_center_to_departure_km=float(r['grid_center_mapping_distance_km']),
   departure_retained_after_land_position_adjustment=True))
 # Refresh departure member names and IDs exclusively from the 4258-row join.
 byid={r['departure_id']:r for r in departures}
 counts=Counter(r['departure_id'] for r in relations)
 assert len(settlements)==4258 and len(set(r['entity_id'] for r in settlements))==4258
 assert len(byid)==1331 and set(counts)==set(byid)
 assert sum(counts.values())==4258
 with rasterio.open(BASE/'land_mask_5km.tif') as d:land=d.read(1)
 assert all(land[r['land_grid_row'],r['land_grid_col']]==1 for r in settlements)
 assert all(land[int(r['row']),int(r['col'])]==0 for r in departures)
 tab=OUT/'01_最终点位与对应'
 write(tab/'最终聚居点_4258条.csv',settlements);geo(tab/'最终聚居点_4258条.geojson',settlements)
 write(tab/'最终出发点_1331个.csv',departures);geo(tab/'最终出发点_1331个.geojson',departures)
 write(tab/'出发点与聚居点对应_4258条.csv',relations)
 write(tab/'16条陆地定位调整记录.csv',[r for r in settlements if r['position_shift_projected_km']>0])
 sources={
  '原始保留聚居点_4258条_调整前.csv':INV/'保留点位_港口与沿海.csv',
  '整合后全部候选_8639条_含去留.csv':INV/'全部点位_分类及去留.csv',
  '原始GeoNames候选_8019条.csv':PROJECT/'人类聚居点/聚居点候选母表_AMAP.csv',
  '各国来源清单_2098条.csv':PROJECT/'人类聚居点/分国点位/泛北极聚居及活动点_统一来源清单.csv',
  '原始10117条来源逐条去向.csv':INV/'原始10117条来源逐条去向.csv',
  '排除记录与原因_4381条.csv':INV/'排除点位及原因.csv',
 }
 for name,p in sources.items():shutil.copy2(p,OUT/'02_原始清单与筛选溯源'/name)
 for name in ['land_mask_5km.tif','AMAP_mask_5km.tif','marine_geometry_mask_5km.tif','departure_index_5km.tif']:
  shutil.copy2(BASE/name,OUT/'03_正式5km栅格'/name)
 records=OUT/'04_规则与处理记录'
 for p in [BASE/'空间基准与映射统计.json',BASE/'聚居点陆地定位/陆地定位统计.json',BASE/'聚居点陆地定位/00_说明.md',BASE/'00_使用说明.md']:
  shutil.copy2(p,records/p.name)
 shutil.copy2(PROJECT/'研究配置.json',records/'研究配置_快照.json')
 # The original preferred vector-basemap display is retained; deleted raster map excluded.
 for name in ['Arctic_departure_points_1331.jpg','Arctic_departure_points_1331.pdf','图注与说明.md']:
  shutil.copy2(BASE/'出发点分布图'/name,OUT/'05_展示图'/name)
 adjusted=[r for r in settlements if r['position_shift_projected_km']>0]
 summary=dict(version_date='2026-11-04',settlement_records=4258,departure_cells=1331,
   final_settlement_land_cells=len(set((r['land_grid_row'],r['land_grid_col']) for r in settlements)),
   adjusted_sea_to_land_records=len(adjusted),max_adjustment_8neighbor_rings=max(max(abs(r['land_grid_row']-r['original_grid_row']),abs(r['land_grid_col']-r['original_grid_col'])) for r in adjusted),
   max_adjustment_grid_center_distance_km=max(r['original_cell_to_land_cell_distance_km'] for r in adjusted),
   max_original_coordinate_shift_geodesic_km=max(r['position_shift_geodesic_km'] for r in adjusted),
   max_original_coordinate_shift_projected_km=max(r['position_shift_projected_km'] for r in adjusted),
   grid_spacing_m=5000,grid_shape=[2203,1523],TOPAZ_processed=False,water_depth_used=False,
   raw_lists={'GeoNames':8019,'country_sources':2098,'raw_source_rows':10117,'integrated_candidates':8639,'retained':4258,'excluded':4381},
   canonical_coordinates='longitude/latitude in final settlement table are current representative positions; original_longitude/original_latitude are source positions',
   index_raster='positive values join departure_index in final departure table; 0=no departure')
 (OUT/'版本摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 manifest=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file():manifest.append(dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 (OUT/'文件清单与SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2));print('OUTPUT',OUT)

if __name__=='__main__':main()

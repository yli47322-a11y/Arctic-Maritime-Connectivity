"""15 km geodesic catchments around documented port settlements; reuse daily labels."""
from pathlib import Path
import importlib.util
import csv
import json
import numpy as np
import rasterio
from scipy.ndimage import label
from pyproj import Transformer
from pyproj import Geod

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('partition', ROOT / '正式成果/统计港口支撑的联系机会增量.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.OUT = ROOT / '正式成果/港口15km陆地连通支撑的潜在联系机会增量'
base.OUT.mkdir(exist_ok=True)
base.GROUPS = ['两端有港口支撑', '仅一端有港口支撑', '两端均无15km陆地连通港口支撑']
rows = base.read(base.POINTS / '聚居点与出发格对应.csv')
ports = [r for r in rows if r['category'] == '港口聚居点']
plon = np.array([float(r['settlement_longitude']) for r in ports])
plat = np.array([float(r['settlement_latitude']) for r in ports])
geod = Geod(ellps='WGS84')
# Four-neighbour land components on the existing 1-km mask; no mask editing.
mask = ROOT / '空间基准/高分辨率静态基准_2026-11-02/GSHHG_land_alltouched_1km.tif'
with rasterio.open(mask) as ds:
    land = ds.read(1) == 1
    tf, crs = ds.transform, ds.crs
components, ncomponents = label(land)
transform = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
locations = {}
for r in rows:
    x,y = transform.transform(float(r['settlement_longitude']),float(r['settlement_latitude']))
    rr,cc = rasterio.transform.rowcol(tf,x,y)
    orig_r,orig_c = rr,cc
    if not land[rr,cc]:
        radius = 1
        while True:
            r0,r1=max(0,rr-radius),min(land.shape[0],rr+radius+1)
            c0,c1=max(0,cc-radius),min(land.shape[1],cc+radius+1)
            ar,ac=np.where(land[r0:r1,c0:c1])
            if len(ar):
                ar,ac=ar+r0,ac+c0
                xs,ys=rasterio.transform.xy(tf,ar,ac)
                dd=(np.asarray(xs)-x)**2+(np.asarray(ys)-y)**2
                j=int(np.argmin(dd))
                # Search square must encompass the complete closest-distance circle.
                if np.sqrt(dd[j]) <= max(0,radius-1)*1000:
                    rr,cc=int(ar[j]),int(ac[j]);break
            radius += 1
    lx,ly=rasterio.transform.xy(tf,rr,cc)
    locations[r['entity_id']]=(int(components[rr,cc]),int(orig_r),int(orig_c),int(rr),int(cc),
                               float(np.hypot(lx-x,ly-y)/1000),int(not land[orig_r,orig_c]))
port_components=np.array([locations[r['entity_id']][0] for r in ports])
supported = {}
detail = []
for r in rows:
    _, _, distances = geod.inv(np.full(len(ports), float(r['settlement_longitude'])),
                               np.full(len(ports), float(r['settlement_latitude'])), plon, plat)
    j = int(np.argmin(distances))
    distance = float(distances[j] / 1000)
    cid,orow,ocol,lr,lc,snap_distance,snapped=locations[r['entity_id']]
    candidates=np.flatnonzero((distances <= 15000+1e-6) & (port_components == cid))
    flag=bool(len(candidates))
    chosen=int(candidates[np.argmin(distances[candidates])]) if flag else None
    supported[r['entity_id']] = flag
    detail.append(dict(entity_id=r['entity_id'], name=r['name'], country_code=r['country_code'],
                       original_category=r['category'], longitude=r['settlement_longitude'],
                       latitude=r['settlement_latitude'], departure_id=r['departure_id'],
                       nearest_port_id=ports[j]['entity_id'], nearest_port_name=ports[j]['name'],
                       nearest_port_country=ports[j]['country_code'], distance_km=distance,
                       within_15km_any_port=int(distance<=15+1e-9),
                       land_component_1km=cid,land_row_1km=lr,land_col_1km=lc,
                       marine_original_cell_snapped_for_land_test=snapped,
                       distance_to_land_test_cell_centre_km=snap_distance,
                       supporting_port_id=ports[chosen]['entity_id'] if flag else '',
                       supporting_port_name=ports[chosen]['name'] if flag else '',
                       supporting_port_distance_km=float(distances[chosen]/1000) if flag else '',
                       land_connected_port_count_within15km=len(candidates),
                       port_supported_15km=int(flag), newly_supported=int(flag and r['category'] != '港口聚居点')))
base.write('4258聚居地_港口15km覆盖与最近港口.csv', detail)
original_read = base.read
def expanded_read(path):
    result = original_read(path)
    if path.name == '聚居点与出发格对应.csv':
        for row in result:
            row['category'] = '港口聚居点' if supported[row['entity_id']] else '沿海聚居点'
    return result
base.read = expanded_read
# The original routine checks a fixed port count. Use the identical calculation
# with that single count assertion generalized for the enlarged catchment.
source = (ROOT / '正式成果/统计港口支撑的联系机会增量.py').read_text(encoding='utf-8')
source = source.replace('assert (n, p, len(departures)) == (4258, 272, 1248)',
                        'assert n == 4258 and len(departures) == 1248')
namespace = base.__dict__
exec(compile(source[source.index('def main():'):source.index('if __name__')], str(__file__), 'exec'), namespace)
print(f'Original ports={len(ports)}; supported within 15 km={sum(supported.values())}', flush=True)
base.main()
meta_path = base.OUT / '计算记录.json'
meta = json.loads(meta_path.read_text(encoding='utf-8'))
meta.update(endpoint_rule='within_15km_geodesic_and_same_four_neighbour_land_component_1km',
            documented_port_records=len(ports), port_supported_settlement_records=sum(supported.values()),
            newly_supported_records=sum(r['newly_supported'] for r in detail),
            coordinate_basis='original settlement vector coordinates, not 5km departure grid centres',
            distance_method='WGS84 ellipsoidal geodesic, minimum over 272 documented port settlement points',
            radius_km=15, road_or_ferry_connectivity_required=False,
            land_mask=str(mask),land_connectivity=4,
            within15km_without_land_requirement=sum(r['within_15km_any_port'] for r in detail),
            removed_due_to_land_disconnection=sum(r['within_15km_any_port'] and not r['port_supported_15km'] for r in detail),
            original_marine_cell_records=sum(r['marine_original_cell_snapped_for_land_test'] for r in detail),
            land_mapping_rule='original land cell; original sea cell mapped to nearest land cell for this test only',
            land_path_length_capped=False)
meta['port_supported_departure_cells'] = meta.pop('documented_port_departure_cells')
meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

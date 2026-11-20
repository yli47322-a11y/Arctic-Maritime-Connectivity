from 整理加拿大与阿拉斯加 import RAW,OUT,Boundary,write,geo,norm
from 整理北欧官方点位 import ring_centroid
from pyproj import Transformer
from scipy.spatial import cKDTree
import csv,json,numpy as np
from collections import Counter

def read(name):
    with (OUT/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def sweden():
    tr=Transformer.from_crs(3006,4326,always_xy=True);rows=[]
    for layer in [0,1]:
        j=json.loads((RAW/f'sweden_mirror_layer{layer}_north.json').read_text(encoding='utf-8'))
        if j.get('exceededTransferLimit'):raise ValueError('Incomplete source page')
        for f in j['features']:
            a=f['attributes'];rings=f['geometry']['rings'];_,x,y=max(ring_centroid(r) for r in rings);lon,lat=tr.transform(x,y)
            code=a['tatortskod'] if layer==0 else a['smaort']
            rows.append(dict(entity_id='SE_SCB2023_'+code,name=a.get('tatort',''),source_settlement_code=code,country_code='SE',longitude=lon,latitude=lat,population_reported=a['bef'],population_reference_year=2023,source_entity_type='tatort' if layer==0 else 'smaort',municipality_code=a['kommun'],municipality_name=a['kommunnamn'],source_population_floor=200 if layer==0 else 50,source_status='SCB_data_republished_by_samhallsplanering_skane_not_direct_SCB_download',source_url='https://www.arcgis.com/home/item.html?id=23dbc7de31474b50a2cb47ab0cec53c8',source_objectid=a['OID'],position_method='largest_polygon_ring_centroid_EPSG3006',name_status='source_name' if layer==0 else 'source_has_code_only_not_named',marine_grid_mapping='deferred_by_user_2026-11-01',historical_residence_1991_2025='not_established'))
    Boundary().classify(rows);rows=[r for r in rows if r['inside_amap']==1]
    write('瑞典_AMAP内统计聚居区_2023_公开转载.csv',rows);geo('瑞典_AMAP内统计聚居区_2023_公开转载.geojson',rows)
    return dict(records=len(rows),types=dict(Counter(r['source_entity_type'] for r in rows)),missing_names=sum(not r['name'] for r in rows))

def russian_correspondence():
    rr=read('俄罗斯_AMAP内研究衍生名录_年份待核.csv');gn=read('俄罗斯_候选点_待补官方名录.csv')
    def xyz(rows):
        lat=np.radians([float(r['latitude']) for r in rows]);lon=np.radians([float(r['longitude']) for r in rows]);return np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
    tree=cKDTree(xyz(gn));out=[]
    for r,pos in zip(rr,xyz(rr)):
        aliases=lambda a:{norm(s) for s in [a['name'],a.get('ascii_name',''),*a.get('alternate_names','').split(',')] if s}
        candidates=tree.query_ball_point(pos,2*np.sin(10/6371/2));hits=[]
        for i in candidates:
            g=gn[i]
            if norm(r['name']) in aliases(g):
                dist=2*6371*np.arcsin(min(1,np.linalg.norm(pos-tree.data[i])/2));hits.append((g,dist))
        out.append(dict(research_entity_id=r['entity_id'],name=r['name'],longitude=r['longitude'],latitude=r['latitude'],oktmo=r['oktmo'],geonames_ids=';'.join(g['geonameid'] for g,d in hits),geonames_names=';'.join(g['name'] for g,d in hits),distances_km=';'.join(f'{d:.3f}' for g,d in hits),correspondence_status='unique_name_and_distance_candidate' if len(hits)==1 else 'ambiguous_multiple' if hits else 'no_name_correspondence_within10km',rule='normalized_exact_name_or_alias_and_distance_le10km; candidate correspondence only, no automatic merge'))
    write('俄罗斯_研究名录与GeoNames对应线索.csv',out)
    return dict(records=len(out),status=dict(Counter(r['correspondence_status'] for r in out)))

if __name__=='__main__':
    record=dict(Sweden=sweden(),Russia=russian_correspondence())
    (OUT/'瑞典补充及俄罗斯对应整理记录.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(record,ensure_ascii=False,indent=2))

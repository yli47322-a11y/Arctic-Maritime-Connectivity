"""Consolidate source inventories with explicit uncertainty; never select marine cells."""
from pathlib import Path
import csv,json,zipfile,io,shutil
from collections import Counter,defaultdict
from 整理加拿大与阿拉斯加 import Boundary,write,geo,OUT,RAW,ROOT
from 整理北欧官方点位 import ring_centroid

def read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def iceland():
    rows=read('冰岛_官方聚居实体及人口_2025.csv')
    original=RAW/'iceland_inventory_before_official_coordinate_supplement.csv'
    if not original.exists():shutil.copyfile(OUT/'冰岛_官方聚居实体及人口_2025.csv',original)
    ff=json.loads((RAW/'iceland_pending_official_names.geojson').read_text(encoding='utf-8'))['features']
    ids={'195':91144,'685':82407,'971':91655,'1166':91202}
    # Proxies are explicitly separate from statistical-nucleus polygon centres.
    gn_ids={'684':'2627822','687':'2625399','705':'2625204','990':'2630330'}
    gn={}
    with zipfile.ZipFile(RAW/'geonames_IS.zip') as z:
        for line in io.TextIOWrapper(z.open('IS.txt'),encoding='utf-8'):
            a=line.rstrip('\n').split('\t')
            if len(a)==19 and a[0] in gn_ids.values():gn[a[0]]=a
    buildings=json.loads((RAW/'iceland_pending_official_buildings.geojson').read_text(encoding='utf-8'))['features']
    bids={'688':4793,'1168':14836,'1169':3221}
    log=[]
    for r in rows:
        code=r['source_locality_code']
        if code in ids:
            f=next(f for f in ff if f['properties']['id']==ids[code]);g=f['geometry'];polys=g['coordinates'] if g['type']=='MultiPolygon' else [g['coordinates']]
            _,lon,lat=max(ring_centroid(poly[0]) for poly in polys)
            r.update(longitude=lon,latitude=lat,position_method='largest_official_named_nucleus_polygon_ring_centroid_WGS84',coordinate_source='https://gis.lmi.is/geoserver/IS_50V/ows',official_map_objectid=ids[code],coordinate_status='official_named_nucleus_polygon',coordinate_candidates='',coordinate_note='Official named nucleus boundary; not a marine entry point')
        elif code in gn_ids:
            a=gn[gn_ids[code]]
            r.update(longitude=float(a[5]),latitude=float(a[4]),position_method='same_named_farm_in_identified_municipality_proxy_not_statistical_centroid',coordinate_source='https://www.geonames.org/'+a[0],geonames_correspondence_ids=a[0],coordinate_status='approximate_named_locality_proxy',coordinate_candidates='',coordinate_note='Farm location used only as an approximate locality marker; statistical nucleus boundary not acquired')
        elif code in bids:
            f=next(f for f in buildings if f['properties']['objectid']==bids[code]);lon,lat=f['geometry']['coordinates']
            r.update(longitude=lon,latitude=lat,position_method='official_same_named_building_or_farm_proxy_in_identified_municipality',coordinate_source='https://gis.lmi.is/geoserver/IS_50V/ows',official_map_objectid=bids[code],coordinate_status='approximate_named_locality_proxy',coordinate_candidates='',coordinate_note='Official mapped feature position, not statistical nucleus centre; Solberg: Svalbardsstrond, Gljufurarholt/Vellir: Olfus')
        else:continue
        log.append(dict(entity_id=r['entity_id'],name=r['name'],longitude=r['longitude'],latitude=r['latitude'],coordinate_status=r['coordinate_status'],source=r['coordinate_source'],note=r['coordinate_note']))
    Boundary().classify(rows)
    write('冰岛_官方聚居实体及人口_2025.csv',rows);geo('冰岛_官方聚居实体_有位置.geojson',rows)
    write('冰岛_11条位置补充及精度说明.csv',log)
    write('冰岛_坐标待辨识.csv',[r for r in rows if not r['longitude']],fields=list(rows[0]))
    return rows

def main():
    iceland()
    specifications=[('GL','格陵兰_72个官方居住点.csv'),('CA','加拿大_AMAP内官方聚居地名实体.csv'),('US','阿拉斯加_AMAP内官方社区.csv'),('NO','挪威_AMAP内官方统计聚居区_2025.csv'),('FI','芬兰_AMAP内官方聚居区_2025.csv'),('IS','冰岛_官方聚居实体及人口_2025.csv'),('FO','法罗群岛_地理实体归并表_118点.csv'),('RU','俄罗斯_AMAP内研究衍生名录_年份待核.csv'),('SE','瑞典_AMAP内统计聚居区_2023_公开转载.csv'),('SJ','斯瓦尔巴及扬马延_活动类型与历史状态整理.csv')]
    allrows=[];index=[]
    for country,filename in specifications:
        rr=read(filename)
        if country=='GL':
            rr=[dict(r,entity_id=r['居民点编号'],name=r['名称'],longitude=r['经度'],latitude=r['纬度'],population_reported=r['人口2025'],population_reference_year=2025,source_url=r['坐标来源'],position_method=r['坐标方法'],source_entity_type='official_residential_unit') for r in rr]
        index.append(dict(country_code=country,source_units=len(rr),positioned=sum(bool(r.get('longitude')) for r in rr),file=filename,not_final_network_nodes=True))
        for i,r in enumerate(rr):
            eid=r.get('entity_id') or r.get('node_id') or country+'_SOURCE_'+str(i+1)
            name=r.get('name') or r.get('official_name') or r.get('locality_name') or r.get('source_settlement_code') or eid
            allrows.append(dict(inventory_record_id=country+'::'+eid,source_entity_id=eid,name=name,country_code=country,longitude=r.get('longitude',''),latitude=r.get('latitude',''),population_reported=r.get('population_reported',''),population_reference_year=r.get('population_reference_year',''),activity_type=r.get('activity_type') or r.get('source_entity_type') or r.get('feature_code') or 'source_community_or_residential_unit',coordinate_status=r.get('coordinate_status') or r.get('position_method') or 'source_position_not_final_marine_entry',source_file=filename,source_url=r.get('source_url',''),residence_history_1991_2025='not_established',marine_grid_mapping='deferred_by_user',record_scope='source_unit_not_confirmed_unique_current_resident_node',original_properties_json=json.dumps(r,ensure_ascii=False)))
    write('泛北极聚居及活动点_统一来源清单.csv',allrows);geo('泛北极聚居及活动点_统一来源清单.geojson',allrows)
    write('00_当前主清单索引.csv',index)
    # Keep unresolved correspondences separate; do not drop records to force a clean total.
    pending=[]
    for r in read('俄罗斯_研究名录与GeoNames对应线索.csv'):
        status=r.get('reciprocal_status',r['correspondence_status'])
        if status not in ['unique_name_and_distance_candidate','reciprocal_unique_candidate']:pending.append(dict(country='RU',id=r['research_entity_id'],name=r['name'],issue=status,meaning='source inventory preserved; identity correspondence unresolved, not a missing location'))
    for r in read('冰岛_11条位置补充及精度说明.csv'):
        if r['coordinate_status']=='approximate_named_locality_proxy':pending.append(dict(country='IS',id=r['entity_id'],name=r['name'],issue='approximate_position',meaning='locality marker available; statistical nucleus boundary remains unavailable'))
    for r in read('阿拉斯加_资料图索引补充_待辨名.csv'):pending.append(dict(country='US',id=r['entity_id'],name=r['name'],issue='profile_only_unit_identity_unresolved',meaning='retained in source supplement, not silently added as independent community'))
    for r in read('斯瓦尔巴及扬马延_活动类型与历史状态整理.csv'):
        if r['activity_type']=='待辨识':pending.append(dict(country='SJ',id=r['geonameid'],name=r['name'],issue='activity_or_entity_identity_unresolved',meaning='retained as candidate; not assigned resident status'))
    write('待补齐与待辨识事项.csv',pending)
    result=dict(source_inventory_rows=len(allrows),positioned_rows=sum(bool(r['longitude']) for r in allrows),country_inventory=index,iceland_coordinate_supplement=11,iceland_approximate_proxies=7,pending_items=len(pending),note='Sources consolidated; no claim of exhaustive currently inhabited settlements or a final deduplicated network-node total. GeoNames 8019 candidate rows preserved separately.')
    (OUT/'清单整合交付记录.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

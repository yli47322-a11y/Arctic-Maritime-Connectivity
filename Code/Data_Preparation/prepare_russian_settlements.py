"""Retain public derivative provenance and original population strings; no marine mapping."""
from 整理加拿大与阿拉斯加 import Boundary,write,geo,RAW,OUT
import csv,json,hashlib,math
from collections import Counter

p=RAW/'俄罗斯公开衍生名录'/'settlements.csv'
rows=[];invalid=[];total=0
with p.open(encoding='utf-8-sig',newline='') as f:
    for a in csv.DictReader(f,delimiter=';'):
        total+=1
        r=dict(a,entity_id='RU_RCSI_DERIVED_'+a['id'],name=a['settlement'],country_code='RU',population_reported=a['population'],population_reference_year='mixed_or_not_verified_not_2025',source_url='https://github.com/NickyX3/russia_settlements',original_dataset_url='https://data.rcsi.science/data-catalog/datasets/160/',source_status='public_third_party_derivative_not_direct_official_download',license='CC BY-NC-SA; see source README',residence_status='not_confirmed_current_or_continuous_1991_2025',marine_grid_mapping='deferred_by_user_2026-11-01')
        try:
            lon=float(a['longitude_dd']);lat=float(a['latitude_dd'])
            if not(math.isfinite(lon) and math.isfinite(lat) and -180<=lon<=180 and -90<=lat<=90):raise ValueError()
            r.update(longitude=lon,latitude=lat)
        except (ValueError,TypeError):invalid.append(r);continue
        rows.append(r)
Boundary().classify(rows)
inside=[r for r in rows if r['inside_amap']==1]
write('俄罗斯_AMAP内研究衍生名录_年份待核.csv',inside)
geo('俄罗斯_AMAP内研究衍生名录_年份待核.geojson',inside)
if invalid:write('俄罗斯_研究衍生名录_坐标无效或缺失.csv',invalid)
record=dict(source_rows=total,inside_amap=len(inside),invalid_coordinates=len(invalid),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),regions=dict(Counter(r['region'] for r in inside)),positive_population=sum(float(r['population'] or 0)>0 for r in inside),note='人口0不判为废弃；不按人口删点；本表不替代官方核定名录，也不与GeoNames直接累加。')
(OUT/'俄罗斯研究衍生名录整理记录.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))

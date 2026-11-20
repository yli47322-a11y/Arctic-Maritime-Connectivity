from pathlib import Path
import csv,json,zipfile
from collections import Counter
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'港口与沿海聚居点_规则结果'
def read(n):
    with (OUT/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
rows=read('全部点位_分类及去留.csv')
for r in rows:
    if r['country_code']=='RU' and r['original_type'] in ['остров','р-н']:
        category='非聚居实体';action='排除';why='来源为岛屿或地区，不作为聚居点'
    elif float(r['coast_distance_km'])>10:
        category='内陆港口_超10km' if r['port_status']=='documented_port_or_quay' else '非沿海点'
        action='排除';why='距海岸超过10km；港口与其他点适用同一距岸限制'
    else:
        category='港口聚居点' if r['port_status']=='documented_port_or_quay' else '沿海聚居点'
        action='保留';why='距海岸不超过10km；'+('有港口/码头资料支撑' if category=='港口聚居点' else '未取得港口资料支撑，按沿海点保留')
    r.update(category=category,disposition=action,reason=why,selection_level='全部聚居点统一距岸≤10km_港口仅作类内区分')
kept=[r for r in rows if r['disposition']=='保留'];excluded=[r for r in rows if r['disposition']=='排除']
for n,rr in [('全部点位_分类及去留.csv',rows),('保留点位_港口与沿海.csv',kept),('港口聚居点.csv',[r for r in kept if r['category']=='港口聚居点']),('沿海聚居点.csv',[r for r in kept if r['category']=='沿海聚居点']),('排除点位及原因.csv',excluded),('排除的内陆港口.csv',[r for r in excluded if r['category']=='内陆港口_超10km'])]:write(n,rr)
byid={r['entity_id']:r for r in rows};audit=read('原始10117条来源逐条去向.csv')
for r in audit:
    a=byid[r['entity_id']];r.update(category=a['category'],disposition=a['disposition'],reason=a['reason'])
write('原始10117条来源逐条去向.csv',audit)
index=[]
for c in sorted(set(r['country_code'] for r in rows)):
    rr=[r for r in rows if r['country_code']==c];counts=Counter(r['category'] for r in rr);index.append(dict(country_code=c,港口聚居点=counts['港口聚居点'],沿海聚居点=counts['沿海聚居点'],保留合计=sum(r['disposition']=='保留' for r in rr),排除合计=sum(r['disposition']=='排除' for r in rr)))
write('按国家地区汇总.csv',index)
(OUT/'保留点位.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=[dict(type='Feature',geometry=dict(type='Point',coordinates=[float(r['longitude']),float(r['latitude'])]),properties=r) for r in kept]),ensure_ascii=False),encoding='utf-8')
stats=dict(input_source_records=10117,linked_source_pairs=1478,classified_inventory_units=len(rows),categories=dict(Counter(r['category'] for r in rows)),retained=len(kept),excluded=len(excluded),pending=0,coastal_threshold_km=10,port_exempt_from_threshold=False,all_original_sources_accounted_for=len(audit)==10117)
(OUT/'规则分类统计.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'00_结果说明.md').write_text('''# 统一10km距岸限制的当前结果

按用户最新要求，港口也适用距海岸≤10km限制。已明确的非聚居实体排除；其余超过10km一律排除；不超过10km者，有港口资料归港口聚居点，其余归沿海聚居点。当前保留4258条（264条港口、3994条沿海）；排除4381条，其中48条是本轮新增排除的超10km内陆港口。原始港口证据未删除。

8019条GeoNames来源及2098条国家来源经1478组唯一名称/位置对应形成8639条分类单元。全部来源的逐条去向可查表，原身份、可能重复、位置近似与历史状态备注保留，不阻断用户要求的规则分类。点数是清单单元数，不宣称全部人工核定的独立现存居民点。

距离沿用GSHHG 2.3.7 high L1陆海边界及已有代表点计算，不是到湖岸、海格或港口的距离；没有开展海格映射。旧港口不受10km限制的说明由本版覆盖。此前90条行政/汇总/区外记录位于本轮输入之外，另存原表。
''',encoding='utf-8')
p=ROOT.parent/'研究配置.json';j=json.loads(p.read_text(encoding='utf-8'));rule=j['spatial_scope']['settlement_inclusion_rule'];rule.update(include=['port_settlements_within10km_of_coast','other_settlements_within10km_of_coast'],classification_priority=['exclude_known_nonsettlement','exclude_all_distance_gt10km_including_ports','within10km_classify_port_supported_or_coastal'],rule_classified_counts=stats,port_exempt_from_coastal_threshold=False);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=ROOT.parent/'项目备忘.md';text=p.read_text(encoding='utf-8');note='\n## 港口同样限制距岸10km（用户最新要求，2026-11-01）\n\n港口与其他聚居点一律应用距岸≤10km。超10km的内陆港口排除。地图加入全部排除点并弱化为背景点。此项覆盖此前“所有有港口支撑点不受距离限制”的规则。最新保留4258条（264港口、3994沿海），新增排除48条超距港口。\n'
if note not in text:p.write_text(text+note,encoding='utf-8')
with zipfile.ZipFile(ROOT/'港口与沿海聚居点_规则结果_2026-11-01.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in OUT.iterdir():
        if f.is_file():z.write(f,f.name)
print(json.dumps(stats,ensure_ascii=False))

"""Package country inventories and explicit outstanding data issues, without grid mapping."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'分国点位'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def main():
    # Preserve source statistical subunits while making one row per official village point.
    fo=read(OUT/'法罗群岛_官方聚居实体及人口_2025.csv');groups=defaultdict(list)
    for r in fo:groups[r['official_map_objectid'] or r['entity_id']].append(r)
    names={str(f['properties']['OBJECTID']):f['properties']['stad'] for f in json.loads((ROOT/'原始数据/faroe_official_villages.geojson').read_text(encoding='utf-8'))['features']}
    merged=[]
    for sid,rr in groups.items():
        r=rr[0].copy();r['entity_id']='FO_MAP_'+sid;r['name']=names.get(sid,r['name']);r['statistical_unit_ids']=';'.join(a['source_locality_code'] for a in rr);r['statistical_unit_names']=';'.join(a['name'] for a in rr);r['statistical_unit_count']=len(rr)
        vals=[a['population_reported'] for a in rr]
        if all(v not in ['',None] for v in vals):r['population_reported']=sum(int(float(v)) for v in vals)
        else:r['population_reported']='';r['population_status']='missing_in_at_least_one_statistical_component'
        r['geographic_entity_rule']='same_official_mapping_OBJECTID';merged.append(r)
    write(OUT/'法罗群岛_地理实体归并表_118点.csv',merged)
    # Remaining countries are transparent supplements, not renamed official inventories.
    gn=read(ROOT/'聚居点候选母表_AMAP.csv');fallback={'RU':'俄罗斯','SE':'瑞典','SJ':'斯瓦尔巴及扬马延'}
    for c,zh in fallback.items():
        rows=[dict(r,inventory_level='GeoNames候选_尚未接入完整官方居住名录',marine_grid_mapping='按用户要求暂缓') for r in gn if r['country_code']==c]
        write(OUT/(zh+'_候选点_待补官方名录.csv'),rows)
    # Keep tiny settlements in all countries, including those outside statistical population thresholds.
    write(OUT/'各国小型及未核定聚居点补充候选_原始8019条.csv',gn)
    pending=[]
    for r in read(OUT/'冰岛_坐标待辨识.csv'):pending.append(dict(country='冰岛',record_id=r['entity_id'],name=r['name'],issue='尚缺唯一坐标对应',source=r['source_url']))
    for r in read(OUT/'阿拉斯加_资料图索引补充_待辨名.csv'):pending.append(dict(country='阿拉斯加',record_id=r['entity_id'],name=r['name'],issue='资料图索引独有点，需辨认是否异名或新增社区',source=r['source_url']))
    pending.extend([dict(country='瑞典',record_id='',name='',issue='SCB官方统计聚居地和小聚居地服务已定位；下载连接失败，未冒充已下载',source='https://www.scb.se/vara-tjanster/oppna-data/oppna-geodata/statistiska-tatorter/'),dict(country='俄罗斯',record_id='',name='',issue='155921个居民点的研究数据集需登录/注册，未下载；联邦普查公开汇总有农村3000人门槛，不能当完整小村名录',source='https://data.rcsi.science/data-catalog/datasets/160/'),dict(country='斯瓦尔巴及扬马延',record_id='',name='',issue='现有11条地名候选的居住、科研、废弃状态需要分别整理',source='https://download.geonames.org/export/dump/')])
    write(OUT/'待补齐与待辨识事项.csv',pending)
    ca=json.loads((OUT/'分国整理记录.json').read_text(encoding='utf-8'));no=json.loads((OUT/'北欧分国整理记录.json').read_text(encoding='utf-8'))
    index=[dict(国家地区='格陵兰',清单记录数=72,有坐标数=72,资料性质='官方2025年有人口记录的居住实体',文件='格陵兰_72个官方居住点.csv',主要限制='未建立逐年居住史；科考站另列'),dict(国家地区='加拿大',清单记录数=ca['Canada']['inside_amap_settlement_features'],有坐标数=ca['Canada']['inside_amap_settlement_features'],资料性质='联邦官方聚居地名实体',文件='加拿大_AMAP内官方聚居地名实体.csv',主要限制='官方地名不等于已确认现存常住社区'),dict(国家地区='阿拉斯加',清单记录数=ca['Alaska']['inside_amap_community_records'],有坐标数=ca['Alaska']['inside_amap_community_records'],资料性质='州政府社区资料',文件='阿拉斯加_AMAP内官方社区.csv',主要限制='交通属性多为2018年；人口字段年份未给出'),dict(国家地区='挪威',清单记录数=no['Norway']['official_units_inside_AMAP'],有坐标数=no['Norway']['official_units_inside_AMAP'],资料性质='官方2025年统计聚居区',文件='挪威_AMAP内官方统计聚居区_2025.csv',主要限制='来源门槛200人，更小村落保留在补充候选'),dict(国家地区='芬兰',清单记录数=no['Finland']['official_units_inside_AMAP'],有坐标数=no['Finland']['official_units_inside_AMAP'],资料性质='官方2025年聚居区',文件='芬兰_AMAP内官方聚居区_2025.csv',主要限制='保留人口类别码，不能当人口数；小村仍需补齐'),dict(国家地区='冰岛',清单记录数=no['Iceland']['official_named_units'],有坐标数=no['Iceland']['positioned'],资料性质='官方2025年人口表中的具名居住单元',文件='冰岛_官方聚居实体及人口_2025.csv',主要限制=f"{no['Iceland']['coordinate_unresolved']}条坐标待辨识；包含2025年人口为0的记录"),dict(国家地区='法罗群岛',清单记录数=len(merged),有坐标数=len(merged),资料性质='官方人口表与官方村落点对应后的地理实体',文件='法罗群岛_地理实体归并表_118点.csv',主要限制='原表119个统计分区归并为118个地图实体；5条原人口记录为空')]
    for c,zh in fallback.items():
        n=sum(r['country_code']==c for r in gn);index.append(dict(国家地区=zh,清单记录数=n,有坐标数=n,资料性质='补充候选，未完成官方名录整理',文件=zh+'_候选点_待补官方名录.csv',主要限制='不能计作已经核定的独立常住居民点数量'))
    write(OUT/'00_分国清单索引.csv',index)
    (OUT/'分国清单汇总记录.json').write_text(json.dumps({'scope':'AMAP','marine_grid_mapping':'deferred_by_user_2026-11-01','country_inventory_index':index,'faroe_statistical_rows':len(fo),'faroe_cartographic_entities':len(merged),'faroe_components_merged':[{'name':r['name'],'statistical_units':r['statistical_unit_names'],'population_2025':r['population_reported']} for r in merged if r['statistical_unit_count']>1],'pending_issues':len(pending),'final_panarctic_resident_network_nodes_selected':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(index,ensure_ascii=False,indent=2));print('pending issues',len(pending))
if __name__=='__main__':main()

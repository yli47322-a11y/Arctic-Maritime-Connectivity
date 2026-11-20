"""Prepare sourced community inventories; marine-grid mapping is intentionally deferred."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,csv,io,zipfile,hashlib,re
import numpy as np
import shapefile
from pyproj import Transformer,CRS
from matplotlib.path import Path as PolygonPath

ROOT=Path(__file__).resolve().parent;RAW=ROOT/'原始数据';OUT=ROOT/'分国点位';OUT.mkdir(exist_ok=True)
def write(name,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def geo(name,rows):
    (OUT/name).write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','geometry':{'type':'Point','coordinates':[float(r['longitude']),float(r['latitude'])]},'properties':r} for r in rows if r.get('longitude') not in [None,'']]},ensure_ascii=False),encoding='utf-8')
def norm(s):
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFKD',s.casefold()) if c.isalnum() and not unicodedata.combining(c))
class Boundary:
    def __init__(self):
        with zipfile.ZipFile(RAW/'AMAP_area.zip') as z:
            b='amaplim_lam_poly';rd=shapefile.Reader(shp=io.BytesIO(z.read(b+'.shp')),shx=io.BytesIO(z.read(b+'.shx')),dbf=io.BytesIO(z.read(b+'.dbf')))
            self.shapes=list(rd.iterShapes());self.tr=Transformer.from_crs(4326,CRS.from_wkt(z.read(b+'.prj').decode()),always_xy=True)
    def classify(self,rows):
        valid=[r for r in rows if r.get('longitude') not in [None,''] and r.get('latitude') not in [None,'']]
        if not valid:return
        xx,yy=self.tr.transform([float(r['longitude']) for r in valid],[float(r['latitude']) for r in valid]);p=np.column_stack([xx,yy]);inside=np.zeros(len(valid),dtype=bool)
        for s in self.shapes:
            cuts=list(s.parts)+[len(s.points)];a=np.zeros(len(valid),dtype=bool)
            for b,e in zip(cuts[:-1],cuts[1:]):a ^= PolygonPath(np.asarray(s.points[b:e])).contains_points(p,radius=.01)
            inside |= a
        for r,k in zip(valid,inside):r['inside_amap']=int(k)

def alaska(boundary):
    features=json.loads((RAW/'alaska_transport.geojson').read_text(encoding='utf-8'))['features'];profiles=json.loads((RAW/'alaska_communities.geojson').read_text(encoding='utf-8'))['features']
    index=defaultdict(list)
    for f in profiles:index[norm(f['properties']['Place'])].append(f)
    groups=defaultdict(list)
    for f in features:groups[f['properties']['CommunityID_1'] or 'MISSING_'+str(f['properties']['OBJECTID'])].append(f)
    records=[];admin=[];missing=[]
    for sid,ff in groups.items():
        ff.sort(key=lambda f:(f['properties'].get('AsOfDate') or 0,f['properties'].get('last_edited_date_1') or 0,f['properties']['OBJECTID']),reverse=True)
        f=ff[0];p=f['properties'];name=p['CommunityName'];g=f.get('geometry');lon,lat=g['coordinates'][:2] if g else ('','')
        n=norm(name);profile=index.get(n,[]);profile=profile[0]['properties'] if len(profile)==1 else {}
        desc=p.get('Description') or '';flags=[s for s in ['uninhabited','unpopulated','abandoned','seasonal','relocat'] if s in desc.casefold()]
        stamp=p.get('AsOfDate');year=datetime.fromtimestamp(stamp/1000,timezone.utc).date().isoformat() if stamp else ''
        r=dict(entity_id='AK_DCRA_'+sid,name=name,country_code='US',region='Alaska',longitude=lon,latitude=lat,position_source='Alaska_DCRA_Community_Transportation_Overview',source_url='https://services2.arcgis.com/0DjevcWawQ1dy3il/ArcGIS/rest/services/Community_Transportation_Overview/FeatureServer/0',source_record_ids=';'.join(str(x['properties']['OBJECTID']) for x in ff),source_rows_for_same_community_id=len(ff),transport_attributes_asof=year,source_coastal=p.get('Coastal'),source_harbor_dock=p.get('HarborDock'),source_cargo_barge=p.get('CargoBarge'),source_road_connection=p.get('RoadConnection'),community_class=profile.get('Class',''),population_reported=profile.get('Population',''),population_reference_year='not_given_in_profile_field',residence_status='not_established_for_study_period',source_description_flags=';'.join(flags),source_description=desc,inside_amap='not_positioned' if not g else '',entity_status='official_community_source_record_requires_final_scope',marine_grid_mapping='deferred_by_user_2026-11-01')
        # County/borough seats and borough polygons are not interchangeable with settlement entities.
        if name=='Alaska' or name.endswith(' Borough') or name.endswith(' Census Area'):
            r['entity_status']='administrative_or_statewide_record_not_an_independent_settlement';admin.append(r)
        elif not g:missing.append(r)
        else:records.append(r)
    boundary.classify(records);boundary.classify(admin)
    inside=[r for r in records if r['inside_amap']==1];outside=[r for r in records if r['inside_amap']==0]
    # Profile-only places are a supplement, not auto-merged by distance.
    names={norm(r['name']) for r in records};extra=[]
    for f in profiles:
        p=f['properties'];name=p['Place'];g=f.get('geometry')
        if norm(name) in names or not g:continue
        if p['Class'] in {'Non-Unified Home Rule Borough','First Class Borough'}:continue
        lon,lat=g['coordinates'][:2];extra.append(dict(entity_id='AK_DCRA_PROFILE_'+str(p['OBJECTID']),name=name,country_code='US',region='Alaska',longitude=lon,latitude=lat,community_class=p['Class'],population_reported=p['Population'],population_reference_year='not_given_in_profile_field',position_source='DCRA_Community_Profiles_Map_Index',source_url='https://services2.arcgis.com/0DjevcWawQ1dy3il/ArcGIS/rest/services/Community_Profiles/FeatureServer/0',entity_status='profile_only_possible_alias_or_additional_community',marine_grid_mapping='deferred_by_user_2026-11-01'))
    boundary.classify(extra);extra=[r for r in extra if r['inside_amap']==1]
    inside.sort(key=lambda r:r['name']);write('阿拉斯加_AMAP内官方社区.csv',inside);geo('阿拉斯加_AMAP内官方社区.geojson',inside)
    write('阿拉斯加_行政区域记录_不作居民点.csv',admin);write('阿拉斯加_AMAP外社区_保留来源.csv',outside);write('阿拉斯加_资料图索引补充_待辨名.csv',extra,['entity_id','name','country_code','region','longitude','latitude','community_class','population_reported','population_reference_year','position_source','source_url','entity_status','marine_grid_mapping','inside_amap'])
    if missing:write('阿拉斯加_缺坐标记录.csv',missing)
    return dict(raw_transport_rows=len(features),unique_community_ids=len(groups),inside_amap_community_records=len(inside),administrative_records_separated=len(admin),outside_amap_community_records=len(outside),unpositioned_nonadmin_records=len(missing),profile_only_supplement=len(extra),coastal_source_counts=dict(Counter(r['source_coastal'] or 'not_reported' for r in inside)),transport_asof_dates=dict(Counter(r['transport_attributes_asof'] for r in inside)),source_claim='official community list and transportation attributes, not proof that all records were continuously inhabited1991-2025')

def canada(boundary):
    raw=json.loads((RAW/'canada_official_settlement_records.json').read_text(encoding='utf-8'));group=defaultdict(list)
    for r in raw:group[(r.get('feature',{}).get('id') or r['id'],r['concise']['code']=='IR')].append(r)
    rows=[]
    for (fid,isreserve),rr in group.items():
        # Official aliases of the same CGNDB feature are one geographic entity, not separate nodes.
        rr.sort(key=lambda r:r['name']);r=rr[0];names=sorted({a['name'] for a in rr});types=sorted({a['concise']['code'] for a in rr});generic=sorted({str(a['generic']['code']) for a in rr})
        rows.append(dict(entity_id='CA_CGNDB_'+fid,name=r['name'],official_names=';'.join(names),source_name_ids=';'.join(a['id'] for a in rr),country_code='CA',province_code=r['province']['code'],longitude=r['longitude'],latitude=r['latitude'],source_feature_types=';'.join(types),generic_codes=';'.join(generic),source_official_status='O',name_decision_dates=';'.join(sorted({a.get('decision','') or '' for a in rr})),source_url=r['links']['feature']['href'],position_source='Natural_Resources_Canada_CGNDB',population_reported='',population_reference_year='',residence_status='official_name_status_does_not_establish_current_or_historical_population',entity_status='reserve_administrative_area_not_automatically_a_settlement' if isreserve else 'official_named_settlement_feature_requires_residence_resolution',inside_amap='',marine_grid_mapping='deferred_by_user_2026-11-01',position_variants_in_same_feature=len({(a['latitude'],a['longitude']) for a in rr})))
    boundary.classify(rows);inside=[r for r in rows if r['inside_amap']==1];settlements=[r for r in inside if r['entity_status']!='reserve_administrative_area_not_automatically_a_settlement'];reserves=[r for r in inside if r['entity_status']=='reserve_administrative_area_not_automatically_a_settlement']
    settlements.sort(key=lambda r:(r['province_code'],r['name']));write('加拿大_AMAP内官方聚居地名实体.csv',settlements);geo('加拿大_AMAP内官方聚居地名实体.geojson',settlements);write('加拿大_AMAP内保留地行政记录_另列.csv',reserves)
    return dict(official_name_records_downloaded=len(raw),unique_source_features=len(group),inside_amap_settlement_features=len(settlements),inside_amap_reserve_administrative_features=len(reserves),feature_types=dict(Counter(r['source_feature_types'] for r in settlements)),position_variants_needing_review=sum(r['position_variants_in_same_feature']>1 for r in settlements),source_claim='CGNDB official geographic-name features, not a fully confirmed resident-community inventory')

def greenland():
    p=ROOT/'实体整理/格陵兰官方居住实体_带位置与码头线索.csv';rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
    output=[]
    for r in rows:
        output.append(dict(居民点编号=r['entity_id'],名称=r['name'].split('(')[0].strip(),国家地区='格陵兰',经度=r['longitude'],纬度=r['latitude'],人口1991=r['population_1991'],人口2025=r['population_2025'],官方居住地编码=r['official_locality_code'],坐标来源=r.get('position_source_url',''),坐标方法=r['position_method'],居住依据='官方2025年人口记录',历史连续居住状态='尚未建立逐年记录',码头岸线记录数=r['official_quay_edge_count'],海格匹配='按用户要求暂缓'))
    write('格陵兰_72个官方居住点.csv',output);return {'official_residential_entities':len(rows),'positioned':sum(r['longitude']!='' for r in rows),'population_reference_year':2025}

def main():
    boundary=Boundary();stats={'Greenland':greenland(),'Canada':canada(boundary),'Alaska':alaska(boundary),'marine_grid_mapping':'deferred_by_user_2026-11-01','boundary':'AMAP_official_polygon','historical_node_panel_constructed':False,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW/'alaska_transport.geojson',RAW/'alaska_communities.geojson',RAW/'canada_official_settlement_records.json']}}
    (OUT/'分国整理记录.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

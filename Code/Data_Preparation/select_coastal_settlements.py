"""Account for every initial candidate and sourced unit under the user inclusion rule."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,math,re,hashlib
import numpy as np
import shapefile
from scipy.spatial import cKDTree
from 整理实体与海上入口线索 import norm

ROOT=Path(__file__).resolve().parent;RAW=ROOT/'原始数据';BASE=ROOT/'分国点位';OUT=ROOT/'港口与沿海筛选';OUT.mkdir(exist_ok=True)
R=6371.0088
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def xyz(ll):
    a=np.radians(np.asarray(ll,float));lon=a[:,0];lat=a[:,1]
    return np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
def km(a,b):return float(2*R*np.arcsin(min(1,np.linalg.norm(a-b)/2)))
def aliases(r):
    a=[r.get(k,'') for k in ['name','ascii_name','local_language_name','official_names','name_da','name_gl']]
    a+=re.split('[;,]',r.get('alternate_names',''))
    a+=r.get('official_names','').split(';')
    a+=[re.sub(r'\s*\([^)]*\)','',r.get('name',''))]
    return {norm(s) for s in a if s}

def coast_distance(rows):
    cache=OUT/'GSHHG距离缓存.json'
    ids=[r['entity_id'] for r in rows]
    sig=hashlib.sha256(('exclude_dateline_polygon_closure_v1'+json.dumps([(r['entity_id'],r['longitude'],r['latitude']) for r in rows])).encode()).hexdigest()
    if cache.exists():
        j=json.loads(cache.read_text(encoding='utf-8'))
        if j['input_sha256']==sig:return j['distances_km']
    p=RAW/'GSHHG_2.3.7_high_L1';rd=shapefile.Reader(shp=str(p/'GSHHS_h_L1.shp'),shx=str(p/'GSHHS_h_L1.shx'))
    aa=[];bb=[]
    for s in rd.iterShapes():
        if s.bbox[3]<45:continue
        cuts=list(s.parts)+[len(s.points)]
        for start,end in zip(cuts[:-1],cuts[1:]):
            v=xyz(s.points[start:end]);lat=np.asarray(s.points[start:end])[:,1]
            for i in np.where(np.maximum(lat[:-1],lat[1:])>=45)[0]:
                # GSHHG splits polygons at the dateline; the artificial closure is not a coast.
                if abs(abs(s.points[start+i][0])-180)<1e-8 and abs(abs(s.points[start+i+1][0])-180)<1e-8:continue
                a=v[i];b=v[i+1];count=max(1,int(np.ceil(np.linalg.norm(a-b)*R)))
                t=np.linspace(0,1,count+1);q=a[None,:]*(1-t[:,None])+b[None,:]*t[:,None];q/=np.linalg.norm(q,axis=1)[:,None]
                aa.append(q[:-1]);bb.append(q[1:])
    a=np.vstack(aa);b=np.vstack(bb);mid=(a+b)/2;tree=cKDTree(mid);v=b-a;vv=(v*v).sum(axis=1)
    points=xyz([(r['longitude'],r['latitude']) for r in rows]);near,_=tree.query(points);dist=[]
    # Every subdivided segment is <=1km; this radius includes the true nearest segment.
    for pt,d in zip(points,near):
        ix=tree.query_ball_point(pt,d+0.501/R);delta=pt-a[ix];vec=v[ix];den=vv[ix]
        t=np.divide((delta*vec).sum(axis=1),den,out=np.zeros(len(ix)),where=den>0);t=np.clip(t,0,1)
        chord=np.linalg.norm(delta-t[:,None]*vec,axis=1).min();dist.append(float(2*R*np.arcsin(min(1,chord/2))))
    cache.write_text(json.dumps(dict(input_sha256=sig,distances_km=dist)),encoding='utf-8')
    print('Coast distance complete',len(dist),flush=True);return dist

def main():
    gn=read(ROOT/'聚居点候选母表_AMAP.csv');official=read(BASE/'泛北极聚居及活动点_统一来源清单.csv')
    gi=defaultdict(set)
    for i,r in enumerate(gn):
        for n in aliases(r):gi[(r['country_code'],n)].add(i)
    gp=xyz([(r['longitude'],r['latitude']) for r in gn]);op=xyz([(r['longitude'],r['latitude']) for r in official]);links=[];rev=defaultdict(list)
    for i,r in enumerate(official):
        raw=json.loads(r['original_properties_json']);names=aliases(dict(raw,name=r['name']));cand=set()
        for n in names:cand|=gi.get((r['country_code'],n),set())
        close=[k for k in sorted(cand) if km(op[i],gp[k])<=10];links.append(close)
        if len(close)==1:rev[close[0]].append(i)
    rows=[];cross=[];merged=set()
    for i,r in enumerate(official):
        raw=json.loads(r['original_properties_json']);g=links[i];match=g[0] if len(g)==1 and len(rev[g[0]])==1 else None
        if match is not None:merged.add(match)
        names=aliases(dict(raw,name=r['name']))|(aliases(gn[match]) if match is not None else set())
        rows.append(dict(entity_id=r['inventory_record_id'],name=r['name'],country_code=r['country_code'],longitude=float(r['longitude']),latitude=float(r['latitude']),source_record_ids=r['inventory_record_id']+(';' +gn[match]['node_id'] if match is not None else ''),geonames_ids=gn[match]['geonameid'] if match is not None else '',source_layer='country_source',source_url=r['source_url'],original_type=raw.get('type') or r['activity_type'],coordinate_status=r['coordinate_status'],identity_status='unique_source_gazetteer_correspondence' if match is not None else 'unmatched_source_record' if not g else 'ambiguous_source_gazetteer_correspondence',_names=names,_raw=raw))
        cross.append(dict(source_id=r['inventory_record_id'],name=r['name'],candidate_geonames_ids=';'.join(gn[k]['node_id'] for k in g),action='linked_to_same_inventory_unit' if match is not None else 'kept_separate_not_automatically_merged'))
    for i,r in enumerate(gn):
        if i in merged:continue
        rows.append(dict(entity_id=r['node_id'],name=r['name'],country_code=r['country_code'],longitude=float(r['longitude']),latitude=float(r['latitude']),source_record_ids=r['node_id'],geonames_ids=r['geonameid'],source_layer='GeoNames_supplement',source_url=r['source_url'],original_type=r['feature_code'],coordinate_status='gazetteer_point',identity_status='ambiguous_source_gazetteer_correspondence' if i in rev else 'gazetteer_only_candidate',_names=aliases(r),_raw=r))
    distances=coast_distance(rows)
    portdoc=json.loads((RAW/'WPI_DOE_north.json').read_text(encoding='utf-8'))
    cmap={'Russia':['RU'],'Canada':['CA'],'United States':['US'],'United States of America':['US'],'Norway':['NO','SJ'],'Denmark':['GL','FO'],'Greenland':['GL'],'Iceland':['IS'],'Finland':['FI'],'Sweden':['SE'],'Faroe Islands':['FO'],'Svalbard':['SJ']}
    ports=defaultdict(list)
    for f in portdoc['features']:
        a=f['attributes'];g=f.get('geometry')
        if not g:continue
        names={norm(a.get('main_port_name') or '')}
        names|={norm(n) for n in re.split('[;,]',a.get('alternate_name') or '') if n}
        for c in cmap.get(a.get('wpi_cc'),[]):
            for n in names:
                if n:ports[(c,n)].append((a,g))
    gl={r['entity_id']:r for r in read(ROOT/'实体整理/格陵兰官方居住实体_带位置与码头线索.csv')}
    audit=[]
    for r,d in zip(rows,distances):
        raw=r['_raw'];evidence=[];wpi=[];p=xyz([(r['longitude'],r['latitude'])])[0]
        for n in r['_names']:
            for a,g in ports.get((r['country_code'],n),[]):
                pp=xyz([(g['x'],g['y'])])[0]
                if km(p,pp)<=10:wpi.append(str(a['wpinumber']))
        if wpi:evidence.append('NGA_WPI_via_US_DOE_name_and_position_link')
        if r['country_code']=='GL' and gl.get(raw.get('居民点编号') or raw.get('entity_id'),{}).get('official_quay_edge_count','0') not in ['0','']:evidence.append('Asiaq_official_quay_same_locality_code')
        if r['country_code']=='US' and raw.get('source_harbor_dock')=='Yes':evidence.append('Alaska_DCRA_harbor_dock_Yes_mostly2018')
        r.update(coast_distance_km=round(d,6),coastline='GSHHG_2.3.7_high_L1_land_ocean_boundary',coastal_threshold_km=10,within_coastal_threshold=int(d<=10),port_status='documented_port_or_quay' if evidence else 'not_established_not_absence',port_evidence=';'.join(evidence),WPI_ids=';'.join(sorted(set(wpi))),historical_residence_1991_2025='not_established',marine_grid_mapping='deferred_by_user')
        nonresident=r['country_code']=='SJ' and r['name'] in ['Hornsund','Hopen Radio','Olonkinbyen']
        unresolved_sj=r['country_code']=='SJ' and r['name'] not in ['Longyearbyen','Barentsburg','Pyramiden','Ny-Ålesund','Hornsund','Hopen Radio','Olonkinbyen']
        invalidtype=r['country_code']=='RU' and raw.get('type') in ['остров','р-н']
        ambiguous=r['identity_status']=='ambiguous_source_gazetteer_correspondence'
        if invalidtype:
            status='排除';reason='来源为岛屿或地区，不是聚居点'
        elif nonresident:
            status='待定';reason='科考、气象或军事驻地：保留活动点来源，是否构成独立聚居单元待核，不因功能类型直接删除'
        elif unresolved_sj:
            status='待定';reason='社区内部地名、未确认站址或研究期内关闭的历史矿区；独立聚居身份及时间范围待明确，不按现状一概删除'
        elif ambiguous:
            status='待定';reason='来源实体对应有歧义，未强行合并或重复计点'
        elif 'approximate_named_locality_proxy' in r['coordinate_status']:
            status='待定';reason='只有近似位置，沿海界定尚不宜作为正式节点依据'
        elif evidence:
            status='保留';reason='有港口/码头来源证据，按用户规则保留，不受10km距离限制'
        elif d<=10:
            status='保留';reason='代表位置距海岸不超过10km；港口状态未确定，不写为确认无港口'
        else:
            status='待定';reason='距海岸超过10km且未取得港口证据；港口名录不完备，不能确认应排除'
        if r['source_layer']=='GeoNames_supplement':r['settlement_evidence']='地名候选；现存独立聚居身份和居住史尚未全部核定'
        else:r['settlement_evidence']='国家来源单元；不同来源定义及年份见原始分表'
        r.update(disposition=status,reason=reason,selection_level='资料规则筛选_不是全部人工核定的正式节点')
        del r['_raw'];del r['_names'];audit.append(r)
    write('全部来源实体_保留去除待定总表.csv',audit);write('国家来源与8019候选对应表.csv',cross)
    for status,name in [('保留','保留_港口及10km沿海聚居候选.csv'),('排除','排除_非聚居实体及独立活动设施.csv'),('待定','待定_逐条原因.csv')]:write(name,[r for r in audit if r['disposition']==status])
    # Every original source row has an explicit destination, including linked aliases.
    mapping={sid:r for r in audit for sid in r['source_record_ids'].split(';')}
    original_names={r['node_id']:r['name'] for r in gn}|{r['inventory_record_id']:r['name'] for r in official}
    source_audit=[]
    for sid in [r['node_id'] for r in gn]+[r['inventory_record_id'] for r in official]:
        r=mapping[sid];source_audit.append(dict(source_record_id=sid,source_original_name=original_names[sid],entity_id=r['entity_id'],name=r['name'],country_code=r['country_code'],disposition=r['disposition'],reason=r['reason'],coast_distance_km=r['coast_distance_km'],port_status=r['port_status']))
    write('原始10117条来源逐条去向.csv',source_audit)
    features=[dict(type='Feature',geometry=dict(type='Point',coordinates=[r['longitude'],r['latitude']]),properties=r) for r in audit if r['disposition']=='保留']
    (OUT/'保留候选.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=features),ensure_ascii=False),encoding='utf-8')
    stats=dict(initial_geonames_records=len(gn),country_source_records=len(official),total_source_rows=len(source_audit),linked_source_pairs=len(merged),inventory_units=len(audit),dispositions=dict(Counter(r['disposition'] for r in audit)),retained_documented_port=sum(r['disposition']=='保留' and r['port_status']=='documented_port_or_quay' for r in audit),retained_coastal_port_unknown=sum(r['disposition']=='保留' and r['port_status']!='documented_port_or_quay' for r in audit),country_counts={c:dict(Counter(r['disposition'] for r in audit if r['country_code']==c)) for c in sorted(set(r['country_code'] for r in audit))},formal_all_nodes_verified=False)
    assert len(source_audit)==10117 and len(mapping)==10117
    (OUT/'筛选统计.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()

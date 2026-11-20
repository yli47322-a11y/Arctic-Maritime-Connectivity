"""Build a source-preserving provisional settlement catalogue, not final study nodes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import csv, io, json, hashlib, math, time, zipfile
from datetime import datetime, timezone
import requests
import numpy as np
from scipy.spatial import cKDTree
import shapefile
from pyproj import CRS, Transformer
from matplotlib.path import Path as PolygonPath

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'原始数据'; RAW.mkdir(parents=True,exist_ok=True)
COUNTRIES=['CA','US','RU','GL','IS','NO','SJ','SE','FI','FO','GB','EE','AX','no-country']
COUNTRY_ZH={'CA':'加拿大','US':'美国','RU':'俄罗斯','GL':'格陵兰','IS':'冰岛','NO':'挪威','SJ':'斯瓦尔巴和扬马延','SE':'瑞典','FI':'芬兰','FO':'法罗群岛','GB':'英国','EE':'爱沙尼亚','AX':'奥兰群岛','':'未指定'}
ACTIVE={'PPL','PPLA','PPLA2','PPLA3','PPLA4','PPLA5','PPLC','PPLCD','PPLG','PPLF','PPLL','PPLR'}
EXCLUDED={'PPLQ':'abandoned','PPLH':'historical','PPLW':'destroyed','PPLCH':'historical_capital','PPLX':'section_of_settlement','PPLS':'group_of_settlements'}

def fetch(url,filename):
    p=RAW/filename
    if not p.exists():
        for attempt in range(3):
            try:
                with requests.get(url,timeout=(20,90),stream=True) as r:
                    r.raise_for_status()
                    with p.with_suffix(p.suffix+'.part').open('wb') as f:
                        for chunk in r.iter_content(1024*512):
                            f.write(chunk)
                if filename.endswith('.zip'):
                    with zipfile.ZipFile(p.with_suffix(p.suffix+'.part')) as z:
                        if not z.namelist(): raise ValueError('empty archive')
                p.with_suffix(p.suffix+'.part').replace(p)
                break
            except Exception:
                if attempt==2: raise
                time.sleep(2)
    return {'url':url,'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'retrieved_or_reused_utc':datetime.now(timezone.utc).isoformat()}

def write_csv(name,rows,fields):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    provenance=[]; failures={}
    jobs={c:(f'https://download.geonames.org/export/dump/{c}.zip',f'geonames_{c}.zip') for c in COUNTRIES}
    jobs['Nordregio']=('https://store.pangaea.de/Publications/Nordregio_2018/Settlement_locations.zip','Nordregio_2017_500plus.zip')
    jobs['coastline']=('https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip','ne_10m_coastline.zip')
    jobs['AMAP_boundary']=('https://www.amap.no/documents/download/1483/inline','AMAP_area.zip')
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(fetch,*v):k for k,v in jobs.items()}
        for fut in as_completed(futures):
            label=futures[fut]
            try: provenance.append(fut.result());print('SOURCE',label,'ready',flush=True)
            except Exception as e: failures[label]=str(e); print('SOURCE',label,'FAILED',str(e),flush=True)
    rows=[]; excluded=[]; facilities=[]
    for c in COUNTRIES:
        p=RAW/f'geonames_{c}.zip'
        if not p.exists(): continue
        with zipfile.ZipFile(p) as z:
            txt=c+'.txt'
            for line in io.TextIOWrapper(z.open(txt),encoding='utf-8'):
                a=line.rstrip('\n').split('\t')
                if len(a)!=19 or float(a[4])<50: continue
                code=a[7]
                if a[6]!='P' and code not in {'STNB','STNM'}: continue
                row=dict(node_id='GN_'+a[0],geonameid=a[0],name=a[1],ascii_name=a[2],alternate_names=a[3],latitude=float(a[4]),longitude=float(a[5]),country_code=a[8],country_zh=COUNTRY_ZH.get(a[8],a[8]),admin1_code=a[10],admin2_code=a[11],feature_code=code,population_reported=int(a[14] or 0),population_status='reported_nonzero_year_unspecified' if int(a[14] or 0)>0 else 'zero_or_unknown_not_confirmed_uninhabited',record_modified=a[18].strip(),source='GeoNames',source_url='https://www.geonames.org/'+a[0],latitude_band='Arctic_Circle_or_north' if float(a[4])>=66.5622 else '60N_to_Arctic_Circle',historical_start='',historical_end='',historical_status='not_established',node_status='candidate_not_final',marine_access_status='not_assessed')
                if a[6]=='P' and code in ACTIVE: rows.append(row)
                elif a[6]=='P': row['exclusion_reason']=EXCLUDED.get(code,'other_P_code_needs_review');excluded.append(row)
                else: row['facility_status']='research_or_weather_station_candidate_not_residential'; facilities.append(row)
    # Official AMAP polygon in its original polar projection avoids dateline issues.
    boundary=RAW/'AMAP_area.zip'
    if not boundary.exists(): raise RuntimeError('AMAP polygon unavailable; do not substitute a latitude boundary')
    with zipfile.ZipFile(boundary) as z:
        base='amaplim_lam_poly'
        reader=shapefile.Reader(shp=io.BytesIO(z.read(base+'.shp')),shx=io.BytesIO(z.read(base+'.shx')),dbf=io.BytesIO(z.read(base+'.dbf')))
        crs=CRS.from_wkt(z.read(base+'.prj').decode())
        transformer=Transformer.from_crs(4326,crs,always_xy=True)
        geometries=list(reader.iterShapes())
    def inside_amap(group):
        if not group: return []
        xx,yy=transformer.transform([r['longitude'] for r in group],[r['latitude'] for r in group])
        xy=np.column_stack([xx,yy]);inside=np.zeros(len(group),dtype=bool)
        for sh in geometries:
            points=np.asarray(sh.points);parts=list(sh.parts)+[len(points)];within=np.zeros(len(group),dtype=bool)
            for b,e in zip(parts[:-1],parts[1:]): within ^= PolygonPath(points[b:e]).contains_points(xy,radius=0.01)
            inside |= within
        return [dict(r,study_boundary='AMAP_official_GIS_868',inside_amap=1) for r,k in zip(group,inside) if k]
    rows=inside_amap(rows);excluded=inside_amap(excluded);facilities=inside_amap(facilities)
    # Approximate ocean-coast distance: sampled coastline, spherical metric.
    # Densify chords at <=1 km in ECEF; small islands omitted by Natural Earth remain a limitation.
    coast=RAW/'ne_10m_coastline.zip'
    if coast.exists():
        with zipfile.ZipFile(coast) as z:
            base=next(n[:-4] for n in z.namelist() if n.endswith('.shp'))
            reader=shapefile.Reader(shp=io.BytesIO(z.read(base+'.shp')),shx=io.BytesIO(z.read(base+'.shx')),dbf=io.BytesIO(z.read(base+'.dbf')))
            cloud=[]
            for sh in reader.iterShapes():
                if sh.bbox[3]<49: continue
                parts=list(sh.parts)+[len(sh.points)]
                for b,e in zip(parts[:-1],parts[1:]):
                    ll=np.radians(np.asarray(sh.points[b:e])); lat=ll[:,1];lon=ll[:,0]
                    xyz=np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
                    if len(xyz)<2: continue
                    for u,v in zip(xyz[:-1],xyz[1:]):
                        n=max(1,math.ceil(np.linalg.norm(v-u)*6371))
                        q=u+(v-u)*np.arange(n)[:,None]/n
                        cloud.append(q/np.linalg.norm(q,axis=1)[:,None])
                    cloud.append(xyz[-1:])
        tree=cKDTree(np.concatenate(cloud))
        for group in [rows,excluded,facilities]:
            if not group: continue
            ll=np.radians([[r['longitude'],r['latitude']] for r in group]);lon=ll[:,0];lat=ll[:,1]
            xyz=np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
            d,_=tree.query(xyz);km=2*np.arcsin(np.minimum(1,d/2))*6371
            for r,d in zip(group,km):
                r['approx_coast_distance_km']=round(float(d),3)
                r['coast_distance_band']='0-5km' if d<=5 else '5-10km' if d<=10 else '10-25km' if d<=25 else '25-50km' if d<=50 else '>50km'
                r['coast_distance_method']='NaturalEarth_10m_coastline_spherical_samples_le_1km_not_port_distance'
    fields=list(rows[0]) if rows else []
    rows.sort(key=lambda r:(r['country_code'],r.get('approx_coast_distance_km',1e9),r['name']))
    if rows:
        write_csv('聚居点候选母表_AMAP.csv',rows,fields)
        write_csv('近海候选浏览表_距岸25km内_非正式筛选.csv',[r for r in rows if r.get('approx_coast_distance_km',1e9)<=25],fields)
        (ROOT/'聚居点候选母表_AMAP.geojson').write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','geometry':{'type':'Point','coordinates':[r['longitude'],r['latitude']]},'properties':r} for r in rows]},ensure_ascii=False),encoding='utf-8')
        write_csv('暂不纳入的地名记录.csv',excluded,fields+['exclusion_reason'])
        write_csv('科考气象站候选_未核定.csv',facilities,fields+['facility_status'])
    summary=[]
    for c in sorted(set(r['country_code'] for r in rows)):
        a=[r for r in rows if r['country_code']==c]
        summary.append(dict(country_code=c,country_zh=COUNTRY_ZH.get(c,c),candidates_AMAP=len(a),candidates_ArcticCircle=sum(r['latitude']>=66.5622 for r in a),approx_within25km=sum(r.get('approx_coast_distance_km',1e9)<=25 for r in a),population_zero_or_unknown=sum(r['population_reported']==0 for r in a)))
    if summary: write_csv('按国家地区汇总.csv',summary,list(summary[0]))
    report=dict(retrieval_date='2026-11-01',boundary='AMAP_official_GIS_868',boundary_url='https://www.amap.no/work-area/document/868',boundary_user_confirmed=True,coastal_threshold_final=False,population_threshold=None,geonames_codes_included=sorted(ACTIVE),candidate_count=len(rows),near_coast_25km_browsing_count=sum(r.get('approx_coast_distance_km',1e9)<=25 for r in rows),excluded_count=len(excluded),research_weather_candidates=len(facilities),duplicate_geonameids=len(rows)-len({r['geonameid'] for r in rows}),country_summary=summary,failures=failures,sources=provenance)
    (ROOT/'整理记录.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'},ensure_ascii=False),flush=True)

if __name__=='__main__': main()

"""Attach settlement-source identities and landing-place clues; do not infer final nodes."""
from pathlib import Path
from collections import defaultdict, Counter
import csv, io, json, re, unicodedata, zipfile, hashlib
import numpy as np
from scipy.spatial import cKDTree
import shapefile
from pyproj import CRS, Transformer
from matplotlib.path import Path as PolygonPath

ROOT=Path(__file__).resolve().parent;RAW=ROOT/'原始数据';OUT=ROOT/'实体整理';OUT.mkdir(exist_ok=True)

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(name,rows,fields=None):
    if fields is None: fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def norm(s):
    s=s.casefold().translate(str.maketrans({'ø':'o','æ':'ae','ð':'d','þ':'th','ł':'l','ß':'ss'}))
    s=''.join(c for c in unicodedata.normalize('NFKD',s) if not unicodedata.combining(c))
    return ''.join(c for c in s if c.isalnum())

def xyz(rows):
    ll=np.radians([[float(r['longitude']),float(r['latitude'])] for r in rows]);lon,lat=ll.T
    return np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])

def distance(a,b):
    p=xyz([a,b]);return float(2*np.arcsin(min(1,np.linalg.norm(p[0]-p[1])/2))*6371)

def main():
    nodes=read(ROOT/'聚居点候选母表_AMAP.csv')
    aliases={};index=defaultdict(set)
    for i,r in enumerate(nodes):
        names={norm(s) for s in [r['name'],r['ascii_name'],*r['alternate_names'].split(',')] if s}
        aliases[i]=names
        for s in names:index[(r['country_code'],s)].add(i)
        r.update(nordregio_link='',official_greenland_link='',official_population_1991='',official_population_2025='',entity_evidence='geonames_candidate_only',nearest_landing_clue_id='',nearest_landing_clue_name='',nearest_landing_clue_type='',landing_clue_distance_km='',same_name_landing_clue='0',possible_duplicate_count=0,formal_node_selected='0',marine_entrance_confirmed='0')

    # Match named independent records, retaining all ambiguous or geographically discordant cases.
    z=zipfile.ZipFile(RAW/'Nordregio_2017_500plus.zip')
    source=list(csv.DictReader(z.read('Settlement_locations.csv').decode('cp1252').splitlines(),delimiter=';'))
    links=[]
    for n,s in enumerate(source,1):
        name=s.get('Settlement Name','');country=s.get('Country','')
        if not name.strip(): continue
        country={'AL':'US','Ru':'RU'}.get(country,country)
        base=re.sub(r',\s*(VN|T|CG|NO|C|TV|HAM|CY|VL|TP|M|RGM)$','',name,flags=re.I)
        parts={norm(name),norm(base)}
        candidates=set()
        for namekey in parts:
            if country:candidates |= index.get((country,namekey),set())
            else:
                for c in {r['country_code'] for r in nodes}:candidates |= index.get((c,namekey),set())
        point={'latitude':s['Lat (epsg:4326)'],'longitude':s['Lon (epsg:4326)']}
        ordered=sorted([(distance(nodes[i],point),i) for i in candidates])
        close=[(d,i) for d,i in ordered if d<=10]
        status='unique_name_and_coordinate_link' if len(close)==1 else 'ambiguous_multiple_candidates' if close else 'no_unique_link'
        matched=close[0][1] if len(close)==1 else None
        row=dict(source_id=f'NORD2017_{n:04}',source_name=name,source_country=country,source_latitude=point['latitude'],source_longitude=point['longitude'],link_status=status,candidate_node_ids=';'.join(nodes[i]['node_id'] for d,i in close),nearest_named_candidate_km=round(ordered[0][0],3) if ordered else '',linked_node_id=nodes[matched]['node_id'] if matched is not None else '',source_scope='Nordregio_source_extent_not_yet_AMAP_clipped')
        links.append(row)
        if matched is not None:nodes[matched]['nordregio_link']+=';'+row['source_id']
    write('Nordregio与候选点对应表.csv',links)

    # Official Greenland locality identifiers; aggregate/unknown units are not entities.
    d=json.loads((RAW/'greenland_population_1991_2025.json').read_text(encoding='utf-8-sig'))
    loc=d['dimension']['locality']['category'];vals=d['value'];greenland=[]
    for code,pos in loc['index'].items():
        label=loc['label'][code];base=re.sub(r'\([^)]*\)','',label).strip()
        kind='aggregate_or_unknown' if code=='0000000GRL' or 'ZZZ' in code else 'station' if '(Station)' in label else 'airport' if '(Airport)' in label else 'residential_locality'
        # '(KAN)' and '(UPE)' disambiguate homonyms in the source but do not supply coordinates.
        candidates=sorted(index.get(('GL',norm(base)),set())) if kind!='aggregate_or_unknown' else []
        status='unique_name_link_needs_position_confirmation' if len(candidates)==1 else 'ambiguous_same_name' if len(candidates)>1 else 'not_linked'
        p91=vals[pos*2];p25=vals[pos*2+1]
        row=dict(source_locality_id=code,source_locality_name=label,locality_type=kind,population_1991=p91,population_2025=p25,link_status=status,candidate_node_ids=';'.join(nodes[i]['node_id'] for i in candidates),source_url='https://bank.stat.gl/pxweb/en/Greenland/Greenland__BE__BE01__BE0120/BEXSTD.PX/')
        greenland.append(row)
        if len(candidates)==1 and kind=='residential_locality':
            r=nodes[candidates[0]];r['official_greenland_link']=code;r['official_population_1991']=p91;r['official_population_2025']=p25
    write('格陵兰官方居住地及人口_1991_2025.csv',greenland)

    # Preserve possible duplicate IDs; physical closeness by itself never triggers merging.
    tree=cKDTree(xyz(nodes));duplicate=[]
    for i,j in sorted(tree.query_pairs(2*np.sin(1/6371/2))):
        common=aliases[i]&aliases[j]
        if not common:continue
        if nodes[i]['country_code']!=nodes[j]['country_code']:continue
        dd=distance(nodes[i],nodes[j])
        duplicate.append(dict(node_id_1=nodes[i]['node_id'],name_1=nodes[i]['name'],node_id_2=nodes[j]['node_id'],name_2=nodes[j]['name'],distance_km=round(dd,3),common_normalized_name=';'.join(sorted(common)),action='retain_both_pending_entity_resolution'))
        nodes[i]['possible_duplicate_count']+=1;nodes[j]['possible_duplicate_count']+=1
    write('同名近邻待辨识.csv',duplicate,['node_id_1','name_1','node_id_2','name_2','distance_km','common_normalized_name','action'])

    # Geographic entrance clues from original gazetteer, not new experimental validation.
    facilities=[];types={'HBR','ANCH','WHRF','PIER','DCK','DCKB','PRT'}
    for file in sorted(RAW.glob('geonames_*.zip')):
        c=file.stem.removeprefix('geonames_')
        with zipfile.ZipFile(file) as z:
            for line in io.TextIOWrapper(z.open(c+'.txt'),encoding='utf-8'):
                a=line.rstrip('\n').split('\t')
                if len(a)!=19 or a[7] not in types or float(a[4])<50:continue
                facilities.append(dict(clue_id='GN_'+a[0],name=a[1],alternate_names=a[3],latitude=float(a[4]),longitude=float(a[5]),feature_code=a[7],country_code=a[8],source_url='https://www.geonames.org/'+a[0],clue_status='geographic_feature_not_confirmed_operating_facility'))
    with zipfile.ZipFile(RAW/'AMAP_area.zip') as z:
        b='amaplim_lam_poly';reader=shapefile.Reader(shp=io.BytesIO(z.read(b+'.shp')),shx=io.BytesIO(z.read(b+'.shx')),dbf=io.BytesIO(z.read(b+'.dbf')))
        tr=Transformer.from_crs(4326,CRS.from_wkt(z.read(b+'.prj').decode()),always_xy=True)
        xx,yy=tr.transform([r['longitude'] for r in facilities],[r['latitude'] for r in facilities]);points=np.column_stack([xx,yy]);inside=np.zeros(len(facilities),dtype=bool)
        for sh in reader.iterShapes():
            cuts=list(sh.parts)+[len(sh.points)];within=np.zeros(len(facilities),dtype=bool)
            for a,b in zip(cuts[:-1],cuts[1:]):within ^= PolygonPath(np.asarray(sh.points[a:b])).contains_points(points,radius=.01)
            inside |= within
        facilities=[r for r,k in zip(facilities,inside) if k]
    write('海港锚地码头地理线索_AMAP.csv',facilities)
    if facilities:
        tree=cKDTree(xyz(facilities));dd,ii=tree.query(xyz(nodes))
        for k,r in enumerate(nodes):
            p=facilities[ii[k]];km=float(2*np.arcsin(min(1,dd[k]/2))*6371)
            r.update(nearest_landing_clue_id=p['clue_id'],nearest_landing_clue_name=p['name'],nearest_landing_clue_type=p['feature_code'],landing_clue_distance_km=round(km,3),same_name_landing_clue=str(int(bool(aliases[k]&{norm(p['name']),*map(norm,p['alternate_names'].split(','))}))))
            # A nearest port is a clue only, even when names agree. No snapping or passability claim.

    for r in nodes:
        evidence=[]
        if r['nordregio_link']:evidence.append('named_in_Nordregio_2017')
        if r['official_greenland_link']:evidence.append('official_Greenland_locality_name_link')
        if evidence:r['entity_evidence']=';'.join(evidence)
    write('聚居点实体整理工作表_AMAP.csv',nodes)
    documented=[r for r in nodes if r['entity_evidence']!='geonames_candidate_only']
    documented.sort(key=lambda r:(r['country_code'],float(r['approx_coast_distance_km']),r['name']))
    write('已有聚居实体来源的候选点.csv',documented,list(nodes[0]))
    write('各国整理进度.csv',[dict(country_code=c,candidates=len(a),named_settlement_source_links=sum(r['entity_evidence']!='geonames_candidate_only' for r in a),official_Greenland_links=sum(bool(r['official_greenland_link']) for r in a),same_name_landing_clues=sum(r['same_name_landing_clue']=='1' for r in a),marine_entrances_confirmed=0) for c in sorted({r['country_code'] for r in nodes}) for a in [[r for r in nodes if r['country_code']==c]]])
    summary=dict(total_candidates=len(nodes),documented_settlement_candidates=len(documented),Nordregio_record_link_status=dict(Counter(r['link_status'] for r in links)),Greenland_source_types=dict(Counter(r['locality_type'] for r in greenland)),Greenland_link_status=dict(Counter(r['link_status'] for r in greenland if r['locality_type']=='residential_locality')),Greenland_residential_locality_positive_2025=sum(isinstance(r['population_2025'],(int,float)) and r['population_2025']>0 for r in greenland if r['locality_type']=='residential_locality'),possible_duplicate_pairs=len(duplicate),harbour_landing_geographic_clues=len(facilities),same_name_nearest_landing_clues=sum(r['same_name_landing_clue']=='1' for r in nodes),confirmed_marine_entrances=0,formal_nodes_selected=0,matching_parameters={'Nordregio_name_and_position_maximum_km':10,'duplicate_name_neighbour_radius_km':1,'parameters_are_linking_review_tools_not_inclusion_or_experiment_thresholds':True},sources=[{'url':'https://www.geonames.org/export/codes.html','role':'geographic features, not operating-port records'},{'url':'https://doi.org/10.1594/PANGAEA.895745','role':'named 2017 settlements, population>=500 subset'},{'url':'https://bank.stat.gl/api/v1/en/Greenland/BE/BE01/BE0120/BEXSTD.PX','role':'official locality identities and two endpoint populations'}],unavailable_sources={'NGA_WPI':'official service timed out; not downloaded or substituted','US_Census_places_2020':'official file returned403; not downloaded or used'},file_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW/'greenland_metadata.json',RAW/'greenland_population_1991_2025.json',RAW/'greenland_population_query.json']})
    (OUT/'整理进度.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

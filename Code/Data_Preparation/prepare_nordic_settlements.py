"""Sourced Nordic settlement inventories; no marine-grid selection."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,xml.etree.ElementTree as ET,zipfile,io,unicodedata,re,hashlib
import numpy as np
import shapefile
from pyproj import Transformer,CRS
from 整理加拿大与阿拉斯加 import Boundary,write,geo,OUT,RAW,ROOT

def norm(s):
    s=s.casefold().translate(str.maketrans({'ø':'o','æ':'ae','ð':'d','þ':'th','ł':'l','ß':'ss'}))
    return ''.join(c for c in unicodedata.normalize('NFKD',s) if c.isalnum() and not unicodedata.combining(c))

def ring_centroid(a):
    a=np.asarray(a);x=a[:,0]-a[0,0];y=a[:,1]-a[0,1]
    # Translate near origin to avoid cancellation for small rings at large projected coordinates.
    if np.any(a[0]!=a[-1]):x=np.r_[x,x[0]];y=np.r_[y,y[0]]
    q=x[:-1]*y[1:]-x[1:]*y[:-1];area=q.sum()/2
    if abs(area)<1e-8:return 0,float(a[:,0].mean()),float(a[:,1].mean())
    return abs(area),float(((x[:-1]+x[1:])*q).sum()/(6*area)+a[0,0]),float(((y[:-1]+y[1:])*q).sum()/(6*area)+a[0,1])

def source_base(entity_id,name,country,lon,lat,url,kind,year):
    return dict(entity_id=entity_id,name=name,country_code=country,longitude=lon,latitude=lat,source_url=url,source_entity_type=kind,population_reported='',population_reference_year=year,population_status='source_count_not_available',inside_amap='',marine_grid_mapping='deferred_by_user_2026-11-01',historical_residence_1991_2025='not_established',entity_status='sourced_inventory_unit_requires_final_scope')

def norway(boundary):
    p=RAW/'norway_tettsteder_2025_north.gml';d=ET.parse(p).getroot();ns={'wfs':'http://www.opengis.net/wfs/2.0','gml':'http://www.opengis.net/gml/3.2'};tr=Transformer.from_crs(25833,4326,always_xy=True);rows=[]
    if d.attrib.get('numberMatched')!=d.attrib.get('numberReturned'):raise ValueError('Norway source response truncated')
    for member in d.findall('wfs:member',ns):
        f=member[0];props={a.tag.split('}')[-1]:(a.text or '').strip() for a in f if len(a)==0};rings=[]
        for ring in f.findall('.//gml:exterior/gml:LinearRing/gml:posList',ns):rings.append(ring_centroid(np.fromstring(ring.text,sep=' ').reshape(-1,int(ring.attrib.get('srsDimension','2')))))
        if not rings:continue
        _,x,y=max(rings);lon,lat=tr.transform(x,y);r=source_base('NO_SSB_'+props['tett_nr'],props['tettstedsnavn'],'NO',lon,lat,'https://kart.ssb.no/api/mapserver/v1/wfs/tettsteder','statistical_urban_settlement',2025);r.update(population_reported=int(props['befolkning_tettsted']),population_status='official_resident_count',source_population_threshold='at_least200_source_definition_not_project_cutoff',position_method='centroid_of_largest_official_polygon_exterior_EPSG25833',source_settlement_code=props['tett_nr']);rows.append(r)
    boundary.classify(rows);inside=[r for r in rows if r['inside_amap']==1];write('挪威_AMAP内官方统计聚居区_2025.csv',inside);geo('挪威_AMAP内官方统计聚居区_2025.geojson',inside)
    return {'official_units_inside_AMAP':len(inside),'retrieved_northern_units':len(rows),'source_population_floor':200,'smaller_places_retained_in_GeoNames_supplement':True}

def finland(boundary):
    rows=[]
    with zipfile.ZipFile(RAW/'Finland_YKRTaajamat2025.zip') as z:
        b='YKRTaajama25';rd=shapefile.Reader(shp=io.BytesIO(z.read(b+'.shp')),shx=io.BytesIO(z.read(b+'.shx')),dbf=io.BytesIO(z.read(b+'.dbf')),encoding='utf-8');tr=Transformer.from_crs(CRS.from_wkt(z.read(b+'.prj').decode()),4326,always_xy=True)
        for sr in rd.iterShapeRecords():
            a=sr.record.as_dict();s=sr.shape
            if s.bbox[3]<7200000:continue
            cuts=list(s.parts)+[len(s.points)];rings=[ring_centroid(s.points[b:e]) for b,e in zip(cuts[:-1],cuts[1:])];_,x,y=max(rings);lon,lat=tr.transform(x,y);r=source_base('FI_SYKE_YKR2025_'+str(a['ykr25nro']),a['taajnimi'],'FI',lon,lat,'https://wwwd3.ymparisto.fi/d3/gis_data/spesific/YKRTaajamat2025.zip','statistical_urban_area',2025);r.update(swedish_name=a['taajnimiru'],statistical_urban_code=a['tktaajtunn'],source_population_class_code=a['vaesto25lk'],source_population_threshold='urban_area_source_definition; small dispersed places not comprehensively included',position_method='centroid_of_largest_official_polygon_ring_EPSG3067',source_population_class_note='vaesto25lk is a class code, not an exact population count');rows.append(r)
    boundary.classify(rows);inside=[r for r in rows if r['inside_amap']==1];write('芬兰_AMAP内官方聚居区_2025.csv',inside);geo('芬兰_AMAP内官方聚居区_2025.geojson',inside)
    return {'official_units_inside_AMAP':len(inside),'exact_population_counts_available':False,'source_population_classes_preserved':True,'small_dispersed_places_in_separate_supplement':True}

def geonames_index(country):
    records=[];index=defaultdict(set)
    with zipfile.ZipFile(RAW/f'geonames_{country}.zip') as z:
        for line in io.TextIOWrapper(z.open(country+'.txt'),encoding='utf-8'):
            a=line.rstrip('\n').split('\t')
            if len(a)!=19 or a[6]!='P':continue
            r=dict(id=a[0],name=a[1],longitude=float(a[5]),latitude=float(a[4]),feature_code=a[7]);i=len(records);records.append(r)
            for s in [a[1],a[2],*a[3].split(',')]:
                if s:index[norm(s)].add(i)
    return records,index

def population_places(boundary,country,filename,dim,url):
    d=json.loads((RAW/filename).read_text(encoding='utf-8'));c=d['dimension'][dim]['category'];counts=d['value'];rows=[];aggregates=[];gn,index=geonames_index(country)
    official_points=[];point_names=defaultdict(list);point_post=defaultdict(list);icelandic_names={}
    if country=='FO' and (RAW/'faroe_official_villages.geojson').exists():
        official_points=json.loads((RAW/'faroe_official_villages.geojson').read_text(encoding='utf-8'))['features']
        for f in official_points:
            point_names[norm(f['properties']['stad'])].append(f)
            if f['properties']['postnr'] is not None:point_post[str(f['properties']['postnr'])].append(f)
    elif country=='IS' and (RAW/'iceland_official_map1.csv').exists():
        meta=json.loads((RAW/'iceland_nuclei_metadata_is.json').read_text(encoding='utf-8'));v=next(v for v in meta['variables'] if v['code']==dim);icelandic_names=dict(zip(v['values'],v['valueTexts']))
        for a in csv.reader((RAW/'iceland_official_map1.csv').open(encoding='utf-8-sig')):
            if len(a)>=4 and a[2].strip() and a[3].strip():point_names[norm(a[0])].append((float(a[3]),float(a[2])))
    for code,i in c['index'].items():
        name=c['label'][code];value=counts[i]
        if code in {'0','999999'} or name.startswith('Rural areas'):
            aggregates.append({'code':code,'name':name,'population_2025':value,'reason':'total_or_rural_area_aggregate_not_one_settlement'});continue
        matches=sorted(index.get(norm(name),set()));lon=lat='';method='not_positioned';ids=''
        if len(matches)==1:
            a=gn[matches[0]];lon,lat=a['longitude'],a['latitude'];ids=a['id'];method='unique_official_name_to_GeoNames_alias_correspondence'
        coord_source='https://www.geonames.org/'+ids if ids else '';cartographic_id='';coord_status='linked' if len(matches)==1 else 'ambiguous_same_name' if matches else 'no_name_correspondence'
        if country=='FO' and official_points:
            choices=point_names.get(norm(name),[])
            if not choices and len(code)>=3:choices=point_post.get(str(int(code[-3:])),[])
            if not choices and ', ' in name and 'municipality' in name:choices=point_names.get(norm(name.split(',')[0]),[])
            if len(choices)==1:
                f=choices[0];lon,lat=f['geometry']['coordinates'][:2];cartographic_id=str(f['properties']['OBJECTID']);method='official_Faroe_environment_agency_village_point';coord_source='https://gis.us.fo/arcgis/rest/services/fyrisitingarlig_kort/us_bygdir/MapServer/0';coord_status='official_cartographic_point'
        elif country=='IS' and lon=='':
            choices=point_names.get(norm(icelandic_names.get(code,name)),[])
            if len(choices)==1:lon,lat=choices[0];method='Statistics_Iceland_published_map_symbol_position_2024';coord_source='https://datawrapper.dwcdn.net/z33Wu/3/';coord_status='official_published_map_position_proxy'
        if lon=='':
            # A unique short base name can resolve a source qualifier; ambiguous bases remain unresolved.
            short=name.split(',')[0];alt=sorted(index.get(norm(short),set())) if ',' in name else []
            if len(alt)==1:
                a=gn[alt[0]];lon,lat=a['longitude'],a['latitude'];ids=a['id'];method='unique_base_name_GeoNames_coordinate_proposal';coord_source='https://www.geonames.org/'+ids;coord_status='base_name_link_needs_qualifier_confirmation'
        status='positive_population_2025' if isinstance(value,(int,float)) and value>0 else 'zero_population_2025' if value==0 else 'missing_or_not_applicable_in_source'
        r=source_base(country+'_STAT_'+code,name,country,lon,lat,url,'statistical_urban_nucleus' if country=='IS' else 'official_village_or_city',2025);r.update(population_reported=value,population_status=status,source_locality_code=code,local_language_name=icelandic_names.get(code,name),position_method=method,geonames_correspondence_ids=ids,coordinate_source=coord_source,official_map_objectid=cartographic_id,coordinate_candidates=';'.join(gn[k]['id'] for k in matches),coordinate_status=coord_status,inside_amap='not_positioned' if lon=='' else '');rows.append(r)
    boundary.classify(rows);zh='冰岛' if country=='IS' else '法罗群岛';write(zh+'_官方聚居实体及人口_2025.csv',rows);geo(zh+'_官方聚居实体_有位置.geojson',rows);write(zh+'_统计汇总区域_不作居民点.csv',aggregates)
    unresolved=[r for r in rows if r['longitude']==''];write(zh+'_坐标待辨识.csv',unresolved,list(rows[0]));return {'official_named_units':len(rows),'positive_population2025':sum(r['population_status']=='positive_population_2025' for r in rows),'zero_population2025':sum(r['population_status']=='zero_population_2025' for r in rows),'missing_population2025':sum(r['population_status']=='missing_or_not_applicable_in_source' for r in rows),'positioned':sum(r['longitude']!='' for r in rows),'inside_AMAP_positioned':sum(r['inside_amap']==1 for r in rows),'coordinate_unresolved':len(unresolved),'aggregate_units_separated':len(aggregates),'coordinate_method_counts':dict(Counter(r['coordinate_status'] for r in rows)),'note':'Unresolved entities retained; coordinates cite either official mapping, official published-map symbols or GeoNames name correspondences; no sea entrances assigned.'}

def main():
    b=Boundary();stats={'Norway':norway(b),'Finland':finland(b),'Iceland':population_places(b,'IS','iceland_nuclei_2025.json','Byggðakjarnar','https://px.hagstofa.is/pxen/api/v1/en/Ibuar/mannfjoldi/2_byggdir/Byggdakjarnar/MAN030101.px'),'Faroe':population_places(b,'FO','faroe_population_2025.json','village/city','https://statbank.hagstova.fo/api/v1/en/H2/IB/IB01/fo_abgd_md_t.px'),'marine_grid_mapping':'deferred_by_user','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW/'norway_tettsteder_2025_north.gml',RAW/'Finland_YKRTaajamat2025.zip',RAW/'iceland_nuclei_2025.json',RAW/'faroe_population_2025.json']}}
    for filename in ['faroe_official_villages.geojson','iceland_official_map1.csv','iceland_nuclei_metadata_is.json']:
        if (RAW/filename).exists():stats['source_hashes'][filename]=hashlib.sha256((RAW/filename).read_bytes()).hexdigest()
    (OUT/'北欧分国整理记录.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

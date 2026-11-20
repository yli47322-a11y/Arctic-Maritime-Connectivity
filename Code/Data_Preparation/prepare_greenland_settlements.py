"""Join Statistics Greenland localities to Asiaq mapping by locality code."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,hashlib
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent;RAW=ROOT/'原始数据';OUT=ROOT/'实体整理'
def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def read(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
to_xy=Transformer.from_crs(4326,3575,always_xy=True);to_ll=Transformer.from_crs(3575,4326,always_xy=True)
def centroid(geometry):
    sx=sy=sa=0
    for ring in geometry['coordinates']:
        a=np.array(ring);x,y=map(np.asarray,to_xy.transform(a[:,0],a[:,1]));q=x[:-1]*y[1:]-x[1:]*y[:-1]
        area=q.sum()/2
        if abs(area)<1e-6:continue
        cx=((x[:-1]+x[1:])*q).sum()/(6*area);cy=((y[:-1]+y[1:])*q).sum()/(6*area)
        sx+=cx*area;sy+=cy*area;sa+=area
    if not sa:raise ValueError('degenerate polygon')
    return abs(sa),sx/sa,sy/sa
def line_midpoint(geometry):
    parts=geometry['coordinates'] if geometry['type']=='MultiLineString' else [geometry['coordinates']]
    starts=[];ends=[]
    for part in parts:
        a=np.asarray(part);x,y=map(np.asarray,to_xy.transform(a[:,0],a[:,1]));p=np.column_stack([x,y]);starts.extend(p[:-1]);ends.extend(p[1:])
    starts=np.asarray(starts);ends=np.asarray(ends);length=np.linalg.norm(ends-starts,axis=1);target=length.sum()/2;c=np.cumsum(length);i=min(int(np.searchsorted(c,target)),len(length)-1);before=c[i-1] if i else 0;t=(target-before)/length[i] if length[i] else 0
    xy=starts[i]+t*(ends[i]-starts[i]);lon,lat=to_ll.transform(*xy);return float(lon),float(lat)

def main():
    localities=read('格陵兰官方居住地及人口_1991_2025.csv')
    built=json.loads((RAW/'asiaq_layer43.geojson').read_text(encoding='utf-8'))['features'];quays=json.loads((RAW/'asiaq_layer16.geojson').read_text(encoding='utf-8'))['features']
    areas=defaultdict(list);ports=defaultdict(list)
    for r in built:areas[r['properties']['lokalitet']].append(centroid(r['geometry']))
    entrances=[]
    for r in quays:
        p=r['properties'];lon,lat=line_midpoint(r['geometry']);row=dict(entrance_id='ASIAQ_QUAY_'+str(p['OBJECTID']),asiaq_locality_code=p['lokalitet'],longitude=lon,latitude=lat,coordinate_method='length_midpoint_of_official_quay_edge_EPSG3575',source_feature_id=p['lokal_id'],source_url='https://kort.nunagis.gl/refserver/rest/services/TekniskGrundkort/TekniskGrundkort/MapServer/16',feature_evidence='mapped_quay_edge',operational_status='not_established',marine_grid_cell='not_assigned',ship_class_suitability='not_assessed')
        ports[p['lokalitet']].append(row);entrances.append(row)
    current=[]
    for r in localities:
        if r['locality_type']!='residential_locality':continue
        code=r['source_locality_id'][-7:];polygons=areas.get(code,[]);pp=ports.get(code,[])
        lon=lat='';method='not_positioned'
        if polygons:
            # Use the largest built-up core; some same-code polygons lie far apart.
            # Do not average disjoint remote mapped patches into a false settlement position.
            _,x,y=max(polygons,key=lambda p:p[0]);lon,lat=to_ll.transform(x,y);method='centroid_of_largest_official_built_up_polygon_EPSG3575'
        elif pp:
            lon=float(np.mean([p['longitude'] for p in pp]));lat=float(np.mean([p['latitude'] for p in pp]));method='mean_of_official_quay_midpoints_position_proxy'
        placename_id='';position_url='https://kort.nunagis.gl/refserver/rest/services/TekniskGrundkort/TekniskGrundkort/MapServer/'+('43' if polygons else '16') if method!='not_positioned' else ''
        if code=='1611IKK' and method=='not_positioned' and (RAW/'official_residential_placenames.geojson').exists():
            # Distinguish the tiny Upernavik locality from identically named villages and straits.
            named=json.loads((RAW/'official_residential_placenames.geojson').read_text(encoding='utf-8'))['features']
            matches=[p for p in named if p['properties']['OBJECTID']==45593020 and p['properties']['Type']==24 and p['properties']['PlacenameOfficial']=='Ikerasaarsuk']
            if len(matches)!=1:raise ValueError('Expected one pinned official Ikerasaarsuk residential feature')
            _,xx,yy=centroid(matches[0]['geometry']);lon,lat=to_ll.transform(xx,yy);method='centroid_of_official_residential_placename_polygon';placename_id='45593020';position_url='https://kort.nunagis.gl/refserver/rest/services/PlacenamesRegister/PlacenamesRegisterPublic/MapServer/0'
        current.append(dict(entity_id='GL_STAT_'+r['source_locality_id'],official_locality_code=r['source_locality_id'],asiaq_locality_code=code,name=r['source_locality_name'],country_code='GL',population_1991=r['population_1991'],population_2025=r['population_2025'],longitude=lon,latitude=lat,position_method=method,position_source_url=position_url,official_placename_objectid=placename_id,official_built_up_polygon_count=len(polygons),official_quay_edge_count=len(pp),entrance_ids=';'.join(p['entrance_id'] for p in pp),marine_entrance_evidence='official_mapped_quay_edges' if pp else 'no_quay_record_does_not_imply_no_sea_access',settlement_existence_evidence='official_population_2025_positive',historical_continuity_1991_2025='not_established_from_two_endpoints',TOPAZ_cell_mapping='not_assigned',formal_network_node='not_final'))
    # Correspondence proposals preserve one official locality entity even if GeoNames has duplicate IDs.
    candidates=list(csv.DictReader((ROOT/'聚居点候选母表_AMAP.csv').open(encoding='utf-8-sig')));gn=[r for r in candidates if r['country_code']=='GL']
    from 整理实体与海上入口线索 import norm,distance
    alias=[{norm(s) for s in [r['name'],r['ascii_name'],*r['alternate_names'].split(',')] if s} for r in gn]
    for r in current:
        if r['longitude']=='':r.update(geonames_proposal_ids='',geonames_proposal_status='official_entity_retained_position_missing',geonames_proposal_distance_km='');continue
        close=sorted((distance(r,p),i) for i,p in enumerate(gn))
        name=norm(r['name'].split('(')[0].strip());named=[(d,i) for d,i in close if d<=10 and name in alias[i]]
        if len(named)==1:
            proposed=named;status='name_and_official_position_correspondence'
        elif len(named)>1:proposed=named;status='multiple_GeoNames_records_near_one_official_entity'
        elif close and close[0][0]<=1:proposed=[close[0]];status='position_only_proposal_name_needs_resolution'
        else:proposed=[];status='official_entity_retained_without_GeoNames_link'
        r.update(geonames_proposal_ids=';'.join(gn[i]['node_id'] for d,i in proposed),geonames_proposal_status=status,geonames_proposal_distance_km=';'.join(str(round(d,3)) for d,i in proposed))
    write('格陵兰官方居住实体_带位置与码头线索.csv',current)
    write('格陵兰官方码头岸线入口点.csv',entrances)
    (OUT/'格陵兰官方居住实体_有位置.geojson').write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','geometry':{'type':'Point','coordinates':[r['longitude'],r['latitude']]},'properties':r} for r in current if r['longitude']!='']},ensure_ascii=False),encoding='utf-8')
    officialcodes={r['asiaq_locality_code'] for r in current}
    extras=[dict(asiaq_locality_code=c,built_polygon_count=len(areas.get(c,[])),quay_edge_count=len(ports.get(c,[])),status='mapped_locality_absent_from_selected_2025_residential_table_do_not_assume_currently_resident') for c in sorted((set(areas)|set(ports))-officialcodes)]
    write('格陵兰地图中其他地点_不自动认定常住.csv',extras)
    summary=dict(official_residential_entities=len(current),entities_with_positions=sum(r['longitude']!='' for r in current),entities_with_official_quay_edges=sum(r['official_quay_edge_count']>0 for r in current),entities_without_quay_record=sum(r['official_quay_edge_count']==0 for r in current),unpositioned_entities=[r['name'] for r in current if r['longitude']==''],total_mapped_quay_edges=len(entrances),quay_edges_linked_to_selected_residential_entities=sum(r['official_quay_edge_count'] for r in current),correspondence_status=dict(Counter(r['geonames_proposal_status'] for r in current)),other_mapped_locality_codes=[r['asiaq_locality_code'] for r in extras],population_reference_years=[1991,2025],sources={'population':'https://bank.stat.gl/api/v1/en/Greenland/BE/BE01/BE0120/BEXSTD.PX','built_up_areas':'https://kort.nunagis.gl/refserver/rest/services/TekniskGrundkort/TekniskGrundkort/MapServer/43','quay_edges':'https://kort.nunagis.gl/refserver/rest/services/TekniskGrundkort/TekniskGrundkort/MapServer/16','service_description':'https://nunagis.gl/en/web-services/'},hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW/'asiaq_layer43.geojson',RAW/'asiaq_layer16.geojson']})
    if (RAW/'official_residential_placenames.geojson').exists():
        summary['sources']['official_residential_placenames']='https://kort.nunagis.gl/refserver/rest/services/PlacenamesRegister/PlacenamesRegisterPublic/MapServer/0'
        summary['hashes']['official_residential_placenames.geojson']=hashlib.sha256((RAW/'official_residential_placenames.geojson').read_bytes()).hexdigest()
    (OUT/'格陵兰官方实体整理记录.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

"""Prepare static coastline/bathymetry only. Never read or write TOPAZ ice files."""
from pathlib import Path
import os,sys,json,csv,time,hashlib,zipfile,io,argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import requests,shapefile,h5netcdf,rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.warp import transform_geom,reproject,Resampling
from rasterio.features import rasterize
from pyproj import CRS,Transformer

ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent
OUT=ROOT/'高分辨率静态基准_2026-11-02';RAW=OUT/'原始数据';TILES=RAW/'GEBCO_2026_15arcsec_50N_tiles'
URL='https://dap.ceda.ac.uk/thredds/dodsC/bodc/gebco/global/gebco_2026/ice_surface_elevation/netcdf/GEBCO_2026.nc'
PROJ='+proj=stere +lon_0=-45 +lat_0=90 +k=1 +R=6378273 +no_defs'
CRS_OUT=CRS.from_proj4(PROJ)
SHAPE=(11014,7614); TRANSFORM=from_origin(-3807000,5507000,1000,1000)

def status(**kw):
 p=OUT/'运行状态.json';d=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
 d.update(kw,updated_local=time.strftime('%Y-%m-%d %H:%M:%S'));p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')

def fetch(a,b):
 p=TILES/f'rows_{a}_{b}.npy'
 if p.exists():
  q=np.load(p,mmap_mode='r')
  if q.shape==(b-a,86400) and q.dtype==np.int16:return a,b,p,'cached'
  raise RuntimeError('Unexpected cached source tile: '+str(p))
 u=f'{URL}.dods?elevation%5B{a}:1:{b-1}%5D%5B0:1:86399%5D'
 err=None
 for attempt in range(4):
  try:
   r=requests.get(u,timeout=(30,180));r.raise_for_status()
   if b'Data:\n' not in r.content:raise RuntimeError('Missing DODS data separator')
   binary=r.content.split(b'Data:\n',1)[1];n=(b-a)*86400
   if np.frombuffer(binary[:8],dtype='>u4').tolist()!=[n,n]:raise RuntimeError('DODS array length mismatch')
   q=np.frombuffer(binary,offset=8,count=n,dtype='>i4')
   if q.min()<-32768 or q.max()>32767:raise RuntimeError('Elevation outside Int16 source range')
   tmp=p.with_suffix('.part.npy');np.save(tmp,q.astype('int16').reshape(b-a,86400));os.replace(tmp,p)
   return a,b,p,'downloaded'
  except Exception as e:
   err=e
   if attempt<3:time.sleep(2**attempt)
 raise RuntimeError(f'{a}:{b}: {err}')

def download(workers):
 bands=[(a,min(a+120,43200)) for a in range(33600,43200,120)]
 status(stage='download_GEBCO_native',expected_tiles=len(bands),completed_tiles=0,TOPAZ_modified=False)
 n=0;errors=[]
 with ThreadPoolExecutor(workers) as pool:
  futures={pool.submit(fetch,a,b):(a,b) for a,b in bands}
  for f in as_completed(futures):
   try:
    a,b,p,kind=f.result();n+=1;print(f'GEBCO {n}/{len(bands)} {a}:{b} {kind}',flush=True)
    status(completed_tiles=n,failed_tiles=errors)
   except Exception as e:errors.append(str(e));status(failed_tiles=errors)
 if errors:raise RuntimeError('Download incomplete; rerun resumes. '+str(errors))

def source_tiff():
 p=RAW/'GEBCO_2026_50N_native15arcsec.tif'
 if p.exists():return p
 tmp=p.with_suffix('.part.tif')
 profile=dict(driver='GTiff',width=86400,height=9600,count=1,dtype='int16',
  crs='EPSG:4326',transform=from_origin(-180,90,1/240,1/240),nodata=-32768,
  tiled=True,blockxsize=512,blockysize=128,compress='deflate',predictor=2,BIGTIFF='IF_SAFER')
 with rasterio.open(tmp,'w',**profile) as ds:
  for a in range(33600,43200,120):
   b=min(a+120,43200);q=np.load(TILES/f'rows_{a}_{b}.npy',mmap_mode='r')
   ds.write(np.asarray(q[::-1]),1,window=Window(0,43200-b,86400,b-a))
 os.replace(tmp,p);return p

def geoms():
 p=PROJECT/'人类聚居点/原始数据/GSHHG_2.3.7_high_L1'
 r=shapefile.Reader(shp=str(p/'GSHHS_h_L1.shp'),shx=str(p/'GSHHS_h_L1.shx'))
 land=[];tf=Transformer.from_crs('EPSG:4326',CRS_OUT,always_xy=True)
 for s in r.iterShapes():
  if s.bbox[3]<50:continue
  # GSHHG L1 outer polygons are land footprints; lakes remain non-marine.
  cuts=list(s.parts)+[len(s.points)]
  for a,b in zip(cuts[:-1],cuts[1:]):
   ll=np.asarray(s.points[a:b]);xx,yy=tf.transform(ll[:,0],ll[:,1])
   land.append({'type':'Polygon','coordinates':[np.column_stack((xx,yy)).tolist()]})
 z=zipfile.ZipFile(PROJECT/'人类聚居点/原始数据/AMAP_area.zip')
 base=next(n[:-4] for n in z.namelist() if n.endswith('amaplim_lam_poly.shp'))
 rr=shapefile.Reader(shp=io.BytesIO(z.read(base+'.shp')),shx=io.BytesIO(z.read(base+'.shx')),dbf=io.BytesIO(z.read(base+'.dbf')))
 src=CRS.from_wkt(z.read(base+'.prj').decode())
 amap=[transform_geom(src,CRS_OUT,s.__geo_interface__) for s in rr.shapes()]
 return land,amap

def geometry():
 land,amap=geoms()
 base=dict(driver='GTiff',height=SHAPE[0],width=SHAPE[1],count=1,crs=CRS_OUT,
  transform=TRANSFORM,tiled=True,blockxsize=512,blockysize=512,compress='deflate',BIGTIFF='IF_SAFER')
 landmask=rasterize(((g,1) for g in land),out_shape=SHAPE,transform=TRANSFORM,
  fill=0,all_touched=True,dtype='uint8')
 area=rasterize(((g,1) for g in amap),out_shape=SHAPE,transform=TRANSFORM,
  fill=0,all_touched=False,dtype='uint8')
 for name,a in [('GSHHG_land_alltouched_1km.tif',landmask),('AMAP_center_mask_1km.tif',area)]:
  with rasterio.open(OUT/name,'w',dtype='uint8',**base) as ds:
   ds.write(a,1);ds.update_tags(role='static_geometry_not_navigability',TOPAZ_processed='false')
 print('GEOMETRY_COMPLETE',flush=True)
 return landmask,area,base

def build():
 status(stage='build_static_1km')
 p=source_tiff()
 if (OUT/'GSHHG_land_alltouched_1km.tif').exists() and (OUT/'AMAP_center_mask_1km.tif').exists():
  with rasterio.open(OUT/'GSHHG_land_alltouched_1km.tif') as d:landmask=d.read(1)
  with rasterio.open(OUT/'AMAP_center_mask_1km.tif') as d:area=d.read(1)
  base=dict(driver='GTiff',height=SHAPE[0],width=SHAPE[1],count=1,crs=CRS_OUT,
   transform=TRANSFORM,tiled=True,blockxsize=512,blockysize=512,compress='deflate',BIGTIFF='IF_SAFER')
 else:landmask,area,base=geometry()
 elev=OUT/'GEBCO_2026_elevation_1km.tif'
 with rasterio.open(p) as src,rasterio.open(elev,'w',dtype='float32',nodata=-32768,predictor=3,**base) as dst:
  reproject(source=rasterio.band(src,1),destination=rasterio.band(dst,1),
   src_transform=src.transform,src_crs=src.crs,src_nodata=src.nodata,
   dst_transform=TRANSFORM,dst_crs=CRS_OUT,dst_nodata=-32768,
   resampling=Resampling.average,num_threads=4,warp_mem_limit=256)
  dst.update_tags(source_native_spacing='15 arcseconds',resampling='area-weighted average',
   sign='negative=below reference sea level; positive=above',TOPAZ_processed='false')
 depth=OUT/'GEBCO_2026_ocean_depth_1km.tif'
 counts=Counter_local=dict(amap_cells=int(area.sum()),land_cells=int(landmask.sum()))
 conflict=0;marine=0
 with rasterio.open(elev) as src,rasterio.open(depth,'w',dtype='float32',nodata=-32768,predictor=3,**base) as dst:
  for _,window in src.block_windows(1):
   q=src.read(1,window=window);r0,c0=int(window.row_off),int(window.col_off);h,w=q.shape
   ll=landmask[r0:r0+h,c0:c0+w];ar=area[r0:r0+h,c0:c0+w]
   ok=(ll==0)&(q<0)&(q!=-32768)
   d=np.where(ok,-q,-32768).astype('float32');dst.write(d,1,window=window)
   conflict+=int(((ll==0)&(ar==1)&(q>=0)).sum());marine+=int((ok&(ar==1)).sum())
  dst.update_tags(land_source='GSHHG 2.3.7 high L1 all_touched',
   meaning='positive depth only where GSHHG non-land and GEBCO elevation negative; elsewhere nodata',
   minimum_draft_threshold='none',TOPAZ_processed='false')
 counts.update(amap_GSHHG_nonland_GEBCO_nonnegative_cells=conflict,amap_positive_depth_cells=marine)
 # Point representation diagnostic on static 1-km geometry only: no entrance assignment.
 points=list(csv.DictReader((PROJECT/'人类聚居点/港口与沿海聚居点_规则结果/保留点位_港口与沿海.csv').open(encoding='utf-8-sig')))
 toxy=Transformer.from_crs(4326,CRS_OUT,always_xy=True)
 with rasterio.open(elev) as ds:
  for r in points:
   xx,yy=toxy.transform(float(r['longitude']),float(r['latitude']));j,i=rasterio.transform.rowcol(TRANSFORM,xx,yy)
   ins=0<=j<SHAPE[0] and 0<=i<SHAPE[1]
   r.update(static_1km_row=j,static_1km_col=i,static_1km_land_alltouched=int(landmask[j,i]) if ins else '',
    static_1km_AMAP_center=int(area[j,i]) if ins else '',
    static_1km_GEBCO_elevation_m=float(next(ds.sample([(xx,yy)]))[0]) if ins else '',
    marine_entrance_assignment='not_assigned')
 with (OUT/'4258点_高分辨率静态位置.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
 counts['points_on_1km_alltouched_land']=sum(r['static_1km_land_alltouched']==1 for r in points)
 counts['points_outside_center_AMAP_mask']=sum(r['static_1km_AMAP_center']==0 for r in points)
 manifest=dict(created_local=time.strftime('%Y-%m-%d %H:%M:%S'),TOPAZ_processed=False,
   source_elevation_url=URL,source_native_grid='15 arcseconds, all native cells retained north of 50N',
   source_GSHHG='existing GSHHG 2.3.7 high L1',source_AMAP='existing official AMAP shapefile',
   working_grid_spacing_m=1000,working_grid_provisional=True,working_grid_shape=SHAPE,
   projection=PROJ,grid_bounds=[-3807000,-5507000,3807000,5507000],
   elevation_resampling='average',land_rasterization='all_touched=True',AMAP_rasterization='cell_center',
   no_depth_threshold=True,final_passability_mask_created=False,counts=counts,
   limitations=['1km is an adjustable working spacing, not approved final resolution.',
    'GSHHG all_touched can close channels narrower than a grid cell; source vector retained.',
    'GEBCO grid spacing does not imply uniform survey accuracy.',
    'No ice assigned to newly represented water; no extrapolation of TOPAZ.',
    'GSHHG non-land versus nonnegative GEBCO elevations remain unresolved, not automatically forced to ocean.'])
 (OUT/'数据来源与处理记录.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 status(stage='complete',outputs=[p.name for p in OUT.glob('*.tif')],counts=counts)
 print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['all','download','build','geometry'],default='geometry');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if a.action!='geometry':
  p.error('Water-depth work is outside the current user scope. Only --action geometry is enabled.')
 TILES.mkdir(parents=True,exist_ok=True)
 try:
  if a.action in ['all','download']:download(a.workers)
  if a.action in ['all','build']:build()
  if a.action=='geometry':geometry()
 except Exception as e:status(stage='failed_resumable',error=str(e));raise

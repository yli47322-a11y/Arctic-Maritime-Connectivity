"""Single-day final fields in the established circular polar point-map layout."""
from pathlib import Path
import csv,json,zipfile,io
import numpy as np,h5netcdf,rasterio,shapefile
from pyproj import CRS,Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Patch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from rasterio.features import rasterize
from affine import Affine

P=Path(__file__).resolve().parent;ROOT=P.parents[1]
CURRENT=P/'AMAP外70N以南无冰_无10km缓冲';OUT=CURRENT/'figures/2025-01-01_最终结果_点位图式';OUT.mkdir(exist_ok=True)
BASE=ROOT/'正式成果/空间基准_5km_最终正式版_v3_全量最近出海_2026-11-04'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
allpoints=read(ROOT/'人类聚居点/港口与沿海聚居点_规则结果/全部点位_分类及去留.csv')
coastal=read(P/'spatial/聚居点_4258条.csv')
noncoastal=[r for r in allpoints if r['category'] in ['非沿海点','内陆港口_超10km']]
deps=read(P/'spatial/出发格.csv')
with rasterio.open(BASE/'03_正式5km栅格/land_mask_5km.tif') as d:land=d.read(1)==1;tf=d.transform;crs=d.crs
ncpath=CURRENT/'processed/PIOMAS_2025-01_逐日_5km_AMAP外70N以南无冰.nc'
with h5netcdf.File(ncpath) as d:
 C=d['siconc'][0]*100;H=d['sithick_effective'][0];domain=d['marine_domain_mask'][:]==1
tr=Transformer.from_crs(4326,crs,always_xy=True)
radius=abs(tr.transform(-45,50)[1]);extent=[tf.c,tf.c+land.shape[1]*5000,tf.f-land.shape[0]*5000,tf.f]
# Extend only the display background with the same GSHHG source and grid spacing.
# The original computational land cells are copied back exactly, without editing.
left=max(0,int(np.ceil((radius+tf.c)/5000)))
right=max(0,int(np.ceil((radius-extent[1])/5000)))
dtf=Affine(5000,0,tf.c-left*5000,0,-5000,tf.f)
display_shape=(land.shape[0],land.shape[1]+left+right)
geoms=[]
rd=shapefile.Reader(str(ROOT/'人类聚居点/原始数据/GSHHG_2.3.7_high_L1/GSHHS_h_L1.shp'))
for sh in rd.iterShapes():
 if sh.bbox[3]<49:continue
 cuts=list(sh.parts)+[len(sh.points)];rings=[]
 for start,end in zip(cuts[:-1],cuts[1:]):
  xy=np.asarray(sh.points[start:end]);gx,gy=tr.transform(xy[:,0],xy[:,1]);rings.append(np.c_[gx,gy].tolist())
 geoms.append(({'type':'Polygon','coordinates':rings},1))
display_land=rasterize(geoms,out_shape=display_shape,transform=dtf,fill=0,all_touched=True,dtype='uint8')
display_land[:,left:left+land.shape[1]]=land.astype('uint8')
display_extent=[dtf.c,dtf.c+display_shape[1]*5000,dtf.f-display_shape[0]*5000,dtf.f]
coast_x=dtf.c+(np.arange(display_shape[1])+.5)*5000
coast_y=dtf.f-(np.arange(display_shape[0])+.5)*5000
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],'font.family':'sans-serif','font.size':6.5,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
boundary=[]
with zipfile.ZipFile(ROOT/'人类聚居点/原始数据/AMAP_area.zip') as z:
 b='amaplim_lam_poly';rd=shapefile.Reader(shp=io.BytesIO(z.read(b+'.shp')),shx=io.BytesIO(z.read(b+'.shx')))
 t=Transformer.from_crs(CRS.from_wkt(z.read(b+'.prj').decode()),crs,always_xy=True)
 for s in rd.iterShapes():
  cut=list(s.parts)+[len(s.points)]
  for a,b in zip(cut[:-1],cut[1:]):
   xy=np.asarray(s.points[a:b]);x,y=t.transform(xy[:,0],xy[:,1]);boundary.append((x,y))
national_borders=[]
with zipfile.ZipFile(ROOT/'人类聚居点/原始数据/ne_10m_admin_0_boundary_lines_land.zip') as z:
 b=next(n[:-4] for n in z.namelist() if n.endswith('.shp'))
 rd=shapefile.Reader(shp=io.BytesIO(z.read(b+'.shp')),shx=io.BytesIO(z.read(b+'.shx')))
 for sh in rd.iterShapes():
  if sh.bbox[3]<49:continue
  cuts=list(sh.parts)+[len(sh.points)]
  for a,b in zip(cuts[:-1],cuts[1:]):
   xy=np.asarray(sh.points[a:b]);gx,gy=tr.transform(xy[:,0],xy[:,1]);national_borders.append((gx,gy))
positions={}
for name,points in [('coastal',coastal),('noncoastal',noncoastal),('departure',deps)]:positions[name]=tr.transform([float(r['longitude']) for r in points],[float(r['latitude']) for r in points])

def panel(fig,ax,data,title,vmax,ticks,label):
 ax.set_aspect('equal');ax.set_xlim(-radius*1.055,radius*1.055);ax.set_ylim(-radius*1.055,radius*1.055);ax.axis('off')
 disc=Circle((0,0),radius,fc=plt.get_cmap('Blues')(0.),ec='#7f888e',lw=.5,zorder=0);ax.add_patch(disc)
 cmap=plt.get_cmap('Blues').copy();cmap.set_bad((0,0,0,0))
 img=ax.imshow(np.ma.masked_where(~domain|land,data),extent=extent,origin='upper',cmap=cmap,norm=Normalize(0,vmax),interpolation='nearest',zorder=1);img.set_clip_path(disc)
 rgba=np.zeros((*display_shape,4),dtype='float32');rgba[display_land==1]=(.865,.875,.855,1)
 im=ax.imshow(rgba,extent=display_extent,origin='upper',interpolation='nearest',zorder=2);im.set_clip_path(disc)
 coastline=ax.contour(coast_x,coast_y,display_land,levels=[.5],colors='#59666e',linewidths=.25,zorder=3.5)
 coastline.set_clip_path(disc)
 for la in [60,70,80]:
  ll=np.linspace(-180,180,721);x,y=tr.transform(ll,np.full_like(ll,la));line,=ax.plot(x,y,c='#b9c3c8',lw=.23,alpha=.8,zorder=3);line.set_clip_path(disc)
 for lo in np.arange(-180,180,45):
  ll=np.linspace(50,89.7,220);x,y=tr.transform(np.full_like(ll,lo),ll);line,=ax.plot(x,y,c='#bdc8ce',lw=.23,alpha=.8,zorder=3);line.set_clip_path(disc)
 for x,y in boundary:
  line,=ax.plot(x,y,c='#565e62',lw=.45,ls=(0,(3,2)),zorder=4);line.set_clip_path(disc)
 for x,y in national_borders:
  line,=ax.plot(x,y,c='#788186',lw=.34,zorder=3.8);line.set_clip_path(disc)
 ax.text(-radius*.955,0,'50°N',ha='left',va='bottom',fontsize=4.5,color='#333',zorder=8)
 for lo,lab in [(-180,'180°'),(-90,'90°W'),(0,'0°'),(90,'90°E')]:
  x,y=tr.transform(lo,48.5)
  rotation=(np.degrees(np.arctan2(y,x))+180)%180-90
  ax.text(x,y,lab,ha='center',va='center',fontsize=5.1,rotation=rotation,rotation_mode='anchor',zorder=8)
 ax.scatter(*positions['noncoastal'],s=.7,c='#8b9398',alpha=.35,lw=0,zorder=5,clip_path=disc)
 ax.scatter(*positions['coastal'],s=.9,c='#a64f2c',alpha=.86,lw=0,zorder=6,clip_path=disc)
 ax.scatter(*positions['departure'],s=3.4,facecolors='none',edgecolors='#5c214c',linewidths=.28,zorder=7,clip_path=disc)
 # Positions follow the user's marked cartographic placement, without moving data.
 for name,rx,ry,rot in [('USA\n(Alaska)',-.90,.21,0),('Canada',-.80,-.36,0),('Greenland (DK)',.01,-.43,90),('Russia',.35,.62,0),('Iceland',.30,-.69,0)]:
  ax.text(rx*radius,ry*radius,name,ha='center',va='center',fontsize=4.5,rotation=rot,zorder=9,color='#252b2e')
 # Place Norway geographically inside mainland Norway, along its long axis.
 x,y=tr.transform(9.0,61.7)
 ax.text(x,y,'Norway',ha='center',va='center',fontsize=3.8,rotation=-55,zorder=9,color='#252b2e')
 # Additional labels only for jurisdictions represented in the settlement registry.
 for name,lo,la,rotation,size in [('Sweden',16,63,-55,3.8),('Finland',26,65.5,-55,3.8),('Faroe Is. (DK)',-10,60.5,0,3.6),('Svalbard (NO)',22,81.2,0,3.6)]:
  x,y=tr.transform(lo,la)
  color='white' if name.startswith('Svalbard') and vmax==100 else '#252b2e'
  ax.text(x,y,name,ha='center',va='center',fontsize=size,rotation=rotation,zorder=9,color=color)
 ax.set_title(title,loc='left',fontsize=7.2,pad=5)
 cb=fig.colorbar(img,ax=ax,orientation='horizontal',fraction=.032,pad=.045,shrink=.73,ticks=ticks,extend='max' if vmax==100 else 'neither')
 cb.set_label(label,fontsize=6.2,labelpad=2);cb.ax.tick_params(labelsize=5.5,length=2,width=.4);cb.outline.set_linewidth(.4)

fig,axes=plt.subplots(1,2,figsize=(183/25.4,116/25.4))
panel(fig,axes[0],C,'a  Sea-ice concentration',100,[0,25,50,75,100],'海冰浓度（%）')
panel(fig,axes[1],H,'b  Effective sea-ice thickness',5,[0,1,2,3,4,5],'有效冰厚（m）')
fig.suptitle('PIOMAS最终处理结果｜2025年1月1日',fontsize=9,y=.98)
handles=[Line2D([],[],marker='o',ls='none',color='#a64f2c',ms=2.5,label=f'沿海聚居点（{len(coastal):,}）'),Line2D([],[],marker='o',ls='none',color='#a6adb1',ms=2.5,label=f'非沿海聚居点（{len(noncoastal):,}，淡化）'),Line2D([],[],marker='o',ls='none',markerfacecolor='none',markeredgecolor='#5c214c',ms=3,label=f'出发格（{len(deps):,}）'),Line2D([],[],color='#565e62',lw=.6,ls=(0,(3,2)),label='AMAP边界')]
handles.append(Line2D([],[],color='#788186',lw=.6,label='国界'))
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,fontsize=6,columnspacing=1.7,handlelength=1.7)
fig.text(.5,.019,'正式5公里掩膜；按既定规则设无冰；计算网格外的陆海仅为展示背景。',ha='center',fontsize=5.8)
fig.subplots_adjust(left=.025,right=.975,top=.88,bottom=.21,wspace=.09)
for ext in ['jpg','png','pdf','svg']:
 fig.savefig(OUT/f'PIOMAS_2025-01-01_最终冰情与聚居点.{ext}',dpi=600)
fig.savefig(OUT/'预览.png',dpi=180);plt.close(fig)
caption='''# PIOMAS单日最终冰情与聚居点

2025年1月1日的最终5公里海冰浓度（a）和PIOMAS有效冰厚（b）。采用原点位分布图的圆形北极投影版式，中央经线45°W，展示50°N以北；数据和正式5公里陆地掩膜直接按原网格绘制，不再插值或平滑。计算网格外的左右陆地背景用同一GSHHG 2.3.7 high L1源按5公里补齐，仅用于展示；计算网格内部陆地掩膜逐格保留不变。加深海岸线轮廓，网格外海洋补以与无冰海洋一致的底色，仅为展示背景，不补造冰情或扩大计算域。

沿海类含港口聚居点，共4258条；非沿海类含超10公里的内陆港口聚居点，共4376条，以浅灰淡化；排除5条已识别非聚居实体。空心紫圈为1248个固定出发格，深灰虚线为原AMAP边界。符号无人口权重，位置不抖动。非沿海点只作空间背景，不作为网络节点。

AMAP外≤70°N海格、残余缺值海格及所有出发格的浓度和有效冰厚按既定规则为0，属于处理假设而非原始模型确认无冰。本图不是POLARIS风险或通航性图。浓度色标主体0—100%，超过100%的源值通过上端扩展表示，原文件数值不改动；有效冰厚色标0—5m。有效冰厚保持体积/总格面积定义。

国界为Natural Earth 1:10m Admin 0 land boundary lines，仅作制图参考，不改变正式海陆掩膜、计算域或点位。来源：https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-boundary-lines/ 。

地理标注仅对应聚居点清单有记录的国家/地区：US、CA、RU、NO、SE、FI、IS、GL、FO、SJ。美国显示USA (Alaska)，格陵兰和法罗群岛以DK注明丹麦，斯瓦尔巴以NO注明挪威；不标注其他背景国家。Norway位于挪威大陆国界内，Alaska保留此前左移位置。

本次仅绘图，没有更新数据、点位、掩膜或执行额外可行性实验。
'''
(OUT/'图注与说明.md').write_text(caption,encoding='utf-8')
(OUT/'导出记录.json').write_text(json.dumps(dict(date='2025-01-01',coastal_records=len(coastal),noncoastal_records=len(noncoastal),excluded_nonsettlement_entities=5,departure_cells=len(deps),land_mask='formal_5km_unchanged',projection=crs.to_string(),display_north_of=50,source_file=str(ncpath),concentration_display_range_percent=[0,100],concentration_over_range_extension=True,effective_thickness_range_m=[0,5],data_modified=False,raster_dpi=600),ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT)

from pathlib import Path
import csv,io,zipfile,json
import numpy as np
import shapefile
from pyproj import CRS,Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch,Circle
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent.parent/'人类聚居点'
PROJECT=ROOT.parent
OUT=PROJECT/'正式成果/港口100km陆地连通_新增港口最小覆盖/图'
OUT.mkdir(exist_ok=True)
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=read(PROJECT/'正式成果/港口100km陆地连通支撑的潜在联系机会增量/4258聚居地_港口100km覆盖与最近港口.csv')
new=read(PROJECT/'正式成果/港口100km陆地连通_新增港口最小覆盖/新增港口_最小覆盖选址.csv')
# Contract: one polar map showing documented and proposed port sites relative to all coastal settlements.
# Reuse the established Python map workflow, complete ocean disc, and official AMAP geometry.
plt.rcParams.update({'font.family':'Times New Roman','font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.6,'savefig.facecolor':'white','legend.frameon':False})
crs=CRS.from_proj4('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +datum=WGS84 +units=m +no_defs');tr=Transformer.from_crs(4326,crs,always_xy=True)
radius=abs(tr.transform(-45,50)[1]);fig=plt.figure(figsize=(183/25.4,188/25.4));ax=fig.add_axes([.04,.09,.92,.88]);ax.set_aspect('equal');ax.set_xlim(-radius*1.045,radius*1.045);ax.set_ylim(-radius*1.045,radius*1.045);ax.set_axis_off()
disc=Circle((0,0),radius,facecolor='#EDF4F7',edgecolor='#81939D',lw=.55,zorder=0);ax.add_patch(disc)

with zipfile.ZipFile(ROOT/'原始数据/ne_10m_land.zip') as z:
    base=next(n[:-4] for n in z.namelist() if n.endswith('.shp'))
    rd=shapefile.Reader(shp=io.BytesIO(z.read(base+'.shp')),shx=io.BytesIO(z.read(base+'.shx')))
    for shape in rd.iterShapes():
        if shape.bbox[3]<49:continue
        verts=[];codes=[];cuts=list(shape.parts)+[len(shape.points)]
        for a,b in zip(cuts[:-1],cuts[1:]):
            pts=np.asarray(shape.points[a:b]);x,y=tr.transform(pts[:,0],pts[:,1]);v=np.column_stack([x,y]);verts.extend(v);codes.extend([MPath.MOVETO]+[MPath.LINETO]*(len(v)-2)+[MPath.CLOSEPOLY])
        patch=PathPatch(MPath(np.asarray(verts),np.asarray(codes)),facecolor='#E9E7DE',edgecolor='#A3AEAA',lw=.26,zorder=1)
        patch.set_clip_path(disc);ax.add_patch(patch)

with zipfile.ZipFile(ROOT/'原始数据/ne_10m_admin_0_boundary_lines_land.zip') as z:
    base=next(n[:-4] for n in z.namelist() if n.endswith('.shp'))
    rd=shapefile.Reader(shp=io.BytesIO(z.read(base+'.shp')),shx=io.BytesIO(z.read(base+'.shx')))
    for shape in rd.iterShapes():
        if shape.bbox[3]<49:continue
        cuts=list(shape.parts)+[len(shape.points)]
        for a,b in zip(cuts[:-1],cuts[1:]):
            v=np.asarray(shape.points[a:b]);xx,yy=tr.transform(v[:,0],v[:,1])
            line,=ax.plot(xx,yy,color='#82959E',lw=.55,zorder=2.1)
            line.set_clip_path(disc)

for lat in [60,70,80]:
    ll=np.linspace(-180,180,721);xx,yy=tr.transform(ll,np.full_like(ll,lat));line,=ax.plot(xx,yy,c='#A8BAC5',alpha=.42,lw=.27,zorder=2);line.set_clip_path(disc)
for lon in np.arange(-180,180,45):
    ll=np.linspace(50,89.5,200);xx,yy=tr.transform(np.full_like(ll,lon),ll);line,=ax.plot(xx,yy,c='#A8BAC5',alpha=.36,lw=.24,zorder=2);line.set_clip_path(disc)
x,y=tr.transform(-145,51.4)
ax.text(x,y,'50°N',fontsize=5.5,ha='center',va='center',color='#526873',zorder=6)
for lon,label in [(-180,'180°'),(-135,'135°W'),(-90,'90°W'),(-45,'45°W'),(0,'0°'),(45,'45°E'),(90,'90°E'),(135,'135°E')]:
    x,y=tr.transform(lon,48.9)
    tangent=(np.degrees(np.arctan2(y,x))+180)%180-90
    ax.text(x,y,label,ha='center',va='center',fontsize=5.8,color='#425862',
            rotation=tangent,rotation_mode='anchor',zorder=7)

with zipfile.ZipFile(ROOT/'原始数据/AMAP_area.zip') as z:
    base='amaplim_lam_poly';rd=shapefile.Reader(shp=io.BytesIO(z.read(base+'.shp')),shx=io.BytesIO(z.read(base+'.shx')));bt=Transformer.from_crs(CRS.from_wkt(z.read(base+'.prj').decode()),crs,always_xy=True)
    for shape in rd.iterShapes():
        cuts=list(shape.parts)+[len(shape.points)]
        for a,b in zip(cuts[:-1],cuts[1:]):
            v=np.asarray(shape.points[a:b]);x,y=bt.transform(v[:,0],v[:,1]);line,=ax.plot(x,y,c='#5B6266',lw=.55,ls=(0,(3,2)),zorder=3);line.set_clip_path(disc)


port=np.array([r['original_category']=='港口聚居点' for r in rows])
x,y=tr.transform([float(r['longitude']) for r in rows],[float(r['latitude']) for r in rows]);x=np.asarray(x);y=np.asarray(y)
ax.scatter(x,y,s=3.0,c='#7795A4',alpha=.60,lw=0,zorder=4,clip_path=disc)
ax.scatter(x[port],y[port],s=16,c='#246D86',marker='^',edgecolors='white',lw=.30,zorder=5,clip_path=disc)
nx,ny=tr.transform([float(r['longitude']) for r in new],[float(r['latitude']) for r in new])
ax.scatter(nx,ny,s=16,c='#D96643',marker='D',edgecolors='white',lw=.30,zorder=6,clip_path=disc)
for name,lon,lat in [('Alaska',-151,63),('Canada',-105,61),('Greenland',-40,74),('Russia',98,62),('Norway',19,64),('Iceland',-19,62)]:
    xx,yy=tr.transform(lon,lat);ax.text(xx,yy,name,ha='center',va='center',fontsize=6.3,color='#344D59',zorder=7)
xx,yy=tr.transform(110,84);ax.text(xx,yy,'Arctic Ocean',fontsize=6.6,fontstyle='italic',color='#526F80',ha='center',va='center',zorder=7)
handles=[Line2D([],[],marker='o',color='none',markerfacecolor='#7795A4',markeredgewidth=0,markersize=3,label='Coastal settlements (4,258)'),
 Line2D([],[],marker='^',color='none',markerfacecolor='#246D86',markeredgecolor='white',markeredgewidth=.3,markersize=5,label='Existing ports (272)'),
 Line2D([],[],marker='D',color='none',markerfacecolor='#D96643',markeredgecolor='white',markeredgewidth=.3,markersize=4.5,label='Proposed ports (236)'),
 Line2D([],[],color='#5B6266',lw=.7,ls=(0,(3,2)),label='AMAP boundary')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.017),ncol=2,fontsize=7,columnspacing=2,handlelength=2)
for ext in ['png','jpg','pdf','svg']:
    fig.savefig(OUT/f'Arctic_ports_100km_land_connected.{ext}',dpi=450 if ext in ['png','jpg'] else 300)
fig.savefig(OUT/'预览.png',dpi=180)
plt.close(fig)
(OUT/'图注.md').write_text('原有港口272个（蓝色三角），新增港口236个（橙红菱形），沿海聚居地4258条（灰蓝圆点），虚线为官方AMAP边界。新增选址为100公里地表距离且1公里掩膜陆地连通约束下的一组最小覆盖布局。全部点位保持原始矢量坐标，无位移。底图Natural Earth 1:10m陆地与国界；完整海洋背景；北极立体投影，中央经线45°W，50°N裁切。原有与新增港口均属于沿海聚居地子集。',encoding='utf-8')
print(OUT)

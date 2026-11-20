from pathlib import Path
import csv, io, json, zipfile
import numpy as np
import shapefile
from pyproj import CRS, Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Circle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'人类聚居点'
OUT=PROJECT/'论文/方法章节/图件_研究区与海冰厚度'
OUT.mkdir(exist_ok=True)
BUILD=PROJECT/'论文/方法章节/.制作'
def read(p):
    return list(csv.DictReader(p.open(encoding='utf-8-sig')))
rows=read(ROOT/'港口与沿海聚居点_规则结果/保留点位_港口与沿海.csv')
excluded=[r for r in read(ROOT/'港口与沿海聚居点_规则结果/排除点位及原因.csv') if r['category']!='非聚居实体']
port=np.array([r['category']=='港口聚居点' for r in rows]);coastal=~port
source=(ROOT/'绘制点位分布与距离统计.py').read_text(encoding='utf-8')
start=source.index('plt.rcParams.update(')
stop=source.index('handles=[Line2D')
draw=source[start:stop]
draw=draw.replace('s=1.7,','s=.65,').replace('s=6,','s=2.0,').replace('s=9,','s=3.2,')
base=draw[:draw.index('ex,ey=')]
border_start=base.index("with zipfile.ZipFile(ROOT/'原始数据/ne_10m_admin_0_boundary_lines_land.zip')")
border_end=base.index('for lat in [60,70,80]:',border_start)
base=base[:border_start]+base[border_end:]
def map_panel(points):
    env=globals().copy()
    exec(draw if points else base,env)
    fig,ax=env['fig'],env['ax']
    fig.set_size_inches(95/25.4,108/25.4)
    ax.set_position([.04,.15,.92,.81])
    return env

left=map_panel(True)
fig,ax=left['fig'],left['ax']
handles=[Line2D([],[],marker='^',color='none',markerfacecolor='#D66A35',markeredgewidth=0,markersize=3,label='Port settlements'),
         Line2D([],[],marker='o',color='none',markerfacecolor='#287DA1',markeredgewidth=0,markersize=2.2,label='Other coastal settlements'),
         Line2D([],[],marker='o',color='none',markerfacecolor='#B9BFC2',markeredgewidth=0,markersize=2,label='Inland settlements'),
         Line2D([],[],color='#5B6266',lw=.6,ls=(0,(3,2)),label='AMAP boundary')]
legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.50,.045),ncol=2,fontsize=5.8,columnspacing=.9,handletextpad=.4,handlelength=1.5)
fig.text(.025,.975,'(a)',fontsize=9,va='top')

right=map_panel(False)
rf,ra=right['fig'],right['ax']
data=PROJECT/'PIOMAS_v2.1_1979-2026'
h=np.memmap(data/'decoded/hiday/hiday.H2025',dtype='<f4',mode='r',shape=(365,120,360))
mean=np.asarray(h,dtype='f8').mean(axis=0)
with np.load(PROJECT/'正式成果/R1_PIOMAS海冰退缩_1979-2025/数据/PIOMAS原生格_月度浓度趋势.npz') as z:
    lon=z['longitude'];lat=z['latitude'];mask=z['amap_ocean'].copy()
mean[~mask]=np.nan
x,y=right['tr'].transform(lon,lat)
def corners(a):
    p=np.pad(a,((1,1),(1,1)),mode='wrap');p[0]=2*p[1]-p[2];p[-1]=2*p[-2]-p[-3]
    return (p[:-1,:-1]+p[1:,:-1]+p[:-1,1:]+p[1:,1:])/4
cmap=LinearSegmentedColormap.from_list('RdBu_positive',plt.get_cmap('RdBu')(np.linspace(.5,1,256)))
cmap.set_bad((0,0,0,0))
mesh=ra.pcolormesh(corners(x),corners(y),mean,cmap=cmap,norm=Normalize(0,4),shading='flat',zorder=.5,rasterized=True)
mesh.set_clip_path(right['disc'])
ca=rf.add_axes([.19,.080,.62,.022])
cb=rf.colorbar(mesh,cax=ca,orientation='horizontal',ticks=[0,1,2,3,4],extend='max')
cb.outline.set_linewidth(.45);cb.ax.tick_params(labelsize=6,length=2,pad=2)
cb.set_label('Annual mean effective sea-ice thickness (m)',fontsize=6.3,labelpad=3)
rf.text(.025,.975,'(b)',fontsize=9,va='top')

def export_panel(env,name,leg=None,cbar=None):
    f=env['fig'];a=env['ax']
    for ext in ['png','svg','pdf']:
        f.savefig(OUT/f'{name}.{ext}',dpi=500,facecolor='white')
    f.canvas.draw();renderer=f.canvas.get_renderer();scale=95/f.bbox.width
    artists=[*a.texts,*f.texts]
    if leg:artists+=leg.get_texts()
    if cbar:artists += [*cbar.ax.get_xticklabels(),cbar.ax.xaxis.label]
    labels=[]
    for t in artists:
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        w=renderer.get_text_width_height_descent(t.get_text(),t.get_fontproperties(),False)[0]*scale
        labels.append(dict(text=t.get_text(),cx=(b.x0+b.width/2)*scale,cy=(f.bbox.height-b.y0-b.height/2)*scale,width=w+2,height=max(3,b.height*scale+.5),rotation=(360-t.get_rotation())%360,size=t.get_fontsize(),italic=t.get_fontstyle()=='italic',color=to_hex(t.get_color())[1:].upper()))
        t.set_visible(False)
    symbols=[]
    if leg:
        for artist,kind,col in zip(leg.legend_handles,['triangle','circle','circle','dash'],['D66A35','287DA1','B9BFC2','5B6266']):
            b=artist.get_window_extent(renderer)
            symbols.append(dict(cx=(b.x0+b.width/2)*scale,cy=(f.bbox.height-b.y0-b.height/2)*scale,kind=kind,color=col))
        leg.set_visible(False)
    colorbar=None
    if cbar:
        b=cbar.ax.get_window_extent(renderer)
        colorbar=dict(x=b.x0*scale,y=(f.bbox.height-b.y1)*scale,w=b.width*scale,h=b.height*scale,colors=[to_hex(cmap(v))[1:].upper() for v in np.linspace(0,1,128)])
        cbar.ax.set_visible(False)
    f.savefig(BUILD/f'{name}_background.png',dpi=500,facecolor='white')
    (BUILD/f'{name}_objects.json').write_text(json.dumps(dict(labels=labels,symbols=symbols,colorbar=colorbar),ensure_ascii=False),encoding='utf-8')
export_panel(left,'a_聚居点分布',leg=legend)
export_panel(right,'b_2025年平均有效冰厚',cbar=cb)
combined=plt.figure(figsize=(190/25.4,108/25.4),facecolor='white')
for i,name in enumerate(['a_聚居点分布','b_2025年平均有效冰厚']):
    axis=combined.add_axes([i*.5,0,.5,1]);axis.imshow(plt.imread(OUT/f'{name}.png'));axis.axis('off')
combined.savefig(OUT/'研究区与2025年海冰厚度.png',dpi=500)
np.savez_compressed(OUT/'2025年平均有效冰厚_原生网格.npz',longitude=lon,latitude=lat,effective_thickness_m=mean,AMAP_ocean=mask)
(OUT/'绘图数据说明.json').write_text(json.dumps(dict(year=2025,days=365,definition='hiday: volume per total grid-cell area',grid='native PIOMAS 120x360',map_source='project settlement-distribution drawing code',settlements=len(rows),ports=int(port.sum()),inland=len(excluded),native_mean_range_m=[float(np.nanmin(mean)),float(np.nanmax(mean))],palette='positive half of the existing RdBu concentration-trend palette',panels_width_mm=95),ensure_ascii=False,indent=2),encoding='utf-8')
print('Completed two redrawn panels; native 2025 effective thickness range',np.nanmin(mean),np.nanmax(mean),flush=True)

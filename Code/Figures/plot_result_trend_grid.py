from pathlib import Path
import ast,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap,BoundaryNorm

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'论文/结果图/3.1a_聚居地联系机会趋势';BUILD=OUT/'.制作';BUILD.mkdir(parents=True,exist_ok=True)
source=ROOT/'正式成果/R2_全区海上联系机会_完整分析/plot_r2.py'
node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='load_basemap')
scope={'ROOT':ROOT};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope);g=scope['load_basemap']()
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
deps=ROOT/'预处理/PIOMAS_5km_标准流程_v1/固定点位/出发格.csv';data=read(deps.with_name('聚居点与出发格对应.csv'));lookup={r['departure_id']:i for i,r in enumerate(read(deps))};idx=np.array([lookup[r['departure_id']] for r in data])
px,py=g['tr'].transform([float(r['settlement_longitude']) for r in data],[float(r['settlement_latitude']) for r in data])
with np.load(ROOT/'正式成果/R2_全区海上联系机会_完整分析/R2_完整分析结果.npz') as z:
 metric=z['node_slope'][:,idx].copy();ships=z['ships'].tolist()
breaks=[min(-6,int(np.floor(metric.min()))),-3,-1,0,1,2,3,5,7,10,15,max(27,int(np.ceil(metric.max())))]
metric[np.abs(metric)<1e-9]=0
colors=['#A3422F','#CE714A','#ECA87B','#C9E5EB','#9ACEDC','#6BB7CC','#429DBD','#277DA9','#245E90','#1F4370','#192D51'];cmap=ListedColormap(colors);norm=BoundaryNorm(breaks,11)
radius=g['radius'];labels=[]
plt.rcParams.update({'font.family':'Times New Roman','font.size':7,'svg.fonttype':'none','pdf.fonttype':42})
for i,ship in enumerate(ships[2:]):
 k=i+2;values=metric[k];fig=plt.figure(figsize=(84/25.4,84/25.4));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(-radius,radius),ylim=(-radius,radius),aspect='equal');ax.axis('off')
 disk=Circle((0,0),radius,fc='#F4F8FA',ec='#849199',lw=.45);ax.add_patch(disk)
 image=ax.imshow(g['display_land'],extent=g['display_extent'],cmap=ListedColormap(['#F4F8FA','#ECEDE8']),vmin=0,vmax=1,interpolation='nearest',zorder=1);image.set_clip_path(disk)
 co=ax.contour(g['coast_x'],g['coast_y'],g['display_land'],levels=[.5],colors='#7B878C',linewidths=.22,zorder=2);co.set_clip_path(disk)
 for key,col,lw,ls in [('boundary','#536570',.48,(0,(3,2))),('national_borders','#889197',.25,'solid')]:
  for x,y in g[key]:ax.plot(x,y,color=col,lw=lw,ls=ls,zorder=3,clip_path=disk)
 for lat in [60,70,80]:
  lo=np.linspace(-180,180,721);ax.plot(*g['tr'].transform(lo,np.full_like(lo,lat)),color='#A8BCC9',lw=.22,alpha=.65,zorder=2,clip_path=disk)
 for lon in np.arange(-180,180,45):
  la=np.linspace(50,90,120);ax.plot(*g['tr'].transform(np.full_like(la,lon),la),color='#A8BCC9',lw=.22,alpha=.65,zorder=2,clip_path=disk)
 zero=values==0;ax.scatter(np.asarray(px)[zero],np.asarray(py)[zero],s=1.8,c='#88949A',lw=0,zorder=4,clip_path=disk)
 order=np.argsort(abs(values),kind='stable');order=order[values[order]!=0]
 ax.scatter(np.asarray(px)[order],np.asarray(py)[order],c=values[order],s=2.3,lw=.07,edgecolors='#F7FAFA',cmap=cmap,norm=norm,zorder=5,clip_path=disk)
 fig.savefig(BUILD/f'{ship.replace(" ","_")}.png',dpi=500,transparent=True);plt.close(fig)
 print('Drawn',ship,flush=True)
for lon in np.arange(-180,180,45):
 # Labels on lateral edges sit inside the circle so they do not widen the gap.
 x,y=g['tr'].transform(lon,51.5)
 u,v=x/radius,-y/radius
 if abs(u)>.95:u*=.91;v*=.91
 label='180°' if lon==-180 else '0°' if lon==0 else f'{abs(lon)}°'+('W' if lon<0 else 'E')
 labels.append(dict(text=label,u=float(u),v=float(v)))
x,y=g['tr'].transform(-145,51.8);labels.append(dict(text='50°N',u=float(x/radius),v=float(-y/radius)))
(BUILD/'panels.json').write_text(json.dumps(dict(ships=ships[2:],colors=colors,breaks=breaks,labels=labels,main_map_diameter_mm=84,gutter_mm=2,main_map_total_width_mm=170,source_slides=list(range(83,93)),metric='OLS settlement opportunity trend, pp per decade, 1979–2025',records=len(data)),ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared ten-map two-column layout.',flush=True)

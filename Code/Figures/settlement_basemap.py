from pathlib import Path
import csv,json,io,zipfile,hashlib,sys
import numpy as np
import shapefile
from pyproj import CRS,Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch,Circle
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'港口与沿海聚居点_规则结果'/'保留点位_港口与沿海.csv'
OUT=ROOT/'图表_点位分布与距岸距离';OUT.mkdir(exist_ok=True)
def write(n,rows):
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with SRC.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
with (SRC.parent/'排除点位及原因.csv').open(encoding='utf-8-sig') as f:excluded=list(csv.DictReader(f))
d=np.array([float(r['coast_distance_km']) for r in rows]);port=np.array([r['category']=='港口聚居点' for r in rows]);coastal=~port
assert len(rows)==4258 and port.sum()==272 and coastal.sum()==3986 and len(excluded)==4381

# Figure contract: descriptive map of the rule-selected inventory; no trend/inference claim.
(OUT/'图表说明.md').write_text(f'''# 图表范围与导出

目的：展示统一10km距岸规则保留的4258条清单记录，区分{int(port.sum())}条港口与{int(coastal.sum())}条沿海聚居点；4381条排除记录以浅灰背景点展示。港口没有距离豁免。不表达历史变化、人口权重、航线或连通性。

距离分布表的“距离”指已有GSHHG距海岸距离，不是点对间距离。分组右闭：首组[0,1]km，其后(1,2]、(2,5]等。所有点和两类各自百分比均给出，不设人口或面积权重。

设计：北极立体投影，中央经线45°W，显示50°N以北；Nature风格单图、183mm宽、Arial 5—7pt、RGB色盲友好蓝/橙色、两种符号、白色画布、可编辑PDF/SVG和600dpi位图。点位置不抖动、不聚合，符号大小无人口含义。AMAP边界为参考线。

图上英文Port-supported与Coastal分别对应本次“港口聚居点”和“沿海聚居点”；Excluded为排除记录（包括48条超10km内陆港口与5条非聚居实体）。港口未获资料支撑不等于事实上无港口；统计单元仍可能存在尚未识别的重复和身份不确定性。空间聚集受资料覆盖影响，不能直接解释为真实聚居密度。

制图使用Natural Earth 1:10m陆地底图，AMAP官方边界。距岸统计沿用GSHHG 2.3.7 high L1处理结果，并非根据底图重新计算。边界与海岸底图精度不相同。

Nature图形规范：https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/
Natural Earth：https://www.naturalearthdata.com/
AMAP：https://www.amap.no/work-area/document/868
''',encoding='utf-8')

edges=np.array([0,1,2,5,10,25,50,100,250,500],float);table=[]
for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
    mask=(d>=lo if i==0 else d>lo)&(d<=hi)
    table.append({'距岸距离区间_km':f'[0,{hi:g}]' if i==0 else f'({lo:g},{hi:g}]','全部点数':int(mask.sum()),'全部占比_%':round(mask.mean()*100,3),'累计占比_%':round((d<=hi).mean()*100,3),'港口聚居点数':int((mask&port).sum()),'港口类占比_%':round((mask&port).sum()/port.sum()*100,3),'沿海聚居点数':int((mask&coastal).sum()),'沿海类占比_%':round((mask&coastal).sum()/coastal.sum()*100,3)})
assert sum(r['全部点数'] for r in table)==len(rows)
write('距岸距离分布表.csv',table)
summary=[]
for name,mask in [('全部',np.ones(len(rows),bool)),('港口聚居点',port),('沿海聚居点',coastal)]:
    a=d[mask];summary.append(dict(类别=name,点数=len(a),最小值_km=round(a.min(),6),P25_km=round(np.quantile(a,.25),6),中位数_km=round(np.median(a),6),均值_km=round(a.mean(),6),P75_km=round(np.quantile(a,.75),6),P95_km=round(np.quantile(a,.95),6),最大值_km=round(a.max(),6)))
write('距岸距离描述统计.csv',summary)
write('图表源数据_保留与排除点.csv',[{k:r[k] for k in ['entity_id','name','country_code','longitude','latitude','category','disposition','coast_distance_km','port_evidence','source_layer']} for r in rows+excluded])
all_d=np.array([float(r['coast_distance_km']) for r in rows+excluded]);all_edges=[0,1,2,5,10,25,50,100,250,500,1000,2500,np.inf];comparison=[]
for i,(lo,hi) in enumerate(zip(all_edges[:-1],all_edges[1:])):
    m=(all_d>=lo if i==0 else all_d>lo)&(all_d<=hi);k=int(m[:len(rows)].sum());x=int(m[len(rows):].sum())
    label=f'[0,{hi:g}]' if i==0 else f'>{lo:g}' if np.isinf(hi) else f'({lo:g},{hi:g}]'
    comparison.append(dict(距岸距离区间_km=label,全部点数=k+x,保留点数=k,排除点数=x,港口保留点数=int(((d>=lo if i==0 else d>lo)&(d<=hi)&port).sum()),沿海保留点数=int(((d>=lo if i==0 else d>lo)&(d<=hi)&coastal).sum())))
assert sum(r['全部点数'] for r in comparison)==len(rows)+len(excluded)
write('全部点位距岸分布_保留与排除对照.csv',comparison)

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

ex,ey=tr.transform([float(r['longitude']) for r in excluded],[float(r['latitude']) for r in excluded])
ax.scatter(ex,ey,s=1.7,c='#8C969B',alpha=.32,lw=0,zorder=3.1,clip_path=disc)
x,y=tr.transform([float(r['longitude']) for r in rows],[float(r['latitude']) for r in rows]);x=np.asarray(x);y=np.asarray(y)
ax.scatter(x[coastal],y[coastal],s=6,c='#287DA1',alpha=.78,lw=0,zorder=4,clip_path=disc)
ax.scatter(x[port],y[port],s=9,c='#D66A35',marker='^',edgecolors='white',lw=.25,zorder=5,clip_path=disc)
for name,lon,lat in [('Alaska',-151,63),('Canada',-105,61),('Greenland',-40,74),('Russia',98,62),('Norway',19,64),('Iceland',-19,62)]:
    xx,yy=tr.transform(lon,lat);ax.text(xx,yy,name,ha='center',va='center',fontsize=6.3,color='#344D59',zorder=7)
xx,yy=tr.transform(110,84);ax.text(xx,yy,'Arctic Ocean',fontsize=6.6,fontstyle='italic',color='#526F80',ha='center',va='center',zorder=7)
handles=[Line2D([],[],marker='^',color='none',markerfacecolor='#D66A35',markeredgecolor='white',markeredgewidth=.25,markersize=5,label=f'Port-supported (n = {port.sum():,})'),Line2D([],[],marker='o',color='none',markerfacecolor='#287DA1',markeredgewidth=0,markersize=3,label=f'Coastal (n = {coastal.sum():,})'),Line2D([],[],marker='o',color='none',markerfacecolor='#C1C5C8',markeredgewidth=0,markersize=3,label=f'Excluded (n = {len(excluded):,})'),Line2D([],[],color='#5B6266',lw=.7,ls=(0,(3,2)),label='AMAP boundary')]
legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.026),ncol=2,fontsize=6.8,columnspacing=2.2,handlelength=2)
fig.text(.5,.012,'Rule-selected inventory records; overlapping points are not displaced.',ha='center',fontsize=5.5)
for ext in ['pdf','svg','png','jpg','tiff']:
    kw={'dpi':600} if ext in ['png','jpg','tiff'] else {}
    if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
    fig.savefig(OUT/f'Arctic_settlement_distribution.{ext}',**kw)
fig.savefig(OUT/'分布图_预览.png',dpi=180)
if len(sys.argv) > 1:
    sys.path.append(sys.argv[1])
    from matplotlib.colors import to_hex
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Mm, Pt

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = 183 / fig.bbox.width
    labels = [(artist, artist.get_window_extent(renderer))
              for artist in [*ax.texts, *fig.texts, *legend.get_texts()]]
    symbols = [artist.get_window_extent(renderer) for artist in legend.legend_handles]
    for artist, _ in labels:
        artist.set_visible(False)
    legend.set_visible(False)
    background = io.BytesIO()
    fig.savefig(background, format='png', dpi=450)
    (OUT/'Arctic_settlement_distribution_background.png').write_bytes(background.getvalue())
    background.seek(0)
    deck = Presentation()
    deck.slide_width, deck.slide_height = Mm(183), Mm(188)
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.shapes.add_picture(background, 0, 0, width=deck.slide_width, height=deck.slide_height)
    for artist, bounds in labels:
        rotation = artist.get_rotation()
        cx = (bounds.x0 + bounds.width / 2) * scale
        cy = 188 - (bounds.y0 + bounds.height / 2) * scale
        # Unrotated text needs a horizontal box before PowerPoint applies rotation.
        width = renderer.get_text_width_height_descent(
            artist.get_text(), artist.get_fontproperties(), False)[0] * scale + 2
        shape = slide.shapes.add_textbox(Mm(cx-width/2), Mm(cy-2), Mm(width), Mm(4))
        shape.rotation = (360-rotation) % 360
        frame = shape.text_frame
        frame.word_wrap = False
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = artist.get_text()
        run.font.name = 'Times New Roman'
        run.font.size = Pt(artist.get_fontsize())
        run.font.italic = artist.get_fontstyle() == 'italic'
        run.font.color.rgb = RGBColor.from_string(to_hex(artist.get_color())[1:].upper())
    for bounds, kind, color, size in zip(
            symbols, [MSO_SHAPE.ISOSCELES_TRIANGLE, MSO_SHAPE.OVAL, MSO_SHAPE.OVAL, None],
            ['D66A35', '287DA1', 'C1C5C8', '5B6266'], [1.65, 1.05, 1.05, 4.5]):
        cx = (bounds.x0 + bounds.width / 2) * scale
        cy = 188 - (bounds.y0 + bounds.height / 2) * scale
        if kind is None:
            shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                Mm(cx-size/2), Mm(cy), Mm(cx+size/2), Mm(cy))
            shape.line.color.rgb = RGBColor.from_string(color)
            shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
            shape.line.width = Pt(.7)
        else:
            shape = slide.shapes.add_shape(kind, Mm(cx-size/2), Mm(cy-size/2), Mm(size), Mm(size))
            shape.fill.solid()
            shape.fill.fore_color.rgb = RGBColor.from_string(color)
            shape.line.fill.background()
    deck.save(OUT/'Arctic_settlement_distribution.pptx')
plt.close(fig)
(OUT/'图注.md').write_text(f'''**Fig. 1 | Spatial distribution of the rule-selected Arctic settlement inventory.** Retained records are shown as port-supported settlements (orange triangles; n = {int(port.sum()):,}) or coastal settlements without documented port support in the assembled sources (blue circles; n = {int(coastal.sum()):,}). Both categories require a representative-point distance of ≤10 km from the GSHHG high-resolution land–ocean boundary. Excluded records (n = 4,381), including 48 port-supported records beyond 10 km and five known non-settlement units, are shown as pale grey background points. The dashed line indicates the AMAP boundary. Points are shown at their recorded coordinates without displacement or spatial aggregation; marker size does not encode population. These are inventory records rather than a fully verified census of distinct, currently inhabited settlements. Base map: Natural Earth; projection: north polar stereographic, central meridian 45° W.

图中仅显示当前规则筛选结果，不代表各点在1991—2025年持续居住，也不表示航线或实际可航达性。港口缺少证据不等于确定没有港口。
''',encoding='utf-8')
(OUT/'导出记录.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),records=len(rows),excluded_background=len(excluded),ports=int(port.sum()),coastal=int(coastal.sum()),map_width_mm=183,map_height_mm=188,raster_dpi=600,projection=crs.to_string(),distance_bin_counts=table),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(table,ensure_ascii=False));print('Exports:',OUT)

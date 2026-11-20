"""Monthly native-grid sea-ice extent and concentration trends."""
from pathlib import Path
from calendar import month_name
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_hex
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.path import Path as Polygon
from matplotlib.patches import Circle
from matplotlib.ticker import MultipleLocator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'PIOMAS_v2.1_1979-2026'
BUILD = OUT / '构建'
FIG = OUT / '图件'
for folder in (BUILD, FIG, OUT / 'PPTX'):
    folder.mkdir(exist_ok=True)
YEARS = np.arange(1979, 2026)
MONTH_ENDS = np.r_[0, np.cumsum([31,28,31,30,31,30,31,31,30,31,30,31])]
SHAPE = (120, 360)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':True,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.linewidth':0.7,'svg.fonttype':'none','pdf.fonttype':42})


def base_map():
    # Reuse the established project's polar display background and overlays.
    template = ROOT / '数据选优/PIOMAS_2025年1月_插值掩膜预实验/plot_final_day_polar.py'
    source = template.read_text(encoding='utf-8')
    start = source.index('tr=Transformer')
    end = source.index('def panel')
    prefix = source[:source.index('def read')]
    prefix = '\n'.join(line for line in prefix.splitlines() if not line.startswith('CURRENT='))
    setup = '''\nwith rasterio.open(BASE/'03_正式5km栅格/land_mask_5km.tif') as d:
    land=d.read(1)==1;tf=d.transform;crs=d.crs
def read(p):
    with p.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))
coastal=read(ROOT/'预处理/PIOMAS_5km_标准流程_v1/固定点位/聚居点_4258条.csv')
noncoastal=[];deps=[]
'''
    scope = {'__file__':str(template)}
    exec(compile(prefix+setup+source[start:end],str(template),'exec'),scope)
    return scope


def fixed_width(file, width):
    return np.array([float(line[i:i+width]) for line in file.read_text().splitlines()
                     for i in range(0, len(line), width) if line[i:i+width].strip()])


overview_only = '--overview-only' in sys.argv
series_only = '--series-only' in sys.argv
dual_axis_only = '--dual-axis-only' in sys.argv
if overview_only:
    with (OUT/'数据/12个月趋势汇总.csv').open(encoding='utf-8-sig') as f:
        slope = np.array([float(r['slope_km2_per_decade']) for r in csv.DictReader(f)])
elif not (series_only or dual_axis_only):
    base = base_map()
    lon, lat = fixed_width(SOURCE/'static/grid.dat',8).reshape(2,*SHAPE)
    wet = fixed_width(SOURCE/'static/io.dat_360_120.output',2).reshape(SHAPE)>0
    metrics = fixed_width(OUT/'数据/grid.dat.pop',8).reshape(7,*SHAPE)
    area = metrics[2]*metrics[3]
    x, y = base['tr'].transform(lon,lat)
    points = np.column_stack([x.ravel(),y.ravel()])
    inside = np.zeros(points.shape[0],bool)
    for bx,by in base['boundary']:
        inside ^= Polygon(np.column_stack([bx,by])).contains_points(points)
    domain = wet & inside.reshape(SHAPE)
    weights = area[domain]
    if not weights.size or np.any(~np.isfinite(weights) | (weights<=0)):
        raise ValueError('Invalid native-grid analysis domain or area')

    extent = np.empty((len(YEARS),12))
    concentration = np.empty((len(YEARS),12,weights.size), dtype=np.float32)
    clipped = 0
    source_min, source_max = 1.0, 0.0
    for i,year in enumerate(YEARS):
        f = SOURCE/'decoded'/'area'/f'aiday.H{year}'
        expected = 365*np.prod(SHAPE)*4
        if f.stat().st_size!=expected:
            raise ValueError(f'{f.name} does not contain 365 source fields')
        raw = np.memmap(f,mode='r',dtype='<f4',shape=(365,*SHAPE))
        for month,(a,b) in enumerate(zip(MONTH_ENDS[:-1],MONTH_ENDS[1:])):
            c = raw[a:b,domain]
            if np.any(~np.isfinite(c) | (c < -0.05) | (c > 1.05)):
                raise ValueError(f'Unexpected concentration values in {year}-{month+1:02d}')
            source_min, source_max = min(source_min,float(c.min())), max(source_max,float(c.max()))
            clipped += int(np.count_nonzero((c<0)|(c>1)))
            c = np.clip(c,0,1)
            extent[i,month] = np.sum((c >= 0.15)*weights,axis=1).mean()
            concentration[i,month] = c.mean(axis=0,dtype=np.float64)
        if i%10==0 or i==len(YEARS)-1:
            print('Read',year,flush=True)

    year_centered = YEARS-YEARS.mean()
    ols_weights = year_centered / (year_centered @ year_centered)
    slope = (ols_weights @ extent)*10
    trend = np.full((12,*SHAPE),np.nan)
    trend[:,domain] = np.einsum('y,ymp->mp',ols_weights,concentration)*1000
    fit = extent.mean(axis=0)+year_centered[:,None]*(slope/10)
    span = int(YEARS[-1]-YEARS[0])
    stats = [
        dict(month=m+1, years=len(YEARS), slope_km2_per_decade=float(slope[m]),
             fitted_change_km2=float(slope[m]*span/10),
             first_month_km2=float(extent[0,m]), last_month_km2=float(extent[-1,m]))
        for m in range(12)]
    with (OUT/'数据/月度海冰范围.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['year','month','source_days','extent_km2','fitted_extent_km2'])
        for i,year in enumerate(YEARS):
            for m,days in enumerate(np.diff(MONTH_ENDS)):
                writer.writerow([year,m+1,days,f'{extent[i,m]:.6f}',f'{fit[i,m]:.6f}'])
    with (OUT/'数据/12个月趋势汇总.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    np.savez_compressed(OUT/'数据/PIOMAS原生格_月度浓度趋势.npz',
                        longitude=lon,latitude=lat,amap_ocean=domain,cell_area_km2=area,
                        trend_pp_per_decade=trend,years=YEARS,month=np.arange(1,13))


def corners(v):
    """Projected cell vertices from neighboring centers, with a cyclic seam."""
    padded = np.pad(v,((1,1),(1,1)),mode='wrap')
    padded[0] = 2*padded[1]-padded[2]
    padded[-1] = 2*padded[-2]-padded[-3]
    return (padded[:-1,:-1]+padded[1:,:-1]+padded[:-1,1:]+padded[1:,1:])/4


navy, rust, dark = '#235477','#af452e','#222f36'
pages = []
notes = (
    'PIOMAS v2.1 aiday，1979—2025年。每年365条源记录按固定月长度归并，2月28条。'
    'AMAP内原生海格按格心入域，海陆采用PIOMAS KMT。原生格面积为HTN×HTE。'
    '原生浓度数值限制在0—1。'
    '日海冰范围为浓度≥15%的格面积总和，月均值对当月日范围取平均。'
    '逐格月均浓度及研究区月均范围以年份做普通最小二乘拟合。'
    '地图格界由投影后相邻格心估计，趋势保持原生格数值。'
    '\nhttps://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/model_grid'
    '\nhttps://arctic.noaa.gov/report-card/report-card-2025/sea-ice-2025/'
    '\nhttps://www.itl.nist.gov/div898/handbook/pmd/section4/pmd431.htm'
    '\nhttps://doi.org/10.5194/tc-6-881-2012'
    '\nhttps://doi.org/10.1088/1748-9326/aade56'
    '\nhttps://github.com/Chutj/sci-figure-gallery'
)


def slide():
    return plt.figure(figsize=(13.333333,7.5),facecolor='white')


def save(fig, stem, note, font_family='Microsoft YaHei'):
    fig.savefig(FIG/f'{stem}.png',dpi=180)
    fig.set_dpi(144)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=list(fig.texts)
    for legend in fig.legends:
        texts.extend(legend.get_texts())
    for ax in fig.axes:
        texts.extend([ax.title,ax._left_title,ax._right_title,*ax.texts])
        if ax.axison:
            for axis in (ax.xaxis,ax.yaxis):
                if axis.get_visible():
                    low, high = sorted(axis.get_view_interval())
                    texts.extend([axis.label,axis.offsetText])
                    texts.extend(t for loc,t in zip(axis.get_majorticklocs(),axis.get_ticklabels())
                                 if low-1e-9<=loc<=high+1e-9)
    seen=set()
    items=[]
    scale=96/fig.dpi
    image_width=fig.get_figwidth()*96
    image_left=(1280-image_width)/2
    for t in texts:
        if id(t) in seen or not t.get_visible() or not t.get_text().strip():
            continue
        seen.add(id(t))
        box=t.get_window_extent(renderer)
        rot=float(t.get_rotation())
        t.set_rotation(0)
        flat=t.get_window_extent(renderer)
        t.set_rotation(rot)
        items.append(dict(text=t.get_text(), cx=(box.x0+box.x1)*scale/2+image_left,
                          cy=(fig.bbox.height-(box.y0+box.y1)/2)*scale,
                          width=flat.width*scale+10,height=flat.height*scale+8,
                          rotation=(-rot)%360,fontSize=t.get_fontsize()*96/72,
                          fontFamily=font_family,
                          color=to_hex(t.get_color()),
                          bold=t.get_fontweight() in ['bold','semibold',700]))
    for t in texts:
        t.set_visible(False)
    background=f'{stem}_background.png'
    fig.savefig(BUILD/background,dpi=180)
    pages.append(dict(name=stem,background=background,items=items,notes=note,
                      imageLeft=image_left,imageWidth=image_width))
    plt.close(fig)
    print('Rendered',stem,flush=True)


def title(fig, main, subtitle):
    fig.text(.056,.924,main,fontsize=23,weight='bold',color=dark)
    fig.text(.057,.863,subtitle,fontsize=11,color='#52636a')


def draw_overview(slopes):
    fig = slide()
    title(fig, '北极海冰范围的逐月下降趋势', 'PIOMAS v2.1   1979—2025年   AMAP内原生海格')
    ax = fig.add_axes([.106, .212, .827, .535])
    months = np.arange(1, 13)
    values = slopes / 1e4
    focus = np.isin(months, [3, 9])
    focus_color, normal_color = '#246679', '#a5bfc7'
    step = 30
    bottom = np.floor((min(0, values.min())-10) / step) * step
    top = max(0, np.ceil(values.max() / step) * step)
    ax.set_xlim(.4, 12.6)
    ax.set_ylim(bottom, top)
    ticks = np.arange(bottom, top+step, step)
    ax.set_yticks(ticks)
    ax.set_xticks(months, [f'{m}月' for m in months])
    ax.tick_params(axis='both', labelsize=10.5, length=0, pad=8, labelcolor=dark)
    for side in ('top', 'bottom', 'left'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('#87949a')
        ax.spines[side].set_linewidth(.65)
    ax.hlines(ticks[1:-1], .4, 12.6, color='#e6ebed', lw=.7, zorder=0)
    ax.set_axisbelow(True)
    ax.bar(months, values, width=.53, color=np.where(focus, focus_color, normal_color),
           edgecolor='none', zorder=2)
    if top > 0:
        ax.axhline(0, color='#53636a', lw=.85, zorder=3)
    fig.text(.106, .798, '海冰范围线性趋势（万 km²／十年）', fontsize=11, color=dark)
    for i, label in enumerate(ax.get_xticklabels()):
        if focus[i]:
            label.set_color(focus_color)
            label.set_fontweight('bold')
    for month, value, highlighted in zip(months, values, focus):
        ax.text(month, value-3 if value<0 else value+3, f'{value:+.1f}'.replace('-', '−'),
                ha='center', va='top' if value<0 else 'bottom',
                fontsize=10.5, color=focus_color if highlighted else '#435761',
                weight='bold' if highlighted else 'normal')
    for month, season in [(3, '冬末'), (9, '夏末')]:
        ax.text(month, 1.019, season, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=9.5, color=focus_color, clip_on=False)
    fig.text(.057, .093, '每根柱子由同一月份的47个年度值拟合得到。柱子向下表示范围缩小，越长表示下降越快。',
             fontsize=10, color='#52636a')
    save(fig, '00_全年逐月汇总', notes)


def regional_concentration_series():
    with np.load(OUT/'数据/PIOMAS原生格_月度浓度趋势.npz') as native:
        domain = native['amap_ocean']
        weights = native['cell_area_km2'][domain]
    mean_percent = np.empty((len(YEARS),12))
    for i, year in enumerate(YEARS):
        f = SOURCE/'decoded'/'area'/f'aiday.H{year}'
        raw = np.memmap(f, mode='r', dtype='<f4', shape=(365,*SHAPE))
        for m, (start, stop) in enumerate(zip(MONTH_ENDS[:-1],MONTH_ENDS[1:])):
            values = np.clip(raw[start:stop,domain],0,1)
            mean_percent[i,m] = 100*np.average(values.mean(axis=0,dtype=np.float64),weights=weights)
        if i%10==0 or i==len(YEARS)-1:
            print('Read concentration',year,flush=True)
    return mean_percent


def draw_series_pages(mean_percent, extent_million, fit_million):
    centered = YEARS - YEARS.mean()
    rate = (centered @ mean_percent)/(centered @ centered)
    concentration_fit = mean_percent.mean(axis=0)+centered[:,None]*rate
    with (OUT/'数据/AMAP月均海冰浓度.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['year','month','area_weighted_concentration_percent',
                         'fitted_concentration_percent','trend_pp_per_decade'])
        for i,year in enumerate(YEARS):
            for m in range(12):
                writer.writerow([year,m+1,f'{mean_percent[i,m]:.8f}',f'{concentration_fit[i,m]:.8f}',f'{rate[m]*10:.8f}'])
    extent_top = np.ceil(extent_million.max()/2)*2
    concentration_top = np.ceil(mean_percent.max()/10)*10
    note = notes + ('\n时间序列对照页。区域浓度为AMAP内全部原生海格的面积加权平均，'
                    '包括零浓度海格。面积分母为固定研究海域总面积。先取逐格月均，再按格面积加权。'
                    '每月47个年值做OLS拟合。')
    for m in range(12):
        fig = slide()
        title(fig, f'{m+1}月海冰范围与平均浓度', '时间序列对照   PIOMAS v2.1   1979—2025年   AMAP内原生海格')
        panels = [
            (0.107, 'a  海冰范围时间序列', extent_million[:,m], fit_million[:,m],
             '月均海冰范围（百万 km²）', extent_top, 2,
             f'{(fit_million[-1,m]-fit_million[0,m])*1000/46:+.2f} 万 km²／十年'),
            (0.573, 'b  海冰浓度时间序列', mean_percent[:,m], concentration_fit[:,m],
             '区域月均海冰浓度（%）', concentration_top, 10,
             f'{rate[m]*10:+.2f} 个百分点／十年'),
        ]
        for left, label, actual, fitted, ylabel, ymax, step, trend_label in panels:
            ax = fig.add_axes([left,.295,.369,.443])
            fig.text(left-.045,.782,label,fontsize=14,weight='bold',color=dark)
            ax.plot(YEARS,actual,color=navy,lw=1.5,marker='o',ms=2.4)
            ax.plot(YEARS,fitted,color=rust,lw=1.45,ls=(0,(5,3)))
            ax.set(xlim=(1978,2026),ylim=(0,ymax),xticks=[1980,1990,2000,2010,2020,2025])
            ax.yaxis.set_major_locator(MultipleLocator(step))
            ax.set_xlabel('年份',fontsize=11)
            ax.set_ylabel(ylabel,fontsize=11)
            ax.tick_params(labelsize=9)
            ax.grid(axis='y',color='#e6eaec',lw=.7)
            ax.set_axisbelow(True)
            fig.text(left,.183,'趋势  '+trend_label.replace('-', '−'),fontsize=13,weight='bold',color=dark)
        fig.text(.057,.102,'蓝线为月均值，红色虚线为线性拟合。右图按格面积对AMAP内全部原生海格的浓度取平均。',
                 fontsize=10,color='#52636a')
        save(fig,f'对照_{m+1:02d}月_范围与浓度时间序列',note)


def read_month_matrix(filename, column):
    values = np.full((len(YEARS),12), np.nan)
    with (OUT/'数据'/filename).open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            index = (int(row['year'])-int(YEARS[0]),int(row['month'])-1)
            values[index] = float(row[column])
    if not np.isfinite(values).all():
        raise ValueError(f'Incomplete monthly series in {filename}')
    return values


def draw_dual_axis_pages(extent, concentration, extent_fit, concentration_fit):
    font = 'Times New Roman'
    blue, orange = '#235477', '#b54c32'
    dash = (0,(5,3))
    note = notes + ('\n双Y轴对照。蓝色左轴为海冰范围，橙色右轴为AMAP内面积加权海冰浓度。'
                    '实线为逐年同月月均值，虚线为线性拟合。各月轴限按数据范围设置。')
    span = YEARS[-1]-YEARS[0]
    with plt.rc_context({'font.family':font}):
        for m in range(12):
            fig = plt.figure(figsize=(7.5*5.6/5.3,7.5),facecolor='white')
            left = fig.add_axes([.148,.096,.712,.806])
            right = left.twinx()
            e, c = extent[:,m], concentration[:,m]
            ef, cf = extent_fit[:,m], concentration_fit[:,m]
            for ax, actual, fitted, color in [(left,e,ef,blue),(right,c,cf,orange)]:
                ax.plot(YEARS,actual,color=color,lw=2.1,marker='o',ms=3.9)
                ax.plot(YEARS,fitted,color=color,lw=1.8,ls=dash,alpha=.83)
                ax.tick_params(axis='y',labelsize=14,colors=color,length=4)
            e_min,e_max=min(e.min(),ef.min()),max(e.max(),ef.max())
            c_min,c_max=min(c.min(),cf.min()),max(c.max(),cf.max())
            e_delta,c_delta=e_max-e_min,c_max-c_min
            e_bottom=max(0,np.floor(e_min-.26*e_delta))
            c_bottom=max(0,5*np.floor((c_min-.26*c_delta)/5))
            e_top=max(e_max+.30*e_delta,e_bottom+(max(e[19:].max(),ef[19:].max())-e_bottom)/.77)
            c_top=min(100,c_bottom+(c_max-c_bottom)/.56)
            if m == 10:
                e_bottom,e_top,c_bottom,c_top=6,12.65,30,75
            left.set(xlim=(1978,2026),ylim=(e_bottom,e_top),xticks=[1980,1990,2000,2010,2020,2025])
            right.set_ylim(c_bottom,c_top)
            left.yaxis.set_major_locator(MultipleLocator(1 if e_top-e_bottom<5 else 2))
            right.yaxis.set_major_locator(MultipleLocator(5 if c_top-c_bottom<28 else 10))
            left.set_ylabel('Sea-ice extent (10⁶ km²)',color=blue,fontsize=15,labelpad=9)
            right.set_ylabel('Sea-ice concentration (%)',color=orange,fontsize=15,labelpad=9)
            left.tick_params(axis='x',labelsize=14,color='#7f8d93',length=4)
            left.spines['left'].set_color(blue)
            left.spines['bottom'].set_color('#839097')
            right.spines['left'].set_visible(False)
            right.spines['bottom'].set_visible(False)
            right.spines['right'].set_visible(True)
            right.spines['right'].set_color(orange)
            e_rate,c_rate=(ef[-1]-ef[0])*10/span,(cf[-1]-cf[0])*10/span
            for y,label,color in [
                (.992,f'{e_rate:+.2f} × 10⁶ km² per decade',blue),
                (.952,f'{c_rate:+.2f} pp per decade',orange),
            ]:
                right.text(.987,y,label.replace('-','−'),transform=right.transAxes,
                           ha='right',va='top',fontsize=14,weight='bold',color=color)
            fig.legend(handles=[
                Line2D([0],[0],color=blue,lw=2,marker='o',ms=4,label='Extent'),
                Line2D([0],[0],color=orange,lw=2,marker='o',ms=4,label='Concentration'),
                Line2D([0],[0],color=blue,lw=1.8,ls=dash,label='Extent fit'),
                Line2D([0],[0],color=orange,lw=1.8,ls=dash,label='Concentration fit'),
            ],loc='upper right',bbox_to_anchor=(.988,.909),bbox_transform=right.transAxes,
               ncol=2,frameon=False,fontsize=12,borderaxespad=.1,columnspacing=1.05,
               handlelength=2,handletextpad=.45,labelspacing=.28,labelcolor='linecolor')
            fig.text(.148,.941,month_name[m+1],fontsize=21,weight='bold',color=dark)
            save(fig,f'双轴_{m+1:02d}月_范围与浓度',note,font_family=font)


if series_only or dual_axis_only:
    extent_series = read_month_matrix('月度海冰范围.csv','extent_km2')/1e6
    manifest = BUILD/'pages.json'
    previous = json.loads(manifest.read_text(encoding='utf-8'))
    if dual_axis_only:
        concentration_series = read_month_matrix('AMAP月均海冰浓度.csv','area_weighted_concentration_percent')
        extent_fit_series = read_month_matrix('月度海冰范围.csv','fitted_extent_km2')/1e6
        concentration_fit = read_month_matrix('AMAP月均海冰浓度.csv','fitted_concentration_percent')
        draw_dual_axis_pages(extent_series,concentration_series,extent_fit_series,concentration_fit)
        updated = [page for page in previous if not page['name'].startswith('双轴_')]+pages
    else:
        extent_fit_series = read_month_matrix('月度海冰范围.csv','fitted_extent_km2')/1e6
        draw_series_pages(regional_concentration_series(),extent_series,extent_fit_series)
        original = [page for page in previous if not page['name'].startswith(('对照_','双轴_'))]
        updated = original+pages+[page for page in previous if page['name'].startswith('双轴_')]
    manifest.write_text(json.dumps(updated,ensure_ascii=False),encoding='utf-8')
    sys.exit(0)

draw_overview(slope)
if overview_only:
    manifest = BUILD / 'pages.json'
    previous = json.loads(manifest.read_text(encoding='utf-8'))
    manifest.write_text(json.dumps(pages + previous[1:], ensure_ascii=False), encoding='utf-8')
    sys.exit(0)

xc,yc = corners(x),corners(y)
radius = base['radius']
norm = TwoSlopeNorm(vmin=-20,vcenter=0,vmax=20)
cmap = plt.get_cmap('RdBu').copy()
cmap.set_bad((0,0,0,0))

for m in range(12):
    fig=slide()
    title(fig,f'{m+1}月海冰范围与浓度趋势','PIOMAS v2.1   1979—2025年   47个同月样本')
    ax=fig.add_axes([.101,.301,.407,.448])
    fig.text(.064,.784,'a  海冰范围时间序列',fontsize=14,weight='bold',color=dark)
    ax.plot(YEARS,extent[:,m]/1e6,color=navy,lw=1.5,marker='o',ms=2.4)
    ax.plot(YEARS,fit[:,m]/1e6,color=rust,lw=1.45,ls=(0,(5,3)))
    ax.set_xlim(1978,2026)
    ax.set_ylim(0,np.ceil(extent.max()/2e6)*2)
    ax.set_xticks([1980,1990,2000,2010,2020,2025])
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylabel('月均海冰范围（百万 km²）',fontsize=11)
    ax.set_xlabel('年份',fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(axis='y',color='#e6eaec',lw=.7)
    ax.set_axisbelow(True)
    fig.text(.103,.194,f'趋势  {slope[m]/1e4:+.2f} 万 km²／十年',fontsize=14,weight='bold',color=dark)
    fig.text(.103,.151,f'1979—2025年拟合变化  {stats[m]["fitted_change_km2"]/1e4:+.1f} 万 km²',fontsize=10,color='#52636a')
    fig.text(.103,.105,'蓝线为月均值，红色虚线为线性拟合。',fontsize=9,color='#52636a')

    ax=fig.add_axes([.557,.191,.396,.595])
    fig.text(.554,.784,'b  海冰浓度趋势',fontsize=14,weight='bold',color=dark)
    ax.set_aspect('equal')
    ax.set_xlim(-radius*1.045,radius*1.045)
    ax.set_ylim(-radius*1.045,radius*1.045)
    ax.axis('off')
    disk=Circle((0,0),radius,facecolor='#dbe8ef',edgecolor='#83949c',lw=.75,zorder=0)
    ax.add_patch(disk)
    land_layer=ax.imshow(np.ma.masked_equal(base['display_land'],0),
                         extent=base['display_extent'],cmap=matplotlib.colors.ListedColormap(['#e1e2dd']),
                         interpolation='nearest',origin='upper',zorder=3)
    land_layer.set_clip_path(disk)
    mesh=ax.pcolormesh(xc,yc,trend[m],cmap=cmap,norm=norm,shading='flat',zorder=2,rasterized=True)
    mesh.set_clip_path(disk)
    coast=ax.contour(base['coast_x'],base['coast_y'],base['display_land'],
                     levels=[.5],colors=['#8a9699'],linewidths=.33,zorder=4)
    coast.set_clip_path(disk)
    for key,color,style,width in [('boundary','#4d5960',(0,(4,3)),.85),
                                  ('national_borders','#8a9699','solid',.42)]:
        collection=LineCollection([np.column_stack(line) for line in base[key]],
                                  colors=color,linestyles=style,linewidths=width,zorder=5)
        ax.add_collection(collection)
        collection.set_clip_path(disk)
    px,py=base['positions']['coastal']
    ax.scatter(px,py,s=1.3,color='#6a294b',alpha=.65,zorder=6).set_clip_path(disk)
    ax.text(-radius*.97,0,'50°N',fontsize=8,color=dark,ha='left',va='bottom')
    cax=fig.add_axes([.625,.145,.263,.019])
    bar=fig.colorbar(mesh,cax=cax,orientation='horizontal',extend='both',ticks=[-20,-10,0,10,20])
    bar.ax.tick_params(labelsize=8,length=3,pad=2)
    bar.outline.set_linewidth(.5)
    bar.set_label('海冰浓度趋势（百分点／十年）',fontsize=9,labelpad=3)
    fig.text(.555,.059,'紫点为沿海聚居地，虚线为AMAP边界，浅蓝为统计域外海洋。',fontsize=8.3,color='#52636a')
    save(fig,f'{m+1:02d}月_海冰范围与浓度',notes)

(BUILD/'pages.json').write_text(json.dumps(pages,ensure_ascii=False),encoding='utf-8')
summary=dict(year_start=1979,year_end=2025,years=47,native_ocean_cells=int(domain.sum()),
             domain_area_km2=float(weights.sum()),source_records=47*365,monthly=stats,
             clipped_cell_days=clipped,source_concentration_min=source_min,source_concentration_max=source_max,
             calendar='365 source fields per year, fixed month bins; February has 28 fields',
             map_scale_pp_per_decade=[-20,20],
             source_grid='HTN × HTE, PIOMAS grid.dat.pop',scope='PIOMAS wet cell centres in AMAP')
(OUT/'数据/分析摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

"""Build full-coverage R2 analytical figures on the existing project style."""
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, PowerNorm, to_hex
from matplotlib.patches import Circle, Patch, Polygon, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress

from run_r2 import ROOT, OUT, rows, DEPS
from summarize_r2 import MONTH_LENGTHS, MONTH_BOUNDS

BUILD=OUT/".构建"
BUILD.mkdir(exist_ok=True)
DATA=np.load(OUT/"R2_完整分析结果.npz")
Y=DATA["years"]
SHIPS=DATA["ships"].tolist()
MON=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
COLORS=["#103F5A","#216787","#5F95AE","#6E8B63","#4D783E","#284D35","#AE6D2C","#D99243","#6B517D","#805E92","#A583AE","#68496E"]
INK="#172A36"
BLUE="#246A83"
ORANGE="#AA552D"
PALE="#DCEBF0"
plt.rcParams.update({"font.family":"Times New Roman","font.size":10,"axes.spines.top":False,
                     "axes.spines.right":False,"axes.labelcolor":INK,"axes.edgecolor":"#73818A",
                     "xtick.color":INK,"ytick.color":INK,"text.color":INK,"savefig.facecolor":"white"})
pages=[]


def register(fig, kind, title, subtitle, note, file=None):
    file=file or f"plot_{len(pages)+1:03}.png"
    fig.savefig(BUILD/file,dpi=160)
    plt.close(fig)
    pages.append(dict(file=file, kind=kind, title=title, subtitle=subtitle, note=note))


def annual_axis(ax, percent=False):
    ax.set_xlim(Y[0], Y[-1])
    ax.set_xticks([1979,1990,2000,2010,2020,2025])
    ax.set_xlabel("Year")
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0,1.035)


def calendar(fig, ax, daily, label, cmap="YlGnBu"):
    cm=plt.get_cmap(cmap).copy()
    cm.set_bad("#E1E4E7")
    img=ax.imshow(daily,origin="lower",extent=(0,365,Y[0]-.5,Y[-1]+.5),
                  interpolation="nearest",aspect="auto",norm=PowerNorm(gamma=4 if cmap=="PuBuGn" else 1,vmin=0,vmax=1),cmap=cm)
    ax.set_xticks((MONTH_BOUNDS[1:]+MONTH_BOUNDS[:-1])/2,MON)
    ax.set_yticks([1979,1990,2000,2010,2020,2025])
    ax.set_ylabel("Year")
    cb=fig.colorbar(img,ax=ax,pad=.018,fraction=.035,ticks=[0,.5,.8,.9,1] if cmap=="PuBuGn" else [0,.25,.5,.75,1],format=PercentFormatter(1))
    cb.set_label(label)


def violin_hist(ax, hist, quant, max_value, bandwidth):
    x=np.arange(hist.shape[1])
    for k,y in enumerate(Y):
        h=hist[k]
        nz=np.flatnonzero(h)
        if len(nz)==1:
            ax.plot([y-.25,y+.25],[nz[0]]*2,c=BLUE,lw=1.3)
            continue
        density=gaussian_filter1d(h.astype(float),bandwidth,mode="nearest")
        density[:nz[0]]=0
        density[nz[-1]+1:]=0
        width=.40*density/max(density)
        ax.fill_betweenx(x,y-width,y+width,color=BLUE,alpha=.60,lw=0)
        low,mid,high=quant[k]
        ax.plot([y,y],[low,high],c=INK,lw=.75)
        ax.scatter([y],[mid],s=3.5,c=INK,zorder=4,linewidths=0)
    ax.set_ylim(-.03*max_value,1.03*max_value)
    annual_axis(ax)
    ax.set_xlim(Y[0]-.6,Y[-1]+.6)


def violin_sample(ax, x, data, color):
    if np.ptp(data) < 1e-10:
        ax.plot([x-.14,x+.14],[data[0]]*2,c=color,lw=2)
        return
    a=ax.violinplot(data,positions=[x],widths=.31,showextrema=False,
                   showmedians=True,quantiles=[[.25,.75]],points=100)
    for shape in a["bodies"]:
        shape.set_facecolor(color);shape.set_alpha(.55)
    for k in ("cmedians","cquantiles"):
        a[k].set_color(color);a[k].set_linewidth(1.25)


def prepare_summary():
    annual=DATA["annual"]
    slope=DATA["trend"]
    best=int(np.argmax(slope))
    delta=100*(annual[-10:].mean(axis=0)-annual[:10].mean(axis=0))
    md=DATA["monthly_delta_pp"]
    m,s=np.unravel_index(np.argmax(md),md.shape)
    d=dict(years=Y.tolist(),ships=SHIPS,annual=annual.tolist(),early=(annual[:10].mean(axis=0)*100).tolist(),
           late=(annual[-10:].mean(axis=0)*100).tolist(),delta=delta.tolist(),slope=slope.tolist(),
           monthly_delta=md.tolist(),biggest_month=[int(m)+1,SHIPS[s],float(md[m,s])],
           largest_trend=[SHIPS[best],float(slope[best])])
    monthly=DATA["monthly"]
    early,late=100*annual[:5].mean(axis=0),100*annual[-5:].mean(axis=0)
    d["five_year"]={"early":early.tolist(),"late":late.tolist(),"delta":(late-early).tolist(),
                    "monthly_delta":(100*(monthly[-5:].mean(axis=0)-monthly[:5].mean(axis=0))).tolist()}
    (BUILD/"summary.json").write_text(json.dumps(d,ensure_ascii=False),encoding="utf-8")


def five_year_overview():
    annual = DATA["annual"]
    change = 100 * (annual[-5:].mean(axis=0) - annual[:5].mean(axis=0))
    slope = DATA["trend"].copy()
    slope[np.ptp(annual, axis=0) == 0] = 0

    fig, ax = plt.subplots(figsize=(12.1, 4.75), layout="constrained")
    other = ax.twinx()
    other.spines["right"].set_visible(True)
    gain_line, = ax.plot(SHIPS, change, color=BLUE, lw=2, marker="o", ms=6,
                         label="Change between five-year means")
    slope_line, = other.plot(SHIPS, slope, color=ORANGE, lw=2, ls="--", marker="s", ms=6,
                             mfc="white", label="47-year linear trend")
    ax.set_ylim(-1, 20)
    other.set_ylim(-.25, 5)
    ax.set_yticks(range(0, 21, 5))
    other.set_yticks(range(6))
    ax.set_ylabel("Opportunity change (percentage points)", color=BLUE, fontsize=15)
    other.set_ylabel("Trend (percentage points per decade)", color=ORANGE, fontsize=15)
    ax.set_xlabel("Ship ice class", fontsize=15)
    ax.tick_params(axis="y", colors=BLUE, labelsize=14)
    other.tick_params(axis="y", colors=ORANGE, labelsize=14)
    ax.tick_params(axis="x", labelsize=13)
    ax.legend(handles=[gain_line, slope_line], loc="upper left", frameon=False, fontsize=14)
    fig.savefig(BUILD / "five-year-class-comparison.png", dpi=220)
    plt.close(fig)

    all_class_annual_figure(annual)


def all_class_annual_figure(annual):
    """Show annual opportunities on one axis with a separate key for each class group."""
    stem = "five-year-all-class-series"
    values = 100 * annual
    shared_top = np.array_equal(values[:, 0], values[:, 1])
    series = [i for i in range(len(SHIPS)) if i != 1 or not shared_top]
    palette = LinearSegmentedColormap.from_list(
        "ice_class_order", ["#17324D", "#286C92", "#3B8C82", "#838B58", "#B4613F"]
    )
    colors = dict(zip(series, palette(np.linspace(0, 1, len(series)))))
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 8.5, "text.color": "#25282C",
                         "axes.labelcolor": "#25282C", "axes.edgecolor": "#41464C",
                         "xtick.color": "#41464C", "ytick.color": "#41464C",
                         "axes.linewidth": .65, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(183 / 25.4, 86 / 25.4))
        ax = fig.add_axes([.088, .275, .882, .64])
        ax.set(xlim=(1979, 2025), ylim=(55, 102.5),
               xlabel="Year", ylabel="Maritime connection opportunity (%)")
        ax.set_xticks(range(1980, 2026, 5))
        ax.set_yticks([60, 70, 80, 90, 100])
        ax.tick_params(length=2.5, width=.6, pad=3)
        ax.grid(False)
        fig.text(.97, .955, "Faint dashed lines show linear trends",
                 ha="right", fontsize=7.5, color="#50575E")

        ymin = ax.get_ylim()[0]
        trend_clip = Rectangle((Y[0], ymin), Y[-1] - Y[0], 100 - ymin,
                               transform=ax.transData)
        nis = SHIPS.index("NIS")
        fill_boundary = np.vstack((np.column_stack((Y, values[:, nis])),
                                   [[Y[-1], 100], [Y[0], 100]]))
        fill_clip = Polygon(fill_boundary, transform=ax.transData)
        line_width = .82
        lines = {}
        for i in series:
            label = "PC1 / PC2" if i == 0 and shared_top else SHIPS[i]
            if i != nis and (i != 0 or not shared_top):
                for band in range(10):
                    fill = ax.fill_between(Y, values[:, i] - band * .65,
                                           values[:, i] - (band + 1) * .65, color=colors[i],
                                           alpha=.085 * (1 - band / 10) ** 2, linewidth=0, zorder=.4)
                    fill.set_clip_path(fill_clip)
            if i != 0 or not shared_top:
                fitted = values[:, i].mean() + (Y - Y.mean()) * DATA["trend"][i] / 10
                trend, = ax.plot(Y, fitted, color=colors[i], alpha=.23, lw=line_width,
                                 ls=(0, (3.5, 2.5)), zorder=2)
                trend.set_clip_path(trend_clip)
            lines[i], = ax.plot(Y, values[:, i], label=label, color=colors[i], ls="-",
                                lw=line_width, solid_capstyle="round", zorder=3)

        for title, indices, left in [("Polar classes", range(7), .088),
                                     ("Other ice classes", range(7, 12), .56)]:
            handles = [lines[i] for i in indices if i in lines]
            fig.text(left, .125, title, fontsize=8.5, weight="bold")
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left, .108),
                       ncols=3, fontsize=8, frameon=False, borderaxespad=0,
                       borderpad=0, handlelength=1.7, columnspacing=1.6, handletextpad=.6)

        fig.savefig(BUILD / f"{stem}.png", dpi=300)
        fig.savefig(OUT / "12船级年度海上联系机会.pdf")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        layers = []

        def add_layer(name, xy, transform, color, width=0, alpha=1, dashed=False):
            pixels = transform.transform(xy)
            pixels[:, 0] /= fig.bbox.width
            pixels[:, 1] = 1 - pixels[:, 1] / fig.bbox.height
            layers.append({
                "name": name, "points": pixels.tolist(), "color": to_hex(color),
                "width": width * fig.dpi / (72 * fig.bbox.width),
                "alpha": alpha, "dashed": dashed,
            })

        def insert_crossings(x, gaps):
            crossing = np.flatnonzero(gaps[:-1] * gaps[1:] < 0)
            extra = x[crossing] - gaps[crossing] * np.diff(x)[crossing] / np.diff(gaps)[crossing]
            return np.union1d(x, extra)

        boundary = values[:, nis]
        for i in series:
            if i == nis or (i == 0 and shared_top):
                continue
            for band in range(10):
                upper = values[:, i] - band * .65
                lower = upper - .65
                x = np.union1d(insert_crossings(Y, upper - boundary),
                               insert_crossings(Y, lower - boundary))
                floor = np.interp(x, Y, boundary)
                top = np.maximum(np.interp(x, Y, upper), floor)
                bottom = np.maximum(np.interp(x, Y, lower), floor)
                if np.any(top > bottom):
                    polygon = np.vstack((np.column_stack((x, top)),
                                         np.column_stack((x[::-1], bottom[::-1]))))
                    add_layer(f"{SHIPS[i]} 渐淡填充 {band + 1}", polygon, ax.transData,
                              colors[i], alpha=.085 * (1 - band / 10) ** 2)

        for i in series:
            if i == 0 and shared_top:
                continue
            fitted = values[:, i].mean() + (Y - Y.mean()) * DATA["trend"][i] / 10
            x = insert_crossings(Y, fitted - 100)
            y = np.interp(x, Y, fitted)
            visible = y <= 100 + 1e-9
            add_layer(f"{SHIPS[i]} 趋势线", np.column_stack((x[visible], y[visible])),
                      ax.transData, colors[i], width=line_width, alpha=.23, dashed=True)

        for i, line in lines.items():
            add_layer(f"{line.get_label()} 年度曲线", line.get_xydata(), line.get_transform(),
                      colors[i], width=line_width)
        for side in ("left", "bottom"):
            spine = ax.spines[side]
            add_layer(f"{side} axis", spine.get_path().vertices, spine.get_transform(),
                      spine.get_edgecolor(), width=spine.get_linewidth())
        for axis, offset in ((ax.xaxis, (0, -1)), (ax.yaxis, (-1, 0))):
            for tick in axis.get_major_ticks():
                mark = tick.tick1line
                start = mark.get_transform().transform(mark.get_xydata()[0])
                end = start + np.asarray(offset) * mark.get_markersize() * fig.dpi / 72
                add_layer("Axis tick", [start, end], fig.get_transform(),
                          mark.get_color(), width=mark.get_markeredgewidth())
        for legend in fig.legends:
            for line in legend.get_lines():
                add_layer(f"{line.get_label()} 图例线", line.get_xydata(), line.get_transform(),
                          line.get_color(), width=line.get_linewidth())

        labels = []
        for artist in fig.findobj(Text):
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            bounds = artist.get_window_extent(renderer)
            rotation = artist.get_rotation()
            labels.append({
                "text": artist.get_text(),
                "x": (bounds.x0 + bounds.width / 2) / fig.bbox.width,
                "y": 1 - (bounds.y0 + bounds.height / 2) / fig.bbox.height,
                "width": (bounds.height if rotation else bounds.width) / fig.bbox.width,
                "font": artist.get_fontsize() * fig.dpi / (72 * fig.bbox.width),
                "bold": artist.get_fontweight() == "bold",
                "color": to_hex(artist.get_color()),
                "rotation": -rotation,
            })
        (BUILD / f"{stem}-objects.json").write_text(
            json.dumps({"labels": labels, "layers": layers}, ensure_ascii=False), encoding="utf-8"
        )
        plt.close(fig)


def ship_figures(i,s):
    # All 47 values for each of the twelve same-month series.
    fig,(a,b)=plt.subplots(1,2,figsize=(12.1,4.7),layout="constrained",gridspec_kw={"width_ratios":[2.35,1]})
    for j,month in enumerate(MON):
        a.plot(Y,DATA["monthly"][:,j,i],color=COLORS[j],lw=1.4,label=month)
    annual_axis(a,True)
    a.set_ylabel("Connection opportunity")
    a.legend(ncol=6,loc="lower center",bbox_to_anchor=(.5,.04),frameon=True,facecolor="white",fontsize=8.6)
    z=DATA["monthly_contribution_pp"][:,i]
    b.bar(np.arange(12),z,color=np.where(z>=0,BLUE,ORANGE),width=.72)
    b.axhline(0,c="#617480",lw=.65)
    full=DATA["monthly_contribution_pp"]
    b.set_ylim(min(-.12,float(full.min())-.12),float(full.max())+.15)
    if np.allclose(z,0):
        a.text(.03,.89,"All 12 months overlap at 100%",transform=a.transAxes,color=INK,fontsize=9.5)
    b.set_xticks(np.arange(12),MON,rotation=48)
    b.set_ylabel("Contribution to annual change (pp)")
    register(fig,"monthly",f"{s}｜全区机会的逐月演变","47 年同月序列与各月对年度机会变化的贡献",
             "右图为末十年与首十年的同月机会差乘以月天数占全年比例。12 个月贡献相加为年度机会变化。")

    fig,(a,b)=plt.subplots(1,2,figsize=(12.1,4.65),layout="constrained",gridspec_kw={"width_ratios":[1,1.26]})
    v=DATA["largest"][:,:,i]
    p10,med,p90=np.quantile(v,[.1,.5,.9],axis=1)
    a.fill_between(Y,p10,p90,color=PALE,label="10th–90th percentile")
    a.plot(Y,med,color=BLUE,lw=1.7,label="Median")
    annual_axis(a,True);a.set_ylabel("Largest component share")
    a.legend(loc="lower right",frameon=False,fontsize=9)
    calendar(fig,b,v,"Largest component share")
    register(fig,"integration",f"{s}｜全区网络整合","年度中位数与第 10–90 百分位 · 逐日日历热图",
             "最大连通群体占比等于当天最大可通行海域分量所含聚居地数除以全区 4,258 个聚居地。")

    names=["Annual connected days","Longest continuous window (days)","Number of continuous windows"]
    hist=DATA["hist"][:,i]
    max_windows=max(1,int(np.flatnonzero(DATA["hist"][:,:,2,:].sum(axis=(0,1)))[-1]))
    scales=[365,365,max_windows]
    fig,axs=plt.subplots(3,1,figsize=(12.1,5.0),layout="constrained",sharex=True)
    for k,ax in enumerate(axs):
        violin_hist(ax,hist[:,k],DATA["pair_quantiles"][:,i,k],scales[k],4.5 if k<2 else .65)
        ax.set_ylabel(["Connected days","Longest run (days)","Window count"][k])
        if k<2:
            ax.set_yticks([0,180,365])
        if k<2: ax.set_xlabel("")
    register(fig,"pair_distribution",f"{s}｜全区点对的年度分布","每一年一个小提琴 · 47 年全部展示 · 每年 9,063,153 个点对",
             "分布按点对精确计数，平滑用于绘图。黑点和竖线分别标记中位数和四分位范围。零天与全年连通的比例在下一页。")

    fig,axs=plt.subplots(2,2,figsize=(12.1,4.95),layout="constrained")
    quant=DATA["pair_quantiles"][:,i]
    for k,ax in enumerate(axs.flat[:3]):
        q25,med,q75=quant[:,k].T
        ax.fill_between(Y,q25,q75,color=PALE,label="25th–75th percentile")
        ax.plot(Y,med,color=BLUE,lw=1.7,label="Median")
        ax.set_ylabel(names[k])
        annual_axis(ax)
        ax.set_ylim(0,scales[k]*1.045)
    axs.flat[0].legend(loc="lower right",frameon=False,fontsize=9)
    ax=axs.flat[3]
    n=int(DATA["candidate"])
    ax.plot(Y,hist[:,0,0]/n,c=ORANGE,lw=1.7,label="Zero connected days")
    ax.plot(Y,hist[:,0,-1]/n,c=BLUE,lw=1.7,label="Connected all year")
    annual_axis(ax,True);ax.set_ylabel("Share of settlement pairs")
    ax.legend(loc="best",frameon=False,fontsize=9)
    register(fig,"quantile",f"{s}｜累计与连续连通的长期变化","中位数和第 25–75 百分位 · 零天与全年连通点对比例",
             "每年独立计算连续窗口，年界处截断。最长窗口的中位数与累计天数中位数对照，可区分总时长与连续性变化。")

    fig,(a,b)=plt.subplots(1,2,figsize=(12.1,4.7),layout="constrained")
    calendar(fig,a,DATA["daily"][:,:,i],"Connected settlement pairs")
    calendar(fig,b,DATA["persistence"][:,:,i],"Previous-day connections retained",cmap="PuBuGn")
    register(fig,"calendar",f"{s}｜全区机会与次日延续率","逐日日历热图 · 每行一个年份",
             "左图为当天连通点对比例，右图分母为前一天连通的点对，采用非线性色阶突出接近 100% 时的差异。首日无上一日数据，以灰色表示。")

    fig,a=plt.subplots(figsize=(12.1,4.7),layout="constrained")
    o=DATA["daily"][:,:,i]
    for m,(lo,hi) in enumerate(zip(MONTH_BOUNDS[:-1],MONTH_BOUNDS[1:])):
        violin_sample(a,m-.18,o[:10,lo:hi].ravel(),ORANGE)
        violin_sample(a,m+.18,o[-10:,lo:hi].ravel(),BLUE)
    a.set_xticks(np.arange(12),MON)
    a.set_ylim(0,1.035);a.yaxis.set_major_formatter(PercentFormatter(1))
    a.set_ylabel("Daily connection opportunity")
    a.legend(handles=[Patch(color=ORANGE,label="1979–1988"),Patch(color=BLUE,label="2016–2025")],
             frameon=False,loc="lower right")
    register(fig,"daily_distribution",f"{s}｜同月逐日机会分布","12 个月完整对比 · 每个样本代表一天",
             "小提琴中心线标记中位数，两条细线标记第 25、75 百分位。常量用横线显示。")


def load_basemap():
    # Reuse the existing projection, land raster, ocean background and boundaries.
    p=ROOT/"数据选优/PIOMAS_2025年1月_插值掩膜预实验/plot_final_day_polar.py"
    code=p.read_text(encoding="utf-8")
    head=code[:code.index("CURRENT=")]
    setup=code[code.index("BASE="):code.index("def read")]
    land=code[code.index("with rasterio.open(BASE"):code.index("ncpath=")]
    geo=code[code.index("tr=Transformer"):code.index("positions={}")]
    geo="\n".join(s for s in geo.splitlines() if not s.startswith("plt.rcParams"))
    obj={"__file__":str(p)}
    exec(compile(head+setup+land+geo,str(p),"exec"),obj)
    return obj


def map_figures(five_year=False):
    g=load_basemap()
    tr,radius=g["tr"],g["radius"]
    data=rows(DEPS.with_name("聚居点与出发格对应.csv"))
    deps={r["departure_id"]:i for i,r in enumerate(rows(DEPS))}
    idx=np.array([deps[r["departure_id"]] for r in data])
    lon,lat=[np.array([float(r[k]) for r in data]) for k in ("settlement_longitude","settlement_latitude")]
    px,py=tr.transform(lon,lat)
    if five_year:
        annual=DATA["node_year"]
        metric=100*(annual[-5:].mean(axis=0)-annual[:5].mean(axis=0))[:,idx]
        breaks=[-20,-10,-5,0,5,10,15,20,30,45,60,100]
        heading="Settlement changes"
        label="Opportunity change (percentage points)"
        median_label,unit="Median change","pp"
        zero_label="No change"
    else:
        metric=DATA["node_slope"][:,idx].copy()
        breaks=[min(-6,int(np.floor(metric.min()))),-3,-1,0,1,2,3,5,7,10,15,
                max(27,int(np.ceil(metric.max())))]
        heading="Settlement trends"
        label="Opportunity trend (pp per decade)"
        median_label,unit="Median trend","pp per decade"
        zero_label="Zero trend"
    metric[np.abs(metric)<1e-9]=0
    colors=ListedColormap(["#A3422F","#CE714A","#ECA87B",
                          "#C9E5EB","#9ACEDC","#6BB7CC","#429DBD",
                          "#277DA9","#245E90","#1F4370","#192D51"])
    norm=BoundaryNorm(breaks,colors.N)
    for i,s in enumerate(SHIPS):
        fig,ax=plt.subplots(figsize=(12.1,5.0),layout="constrained")
        ax.set_xlim(-radius*1.02,radius*1.02);ax.set_ylim(-radius*1.02,radius*1.02)
        ax.set_aspect("equal");ax.axis("off")
        clip=Circle((0,0),radius,facecolor="#F0F5F9",edgecolor="#849199",lw=.7)
        ax.add_patch(clip)
        base=ax.imshow(g["display_land"],extent=g["display_extent"],
                       cmap=ListedColormap(["#F4F8FA","#ECEDE8"]),vmin=0,vmax=1,
                       interpolation="nearest",zorder=1)
        base.set_clip_path(clip)
        border=ax.contour(g["coast_x"],g["coast_y"],g["display_land"],levels=[.5],colors=["#7B878C"],linewidths=.28,zorder=2)
        border.set_clip_path(clip)
        for x,y in g["boundary"]:
            ax.plot(x,y,c="#536570",lw=.65,ls="--",zorder=3,clip_path=clip)
        for x,y in g["national_borders"]:
            ax.plot(x,y,c="#889197",lw=.42,zorder=2,clip_path=clip)
        angles=np.linspace(-180,180,361)
        for latring in [60,70,80]:
            ax.plot(*tr.transform(angles,np.full_like(angles,latring)),c="#9BAEBA",lw=.37,alpha=.7,zorder=2)
        for a in [-180,-135,-90,-45,0,45,90,135]:
            yy=np.linspace(50,90,82)
            ax.plot(*tr.transform(np.full_like(yy,a),yy),c="#9BAEBA",lw=.37,alpha=.65,zorder=2,clip_path=clip)
        values=metric[i]
        zero=values==0
        ax.scatter(px[zero],py[zero],s=6.5,c="#88949A",lw=.15,edgecolors="#F7FAFA",
                   clip_path=clip,zorder=4,rasterized=True)
        order=np.argsort(np.abs(values),kind="stable")
        order=order[~zero[order]]
        sc=ax.scatter(px[order],py[order],c=values[order],s=7.5,lw=.20,edgecolors="#F7FAFA",
                      cmap=colors,norm=norm,clip_path=clip,zorder=5,rasterized=True)
        cb=fig.colorbar(sc,ax=ax,orientation="vertical",fraction=.025,pad=.015,ticks=breaks)
        cb.set_label(label)
        cb.ax.tick_params(length=2.5,labelsize=9)
        if five_year:
            fig.text(.085,.83,"Distribution of settlement changes",fontsize=15,weight="bold",color=INK)
            hist=fig.add_axes([.085,.29,.365,.47])
            edges=np.arange(-22.5,103,5)
            counts,_=np.histogram(values,bins=edges)
            frequency=100*counts/len(values)
            centers=(edges[:-1]+edges[1:])/2
            hist.bar(centers,frequency,width=4.65,color="#A5C8D9",linewidth=0)
            hist.plot(centers,frequency,color="#224C67",linewidth=1.25)
            hist.set(xlim=(edges[0],edges[-1]),ylim=(0,frequency.max()*1.08),
                     xlabel="Opportunity change (percentage points)",
                     ylabel="Share of settlements (%)")
            hist.set_xticks([-20,0,20,40,60,80,100])
            hist.grid(False)
            fig.text(.085,.145,f"Median change  {np.median(values):+.2f} pp   ·   "
                     f"Increased  {(values>0).mean():.1%}",fontsize=10,color=INK)
            fig.text(.085,.088,"Dashed map line  AMAP boundary   ·   Grey points  No change",
                     fontsize=9,color="#4B6270")
            register(fig,"map_five_year",f"{s}｜聚居地海上联系机会首末五年变化",
                     "2021–2025 年均值减去 1979–1983 年均值 · 4,258 个沿海聚居地",
                     "色标单位为百分点，12 船级共用分档。蓝色为增加，暖色为减少，灰色为零变化。"
                     "共享出发格的聚居地采用相同变化值。直方图以 5 个百分点分箱，纵轴为各箱聚居地占比。"
                     "频率折线连接各箱中心对应的占比。",
                     file=f"five_year_map_{i+1:02}.png")
        else:
            fig.text(.105,.77,heading,fontsize=16,weight="bold",color=INK)
            for j,(name,value) in enumerate(((median_label,f"{np.median(values):+.2f} {unit}"),
                    ("Increasing",f"{(values>0).mean():.1%}"),("Decreasing",f"{(values<0).mean():.1%}"),
                    (zero_label,f"{(values==0).mean():.1%}"))):
                ypos=.66-j*.096
                fig.text(.105,ypos,name,fontsize=11,color="#4B6270")
                fig.text(.285,ypos,value,fontsize=12,color=INK)
            fig.text(.105,.19,"Dashed line   AMAP boundary",fontsize=10,color="#4B6270")
            fig.text(.105,.13,"Grey points   "+zero_label,fontsize=10,color="#4B6270")
            register(fig,"map",f"{s}｜聚居地海上联系机会趋势","1979–2025 年 · 4,258 个沿海聚居地 · 12 船级共用分段色标",
                     "色标在低至中等增幅范围采用较窄分档。蓝色为上升，暖色为下降，灰色为零趋势。共享出发格的聚居地采用相同趋势值。")
    if five_year:
        return
    fig,a=plt.subplots(figsize=(12.1,4.7),layout="constrained")
    for i in range(len(SHIPS)):
        violin_sample(a,i,metric[i],BLUE)
    a.axhline(0,lw=.65,color="#78858D")
    a.set_xticks(np.arange(len(SHIPS)),SHIPS)
    a.set_ylabel("Settlement opportunity trend (pp per decade)")
    register(fig,"map_distribution","聚居地机会趋势在船级之间的分布","每个样本为一个聚居地 · 12 个船级",
             "与趋势地图使用同一批 47 年拟合斜率。小提琴显示空间分布，中线与边线分别标记中位数和四分位数。")


def prepare_settlement_comparisons():
    departures={r["departure_id"]:i for i,r in enumerate(rows(DEPS))}
    settlements=rows(DEPS.with_name("聚居点与出发格对应.csv"))
    idx=[departures[r["departure_id"]] for r in settlements]
    annual=DATA["node_year"][:,:,idx]
    early,late=annual[:5].mean(axis=0),annual[-5:].mean(axis=0)
    comparisons=[]
    for ship,x,y in zip(SHIPS,early,late):
        delta=y-x
        fit=None
        if np.ptp(x)>0:
            result=linregress(x,y)
            ends=np.array([x.min(),x.max()])
            fit=dict(slope=result.slope,intercept=result.intercept,r2=result.rvalue**2,
                     x=ends.tolist(),y=(result.intercept+result.slope*ends).tolist())
        comparisons.append(dict(ship=ship,x=x.tolist(),y=y.tolist(),fit=fit,
                                higher=int((delta>1e-12).sum()),lower=int((delta< -1e-12).sum()),
                                median_pp=float(100*np.median(delta))))
    (BUILD/"settlement-five-year-comparisons.json").write_text(
        json.dumps(comparisons,ensure_ascii=False,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    if "--five-year-maps-only" in sys.argv[1:]:
        map_figures(five_year=True)
        (BUILD/"settlement_five_year_maps.json").write_text(
            json.dumps(pages,ensure_ascii=False,indent=2),encoding="utf-8")
        sys.exit()
    if "--settlement-comparisons-only" in sys.argv[1:]:
        prepare_settlement_comparisons()
        sys.exit()
    if "--maps-only" in sys.argv[1:]:
        manifest=BUILD/"pages.json"
        previous=json.loads(manifest.read_text(encoding="utf-8"))
        first=next(i for i,page in enumerate(previous) if page["kind"]=="map")
        pages.extend(previous[:first])
        map_figures()
        manifest.write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding="utf-8")
        sys.exit()
    if "--all-class-figure-only" in sys.argv[1:]:
        all_class_annual_figure(DATA["annual"])
    else:
        five_year_overview()
    if not {"--five-year-overview-only", "--all-class-figure-only"}.intersection(sys.argv[1:]):
        prepare_summary()
        for i,s in enumerate(SHIPS):
            ship_figures(i,s)
            print(s,flush=True)
        map_figures()
        prepare_settlement_comparisons()
        (BUILD/"pages.json").write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding="utf-8")


import json
from pathlib import Path

from matplotlib.colors import to_hex
from matplotlib.text import Text
import numpy as np


def export_layers(fig, stem):
    """Split scientific artwork, text, and legend marks into presentation layers."""
    stem=Path(stem)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    width,height=fig.canvas.get_width_height()
    scale=fig.dpi/72
    parts=[]
    hidden=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb=t.get_window_extent(renderer)
        if not np.isfinite(bb.extents).all() or bb.x1<=0 or bb.x0>=width or bb.y1<=0 or bb.y0>=height:
            continue
        rotation=t.get_rotation()
        raw_w,raw_h,_=renderer.get_text_width_height_descent(t.get_text().replace('\n',' '),t.get_fontproperties(),False)
        if rotation%360:
            w=max(raw_w,6)
            h=max(t.get_fontsize()*scale,raw_h)
        else:
            w,h=bb.width,bb.height
        parts.append(dict(kind='text',label=t.get_text(),x=(bb.x0+bb.x1)/2/width,y=1-(bb.y0+bb.y1)/2/height,
                          w=w/width,h=h/height,size=t.get_fontsize()/72/fig.get_figheight(),
                          bold=t.get_weight() in ('bold','heavy','semibold',600,700,800,900),
                          italic=t.get_fontstyle()=='italic',rotation=rotation,color=to_hex(t.get_color())))
        hidden.append(t)
    for cb in getattr(fig,'ppt_colorbars',[]):
        left,bottom,w,h=cb.ax.get_position().bounds
        for j in range(64):
            parts.append(dict(kind='rect',x=left+j*w/64,y=1-bottom-h,w=w/64,h=h,
                              color=to_hex(cb.cmap((j+.5)/64)),label='Colour key'))
        hidden.append(cb.solids)
    for legend in fig.legends:
        for handle in legend.legend_handles:
            if not hasattr(handle,'get_marker') or handle.get_marker()!='o':
                continue
            box=handle.get_window_extent(renderer)
            dia=handle.get_markersize()*scale
            line=handle.get_markeredgecolor()
            parts.append(dict(kind='ellipse',x=(box.x0+box.x1-dia)/2/width,
                y=1-(box.y0+box.y1+dia)/2/height,w=dia/width,h=dia/height,
                color=to_hex(handle.get_markerfacecolor()),line=None if line=='none' else to_hex(line),label='Figure legend'))
            hidden.append(handle)
    for t in hidden:
        t.set_visible(False)
    fig.savefig(stem.with_suffix('.background.png'),dpi=240)
    stem.with_suffix('.layers.json').write_text(json.dumps(dict(ratio=width/height,parts=parts),ensure_ascii=False),encoding='utf-8')
    for t in hidden:
        t.set_visible(True)

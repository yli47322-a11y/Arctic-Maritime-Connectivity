from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex

root=Path(__file__).resolve().parents[3]
out=root/'论文/结果图/3.1a_聚居地联系机会趋势/.制作'
data=np.load(root/'正式成果/R2_全区海上联系机会_完整分析/R2_完整分析结果.npz')
daily=data['daily']; years=data['years']; ships=data['ships']
breaks=np.array([40,50,60,70,80,90,95,98,99,99.5,99.9,100])/100
colors=plt.get_cmap('viridis_r')(np.linspace(.06,.96,len(breaks)-1))[:,:3]
colors=np.clip(.5+1.18*(colors-.5),0,1)
cmap=ListedColormap(colors); norm=BoundaryNorm(breaks,cmap.N,clip=True)
for s,ship in enumerate(ships):
    fig=plt.figure(figsize=(53.333/25.4,24/25.4),dpi=600)
    ax=fig.add_axes([0,0,1,1])
    ax.scatter(np.tile(np.arange(1,366),len(years)),np.repeat(years,365),
               c=daily[:,:,s].ravel(),s=.11,linewidths=0,cmap=cmap,norm=norm)
    ax.set_xlim(.5,365.5); ax.set_ylim(2025.8,1978.2);ax.set_axis_off()
    fig.savefig(out/f'daily_page4_{s:02}.png',dpi=600,transparent=True,pad_inches=0)
    plt.close(fig)
(out/'daily_page4.json').write_text(json.dumps({'ships':ships.tolist(),'breaks':(breaks*100).tolist(),'colors':[to_hex(c) for c in colors]}),encoding='utf-8')
print('Rendered 12 daily data panels from existing daily array.',flush=True)

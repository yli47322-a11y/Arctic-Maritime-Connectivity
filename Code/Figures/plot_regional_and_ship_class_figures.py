from pathlib import Path
import sys,json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'正式成果/R2_区域与船级不平等/.构建'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE))
from editable_figure import export_layers
d=json.loads((SOURCE/'report.json').read_text(encoding='utf-8'))
c=d['modes']['all']; e,l=[np.array(c[k])[2:] for k in ('early','late')]
annual=np.array(c['annual'])[:,2:]
ships=['IAS' if s=='IA Super' else s for s in d['ships'][2:]]
names=['Faroe Islands','Iceland Shelf','East Greenland','Norwegian Sea','Barents Sea','Kara Sea','Laptev Sea','East Siberian Sea','East Bering Sea','Aleutian Islands','West Bering Sea','N. Bering–Chukchi','Beaufort Sea','High Arctic–N. Greenland','Baffin Bay–W. Greenland','Hudson Bay','Labrador–Newfoundland','Gulf of Alaska','Sea of Okhotsk']
order=np.argsort(e[-1],kind='stable')
BLUE='#246F86'; ORANGE='#B65F46'; INK='#233D49'; GREY='#667D89'
W,H=25,20.5
mpl.rcParams.update({'font.family':'Times New Roman','font.size':12,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':GREY,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'savefig.facecolor':'white','pdf.fonttype':42})
CMAP=LinearSegmentedColormap.from_list('balance',[ORANGE,'#E6B8A4','#F1F3F1','#A8CBD3',BLUE])
def figure():return plt.figure(figsize=(W/2.54,H/2.54),dpi=120)
def axcm(f,x,y,w,h):return f.add_axes([x/W,1-(y+h)/H,w/W,h/H])
def label(f,x,y,t,bold=False):return f.text(x/W,1-y/H,t,va='center',fontsize=12,weight='bold' if bold else 'normal')
def title(f,x,t):label(f,x,0.55,t)
def axes_clean(ax):
 ax.tick_params(length=2.5,pad=4)
def matrix_labels(ax,regions=True):
 ax.set_xticks(np.arange(10)+.5,ships,rotation=50,ha='right',rotation_mode='anchor')
 ax.set_yticks(np.arange(19)+.5,[names[i] for i in order] if regions else [])
 ax.tick_params(length=0,pad=5)
 ax.set(xlim=(0,10),ylim=(19,0))
 for s in ax.spines.values():s.set_visible(False)
def colorbar(f,im,x,y,w,labeltext,limit):
 cb=f.colorbar(im,cax=axcm(f,x,y,w,.35),orientation='horizontal',ticks=np.linspace(-limit,limit,5))
 cb.outline.set_visible(False);cb.ax.tick_params(length=2,pad=3);cb.set_label(labeltext,labelpad=5)
 f.ppt_colorbars=getattr(f,'ppt_colorbars',[])+[cb]
def save(f,name,notes):
 f.savefig(OUT/(name+'.png'),dpi=240)
 export_layers(f,OUT/name)
 data=json.loads((OUT/(name+'.layers.json')).read_text(encoding='utf-8'))
 data.update(width_cm=W,height_cm=H,main_plot_left_cm=7,main_plot_right_cm=24,main_plot_width_cm=17,notes=notes)
 (OUT/(name+'.layers.json')).write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
 plt.close(f)

f=figure()
title(f,7,'(a) Regional mean change');title(f,16,'(b) Within-region SD change')
delta=(l-e).T[order]
sdchange=(np.array(c['within_node_sd_late'])[2:]-np.array(c['within_node_sd_early'])[2:]).T[order]
for k,(vals,x) in enumerate([(delta,7),(sdchange,16)]):
 ax=axcm(f,x,1.25,8,11.75)
 rr,cc=np.indices(vals.shape); size_max=65 if k==0 else 15
 area=lambda v:18+95*np.asarray(v)/size_max
 for mask,col in [(vals>=.05,BLUE),(vals<=-.05,ORANGE)]:
  ax.scatter(cc[mask]+.5,rr[mask]+.5,s=area(np.abs(vals[mask])),c=col,linewidths=0,zorder=3)
 z=np.abs(vals)<.05;ax.scatter(cc[z]+.5,rr[z]+.5,s=13,facecolor='white',edgecolor='#91A4AE',lw=.7)
 for y in range(20):ax.axhline(y,c='#E9EFF1',lw=.4,zorder=0)
 matrix_labels(ax,k==0)
 keys=[5,25,65] if k==0 else [1,6,15]
 handles=[Line2D([],[],marker='o',ls='',mec='none',color=GREY,ms=np.sqrt(area(v)),label=str(v)) for v in keys]
 f.legend(handles=handles,title='Magnitude (pp)',loc='upper center',bbox_to_anchor=((x+4)/W,1-14/H),ncol=3,columnspacing=1,handletextpad=.3,borderaxespad=0)
label(f,7,16.0,'(c) Between-region SD')
ax=axcm(f,7,16.75,17,2.6)
a,b=e.std(1),l.std(1);xx=np.arange(10)
ax.vlines(xx,a,b,color='#A9BCC5',lw=1.3);ax.scatter(xx,a,s=35,facecolor='white',edgecolor=GREY,lw=1,zorder=3);ax.scatter(xx,b,s=35,color=BLUE,zorder=4)
ax.set(xlim=(-.5,9.5),ylim=(0,35),xticks=xx,xticklabels=ships,yticks=[0,15,30],ylabel='SD (pp)');axes_clean(ax)
f.legend(handles=[Line2D([],[],ls='',marker='o',color=BLUE,label='Increase'),Line2D([],[],ls='',marker='o',color=ORANGE,label='Decrease'),Line2D([],[],ls='',marker='o',mfc='white',mec='#91A4AE',label='Near zero')],loc='center',bbox_to_anchor=(.67,1-15.8/H),ncol=3,columnspacing=1,handletextpad=.3)
f.legend(handles=[Line2D([],[],ls='',marker='o',mfc='white',mec=GREY,label='1979–1983'),Line2D([],[],ls='',marker='o',color=BLUE,label='2021–2025')],loc='center right',bbox_to_anchor=(.97,1-16.45/H),ncol=2,handletextpad=.3)
save(f,'figure1','区域差异：a为首末五年区域平均机会增量；b为域内聚居地机会标准差变化；c为19个海域平均机会的标准差，海域等权。全体目的地口径，与方法和3.2正文一致。PC1/2恒定基准不参与差异汇总。IAS=IA Super。主图横向范围7–24厘米，宽17厘米。')

f=figure();title(f,7,'(a) Ship-class SD by region');title(f,16,'(b) Pairwise gap change')
ax=axcm(f,7,2,7,13.4)
spread=annual.std(1);a=spread[:5].mean(0)[order];b=spread[-5:].mean(0)[order];y=np.arange(19)
cols=np.where(b>a,BLUE,ORANGE)
ax.hlines(y,a,b,color=cols,lw=1.2);ax.scatter(a,y,s=32,facecolor='white',edgecolor=GREY,zorder=3);ax.scatter(b,y,s=34,c=cols,zorder=4)
ax.set(xlim=(-.8,40),ylim=(18.7,-.7),xticks=[0,10,20,30,40],yticks=y,yticklabels=[names[i] for i in order],xlabel='SD across 10 ship classes (pp)');axes_clean(ax);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
f.legend(handles=[Line2D([],[],ls='',marker='o',mfc='white',mec=GREY,label='1979–1983'),Line2D([],[],ls='',marker='o',color=GREY,label='2021–2025')],loc='upper left',bbox_to_anchor=(7/W,1-1.1/H),ncol=2,handletextpad=.25,columnspacing=.8,borderaxespad=0)
g=lambda z:np.abs(z[:,None,:]-z[None,:,:]).mean(2)
diff=g(l)-g(e);ax=axcm(f,16,2,8,13.4)
im=ax.pcolormesh(diff,cmap=CMAP,norm=TwoSlopeNorm(0,vmin=-25,vmax=25),edgecolors='white',lw=.6)
ax.invert_yaxis();ax.set_xticks(np.arange(10)+.5,ships,rotation=50,ha='right',rotation_mode='anchor');ax.set_yticks(np.arange(10)+.5,ships);ax.tick_params(length=0,pad=5)
for s in ax.spines.values():s.set_visible(False)
colorbar(f,im,16,17.3,8,'Change in absolute gap (pp)',25)
f.legend(handles=[Line2D([],[],ls='',marker='o',color=ORANGE,label='Narrowed'),Line2D([],[],ls='',marker='o',color=BLUE,label='Widened')],loc='center',bbox_to_anchor=(10.5/W,1-17.7/H),ncol=2,handletextpad=.3,columnspacing=1)
save(f,'figure2','船级差异：a每年在各海域内计算十船级机会标准差，再分别平均1979–1983、2021–2025。b先在各海域内计算船级间绝对机会差，再对19海域等权平均，展示首末五年之差；完整上下三角及自比较保留。蓝色扩大，橙色缩小。全体目的地口径。主图宽17厘米。')

f=figure();title(f,7,'(a) Region × ship-class interaction');title(f,17,'(b) PC5–NIS gap change')
gain=l-e;resid=gain-gain.mean(0,keepdims=True)-gain.mean(1,keepdims=True)+gain.mean()
ax=axcm(f,7,1.5,8,13.8)
im=ax.pcolormesh(resid.T[order],cmap=CMAP,norm=TwoSlopeNorm(0,vmin=-40,vmax=40),edgecolors='white',lw=.55);matrix_labels(ax)
colorbar(f,im,7,17,8,'Change in interaction residual (pp)',40)
ax=axcm(f,17,1.5,7,13.8)
gap=(l[2]-l[-1])-(e[2]-e[-1]);v=gap[order];y=np.arange(19)+.5;cols=np.where(v>=0,BLUE,ORANGE)
ax.axvline(0,color=GREY,lw=.8);ax.hlines(y,0,v,color=cols,lw=1.3);ax.scatter(v,y,c=cols,s=35,zorder=3)
ax.set(xlim=(-13,46),ylim=(19,0),xticks=[-10,0,10,20,30,40],yticks=[],xlabel='Change in gap (pp)');ax.spines['left'].set_visible(False);axes_clean(ax)
for r in [13,16]:
 j=int(np.where(order==r)[0][0]);v=gap[r];ax.text(v+(1 if v>=0 else -1),j+.5,f'{v:+.1f}',ha='left' if v>=0 else 'right',va='center',fontsize=12,color=BLUE if v>=0 else ORANGE)
f.legend(handles=[Line2D([],[],ls='',marker='o',color=ORANGE,label='Narrowed'),Line2D([],[],ls='',marker='o',color=BLUE,label='Widened')],loc='center',bbox_to_anchor=(20.5/W,1-17.4/H),ncol=2,handletextpad=.3,columnspacing=1)
save(f,'figure3','交互分析：a为十船级×19海域的机会增量减去区域平均增量和船级平均增量，再加总体平均增量；蓝色高于加性基准，橙色低于基准。b展示PC5–NIS机会差距的变化，正值为扩大。比较1979–1983和2021–2025；全体目的地口径。IAS=IA Super。主图宽17厘米。')
print('Saved 3 figures and editable annotation layers',OUT)

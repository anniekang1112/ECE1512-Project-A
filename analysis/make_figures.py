"""Publication figures from frozen predictions/histories only; no new experiments."""
import os,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt,numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':250})
OUT=ROOT/'figures/generated';OUT.mkdir(exist_ok=True)
x=json.loads((ROOT/'analysis/generated/consolidated_results.json').read_text());colors={'E0':'#305e94','E1':'#e2922e','E2':'#27826b','E3':'#ad3f4c'}
def finish(fig,name):fig.tight_layout();fig.savefig(OUT/name,bbox_inches='tight');plt.close(fig)
# Learning curves retain every seed and avoid mixing weighted/unweighted objectives.
fig,axes=plt.subplots(3,2,figsize=(8.5,7.6),sharex=True)
for i,d in enumerate(['C16','C17','BRACS']):
 for j,v in enumerate(['E0','E3']):
  ax=axes[i,j];hs=[json.loads((ROOT/'results'/f'{d}_{v}_seed{s}'/'history.json').read_text()) for s in [42,43,44]]
  for key,color,label in [('train_loss',colors[v],'Training objective'),('val_loss','#444444','Validation CE')]:
   a=np.array([[r[key] for r in h] for h in hs]);mean=a.mean(0);sd=a.std(0,ddof=1);e=np.arange(1,31);ax.plot(e,mean,color=color,label=label);ax.fill_between(e,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.14)
  ax.set_title(d+' '+v);ax.set_ylabel('Loss');ax.grid(alpha=.15)
  if i==2:ax.set_xlabel('Epoch')
  if i==0:ax.legend(fontsize=10)
finish(fig,'training_curves_primary.png')
# Companion curves for all four variants, one readable figure per dataset.
for d in ['C16','C17','BRACS']:
 fig,axes=plt.subplots(2,2,figsize=(8.5,6.5),sharex=True)
 for ax,v in zip(axes.flat,['E0','E1','E2','E3']):
  hs=[json.loads((ROOT/'results'/f'{d}_{v}_seed{s}'/'history.json').read_text()) for s in [42,43,44]]
  for key,color,label in [('train_loss',colors[v],'Training objective'),('val_loss','#444444','Validation CE')]:
   a=np.array([[r[key] for r in h] for h in hs]);mean=a.mean(0);sd=a.std(0,ddof=1);e=np.arange(1,31);ax.plot(e,mean,color=color,label=label);ax.fill_between(e,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.14)
  ax.set(title=d+' '+v,xlabel='Epoch',ylabel='Loss');ax.grid(alpha=.15);ax.legend(fontsize=9)
 finish(fig,d+'_training_curves_all_variants.png')
# Full ablations, mean/SD and individual model seeds.
fig,axes=plt.subplots(1,3,figsize=(9,3.3))
for ax,d in zip(axes,['C16','C17','BRACS']):
 for i,v in enumerate(['E0','E1','E2','E3']):
  a=x['primary'][d][v]['accuracy'];ax.errorbar(i,100*a['mean'],yerr=100*a['sd'],fmt='o',color=colors[v],capsize=4)
 ax.set(title=d,xticks=range(4),xticklabels=['E0','E1','E2','E3'],ylabel='Test accuracy (%)',ylim=(0,100));ax.grid(axis='y',alpha=.15)
finish(fig,'primary_ablation_accuracy.png')
# Row-normalized confusion matrices averaged over seeds; panel for each primary comparison.
fig,axes=plt.subplots(2,3,figsize=(9,5.7))
for j,d in enumerate(['C16','C17','BRACS']):
 for i,v in enumerate(['E0','E3']):
  cms=[np.array(json.loads((ROOT/'results'/f'{d}_{v}_seed{s}'/'metrics.json').read_text())['confusion_matrix']) for s in [42,43,44]]
  a=np.mean([cm/cm.sum(1,keepdims=True) for cm in cms],axis=0);ax=axes[i,j];image=ax.imshow(a,vmin=0,vmax=1,cmap='Blues')
  for r in range(len(a)):
   for c in range(len(a)):
    if len(a)>4 and r!=c:continue
    ax.text(c,r,f'{a[r,c]:.2f}',ha='center',va='center',fontsize=9,color='white' if a[r,c]>.6 else 'black')
  ax.set(title=f'{d} {v}',xticks=range(len(a)),yticks=range(len(a)),xlabel='Predicted class',ylabel='True class')
finish(fig,'confusion_primary.png')
fig,axes=plt.subplots(1,3,figsize=(9,3.3))
for ax,k,label in zip(axes,['accuracy','recall_per_class','auc_macro_ovr'],['Accuracy (%)','Positive recall (%)','Macro OVR AUROC']):
 for v in ['E0','E3']:
  ps=[5,20,50];rs=[x['prevalence'][v][str(p)][k][1] if k=='recall_per_class' else x['prevalence'][v][str(p)][k] for p in ps];scale=1 if k=='auc_macro_ovr' else 100
  ax.errorbar(ps,[scale*r['mean'] for r in rs],yerr=[scale*r['sd'] for r in rs],color=colors[v],marker='o',capsize=3,label=v)
 ax.set(xlabel='Positive training bags (%)',ylabel=label,ylim=(0,1 if k=='auc_macro_ovr' else 100),xticks=[5,20,50]);ax.grid(alpha=.15);ax.legend()
finish(fig,'prevalence_three_seeds.png')
fig,axes=plt.subplots(1,3,figsize=(9,3.3))
for ax,d in zip(axes,['C16','C17','BRACS']):
 for v in ['E2','E3']:
  records=x['fraction_selected'][d][v]['selection_records'];means=[];sds=[]
  for p in [.05,.1,.25,.5]:
   vals=[r['selection_score'] for r in records if r['fraction']==p];means.append(statistics.mean(vals));sds.append(statistics.stdev(vals))
  ax.errorbar([5,10,25,50],means,yerr=sds,color=colors[v],marker='o',capsize=3,label=v)
 ax.set(title=d,xlabel='Retained patches (%)',ylabel='Validation AUROC + macro-F1',ylim=(.9,2.05));ax.grid(alpha=.15);ax.legend()
finish(fig,'fraction_validation_only.png')
fig,axes=plt.subplots(1,3,figsize=(9,3.3))
for ax,k,label in zip(axes,['accuracy','auc_macro_ovr','f1_macro'],['Accuracy (%)','Macro OVR AUROC','Macro-F1']):
 for j,(protocol,source) in enumerate([('Course',x['primary']['C17']),('Patient-independent',x['patient'])]):
  ys=[source[v][k]['mean'] for v in ['E0','E1','E2','E3']];es=[source[v][k]['sd'] for v in ['E0','E1','E2','E3']];scale=100 if k=='accuracy' else 1
  ax.errorbar(np.arange(4)+(j-.5)*.12,np.array(ys)*scale,yerr=np.array(es)*scale,marker='o',capsize=3,label=protocol)
 ax.set(xticks=range(4),xticklabels=['E0','E1','E2','E3'],ylabel=label,ylim=(0,100 if k=='accuracy' else 1));ax.grid(alpha=.15);ax.legend(fontsize=10)
finish(fig,'c17_protocol_comparison.png')
print('Generated final figures from frozen outputs')

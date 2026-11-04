"""Summarize completed experiments without training or selecting new models."""
import json,csv,statistics,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/generated';OUT.mkdir(exist_ok=True)
def load(name):return json.loads((ROOT/'results'/name/'metrics.json').read_text())
def mean_sd(values):return {'mean':statistics.mean(values),'sd':statistics.stdev(values) if len(values)>1 else None,'n':len(values)}
def summary(names):
    ms=[load(i) for i in names];r={'source_ids':names}
    for k in ['accuracy','auc_macro_ovr','auprc_macro_ovr','f1_macro','test_loss']:r[k]=mean_sd([m[k] for m in ms])
    for k in ['recall_per_class','specificity_per_class','auc_per_class','auprc_per_class']:r[k]=[mean_sd([m[k][c] for m in ms]) for c in range(len(ms[0][k]))]
    r['best_epochs']=[m['best_epoch'] for m in ms];return r
result={'primary':{},'prevalence':{},'patient':{},'fraction_selected':{},'interactions':{},'seeds':[42,43,44]};rows=[]
for d in ['C16','C17','BRACS']:
 result['primary'][d]={}
 for v in ['E0','E1','E2','E3']:
  ids=[f'{d}_{v}_seed{s}' for s in [42,43,44]];result['primary'][d][v]=summary(ids)
  for s,i in zip([42,43,44],ids):
   m=load(i);rows.append({'dataset':d,'variant':v,'seed':s,**{k:m[k] for k in ['accuracy','auc_macro_ovr','f1_macro','auprc_macro_ovr','best_epoch']},'id':i})
 for metric in ['accuracy','auc_macro_ovr','f1_macro']:
  values=[load(f'{d}_E3_seed{s}')[metric]-load(f'{d}_E2_seed{s}')[metric]-load(f'{d}_E1_seed{s}')[metric]+load(f'{d}_E0_seed{s}')[metric] for s in [42,43,44]]
  result['interactions'].setdefault(d,{})[metric]=mean_sd(values)
for v in ['E0','E3']:
 result['prevalence'][v]={}
 for p in [50,20,5]:
  names=[f'C16_{v}_seed42_prevalence{p:02d}']+[f'diag1_C16_{v}_seed{s}_prevalence{p:02d}_sample{sample}' for s,sample in [(43,2027),(44,2028)]]
  r=summary(names);r.update(total_bags=140,unique_positive=int(140*p/100),sampling_seeds=[2026,2027,2028]);result['prevalence'][v][str(p)]=r
for v in ['E0','E1','E2','E3']:result['patient'][v]=summary([f'diag1_C17_patient_{v}_seed{s}' for s in [42,43,44]])
for d in ['C16','C17','BRACS']:
 result['fraction_selected'][d]={}
 for v in ['E2','E3']:
  source=ROOT/'results'/f'diag1_select_{d}_{v}';selection=json.loads((source/'selection.json').read_text());releases=json.loads((source/'test_releases.json').read_text());names=[r['test_result'] for r in releases]
  r=summary(names);r.update(fraction=selection['selected_fraction'],validation_mean_score=selection['selected_mean_score'],selection_records=selection['records'],validation_scores=selection['mean_validation_scores']);result['fraction_selected'][d][v]=r

(ROOT/'analysis/generated/consolidated_results.json').write_text(json.dumps(result,indent=2))
with (OUT/'primary_seed_results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Saved numerical summaries to analysis/generated')

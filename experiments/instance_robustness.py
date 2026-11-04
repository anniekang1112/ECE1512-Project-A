"""
Prespecified test-time instance subsampling; never interpreted as tumor-fraction control."""

import sys,json,hashlib,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; START=ROOT/'Project_A_Starter_Code'; sys.path.insert(0,str(START))

import torch,numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
from utils import load_config
from dataloader import build_HDF5_feat_dataset,mil_collate_fn

from mil_pipeline import ABMIL
from train import atomic_json

def run(spec,out):

    conf=load_config(str(START/'config'/spec['config'])); conf.data_dir=str((START/conf.data_dir).resolve()); conf.dataset_csv_dir=str(START/'dataset_csv')
    _,_,test=build_HDF5_feat_dataset(conf.data_dir,conf); conf.D_feat=test.feature_dim; device=torch.device(spec.get('device','mps' if torch.backends.mps.is_available() else 'cpu')); torch.set_num_threads(4)
    rows=[]

    for variant in ['E0','E3']:
        source=ROOT/'results'/f"{spec['dataset_code']}_{variant}_seed42"

        if not (source/'best.pt').exists() or not (source/'predictions.json').exists(): raise RuntimeError('Source checkpoint and predictions required: '+str(source))

        conf.top_fraction=.1 if variant=='E3' else 1.; model=ABMIL(conf).to(device); model.load_state_dict(torch.load(source/'best.pt',map_location=device,weights_only=True)); model.eval()
        reference=json.loads((source/'predictions.json').read_text()); baseline={i:np.array(p) for i,p in zip(reference['ids'],reference['probabilities'])}

        for fraction in [.5,.1]:

            for repeat in [0,1,2]:

                labels=[]; probs=[]; reference_probs=[]; ids=[]

                with torch.no_grad():

                    for i in range(len(test)):
                        item=test[i]; slide_id=item['slide_id']; seed=int.from_bytes(hashlib.sha256(f'{slide_id}:{repeat}:2026'.encode()).digest()[:8],'little')

                        # Same permutation for E0/E3, nested subsets across fractions.

                        indices=np.random.default_rng(seed).permutation(len(item['input']))[:max(1,math.ceil(len(item['input'])*fraction))]
                        item['input']=item['input'][indices]

                        if item['coords'] is not None: item['coords']=item['coords'][indices]
                        batch=mil_collate_fn([item]); p=model(batch['input'].to(device),mask=batch['mask'].to(device)).softmax(1).cpu().numpy()[0]

                        ids.append(slide_id); labels.append(item['label']); probs.append(p); reference_probs.append(baseline[slide_id])

                y=np.array(labels); p=np.array(probs); original=np.array(reference_probs); aucs=[float(roc_auc_score(y==c,p[:,c])) for c in range(conf.n_class) if (y==c).any() and not (y==c).all()]
                row={'variant':variant,'fraction':fraction,'repeat':repeat,'accuracy':float(accuracy_score(y,p.argmax(1))),'f1_macro':float(f1_score(y,p.argmax(1),average='macro',labels=list(range(conf.n_class)),zero_division=0)),'auc_macro_ovr':float(np.mean(aucs)) if len(aucs)==conf.n_class else None,'prediction_flip_rate':float((p.argmax(1)!=original.argmax(1)).mean()),'mean_probability_l1_change':float(np.abs(p-original).sum(1).mean())}

                rows.append(row); atomic_json(out/'robustness.json',{'protocol':'Test-time random instance subsampling; not controlled tumor fraction. Fraction/repeat fixed before evaluation; no selection based on these test results.','rows':rows}); print(json.dumps(row),flush=True)

    return rows



if __name__=='__main__':
    spec=json.loads(Path(sys.argv[1]).read_text()); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True); run(spec,out)

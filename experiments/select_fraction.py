"""
Lock a fraction using validation histories only, then release selected test evaluations

"""

import sys,json,statistics,subprocess,hashlib,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

from train import atomic_json

def select(spec,out):
    # write a locked validation choice, then evaluate only the selected weights

    # This selection stage never reads metrics.json, predictions.json or test labels

    candidates=spec['candidates']; scores={}; records=[]
   
    for text,seeds in candidates.items():
        values=[]
   
        for seed,run_id in seeds.items():
            source=ROOT/'results'/run_id
   
            if not (source/'history.json').exists(): raise RuntimeError('Candidate history missing: '+run_id)
   
            history=json.loads((source/'history.json').read_text())
   
            if len(history)!=30: raise ValueError('Candidate budget must be 30 epochs')
            best=max(range(len(history)),key=lambda i:history[i]['val_auc']+history[i]['val_f1']); score=history[best]['val_auc']+history[best]['val_f1']; values.append(score)
            records.append({'fraction':float(text),'seed':int(seed),'run_id':run_id,'val_auc':history[best]['val_auc'],'val_f1':history[best]['val_f1'],'selection_score':score,'epoch':best+1})
   
        if len(values)!=3: raise ValueError('Three paired model seeds required')
   
        scores[text]=statistics.mean(values)
   
    # max keeps the first exact tie: prefer the primary 10% setting
   
    preference=[.1,.05,.25,.5];winner=max(preference,key=lambda f:scores[str(f)])
    result={'dataset':spec['dataset_code'],'variant':spec['variant'],'selected_fraction':winner,'selected_mean_score':scores[str(winner)],'mean_validation_scores':scores,'records':records,'criterion':'mean across seeds of best-validation macro AUROC + macro F1','tie_preference':preference,'test_data_used_for_selection':False}
    result['selection_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    locked=out/'selection.json'
   
    if locked.exists() and json.loads(locked.read_text())!=result: raise RuntimeError('Selection changed after lock; refusing test release')
   
    atomic_json(locked,result)
   
    # Stage separation: no test predictions are accessed until the validation choice above is locked.
   
    releases=[]
   
   
    for seed,source_id in candidates[str(winner)].items():
        source=ROOT/'results'/source_id; source_spec=json.loads((source/'spec.json').read_text())
   
        if source_spec.get('evaluation_policy')!='validation_only':
            releases.append({'seed':int(seed),'source':source_id,'test_result':source_id,'action':'reuse previously reported fixed-10% ablation result'})
            continue
   
        release_id=f"diag1_selected_{spec['dataset_code']}_{spec['variant']}_seed{seed}"; target=ROOT/'results'/release_id; target.mkdir(exist_ok=True)
        release_spec=dict(source_spec,id=release_id,evaluation_policy='final_test',evaluation_source=source_id,selection_lock=str(locked.relative_to(ROOT)))
   
        if (target/'metrics.json').exists():
   
            if json.loads((target/'spec.json').read_text())!=release_spec: raise RuntimeError('Completed release specification mismatch')
   
   
        else:
            atomic_json(target/'spec.json',release_spec)
   
            try:
                with (target/'run.log').open('a') as log: subprocess.run([sys.executable,str(ROOT/'experiments/train.py'),str(target/'spec.json'),str(target)],stdout=log,stderr=subprocess.STDOUT,check=True)
   
            except Exception:
                raise
   
   
        releases.append({'seed':int(seed),'source':source_id,'test_result':release_id,'action':'test evaluation only; no optimizer steps'})
   
    atomic_json(out/'test_releases.json',releases)



if __name__=='__main__':
    spec=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(exist_ok=True,parents=True);select(spec,out)

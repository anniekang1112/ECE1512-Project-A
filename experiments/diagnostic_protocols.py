"""Fixed diagnostic protocols; not alter original course loaders/splits."""
import json
from pathlib import Path
import numpy as np

def apply_protocol(train,val,test,conf,spec):
    
    # adjust train/val in place and return protocol details; test stays fixed
    info={'protocol':spec.get('protocol','course'),'sampling_seed':spec.get('sampling_seed',2026)}
    
    if spec.get('protocol','course')=='patient_independent':
        if conf.dataset!='CAMELYON17': raise ValueError('Patient IDs unavailable for this dataset')
        
        import pandas as pd
        from sklearn.model_selection import StratifiedGroupKFold

        metadata=pd.read_csv(Path(conf.dataset_csv_dir)/'camelyon17.csv').set_index('slide_id')
        pool=sorted(train.data_names+val.data_names); labels=metadata.loc[pool,'label'].to_numpy(); groups=metadata.loc[pool,'case_id'].to_numpy()
        
        # sklearn grouped split: fix the first fold before looking at performance
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
        
        tr,va=next(StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=42).split(pool,labels,groups))
        
        train.data_names=[pool[i] for i in tr]; val.data_names=[pool[i] for i in va]
        
        cases=[set(metadata.loc[d.data_names,'case_id']) for d in [train,val,test]]
        
        if any(cases[i]&cases[j] for i in range(3) for j in range(i)): raise ValueError('Patient overlap in grouped protocol')
        
        if set(np.unique(labels[tr]))!=set(range(conf.n_class)) or set(np.unique(labels[va]))!=set(range(conf.n_class)): raise ValueError('Missing class in grouped split')
        info.update(group_split='StratifiedGroupKFold(10,shuffle=True,random_state=42), first fold; no test labels used',case_counts=[len(x) for x in cases])
    
    
    return info

def prevalence_subset(train,conf,spec,labels):
    
    # return sampling details after selecting a common number of unique bags
    labels=np.asarray(labels); positive=np.flatnonzero(labels==1); negative=np.flatnonzero(labels==0)
    total=int(min(min(len(positive)/p,len(negative)/(1-p)) for p in [.5,.2,.05])); total=(total//20)*20
    if total<100: raise ValueError('Insufficient bags for common-N prevalence study with >=5 positives')
    fraction=float(spec['train_positive_fraction'])
    if fraction not in [.5,.2,.05]: raise ValueError('Unsupported prespecified prevalence')
    n_positive=int(round(total*fraction)); rng=np.random.default_rng(int(spec.get('sampling_seed',2026)))
    # same seeded permutations give paired, nested subsets without replacement
    chosen=np.concatenate([rng.permutation(positive)[:n_positive],rng.permutation(negative)[:total-n_positive]])
    train.data_names=[train.data_names[i] for i in chosen]
    return {'total':total,'unique_positive':n_positive,'unique_negative':total-n_positive,'sampling_seed':int(spec.get('sampling_seed',2026)),'replacement':False}

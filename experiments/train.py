"""
One resumable experiment. 
Test labels are used only after validation selection

"""

import os,sys,json,time,random,hashlib,shutil,resource
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; START=ROOT/'Project_A_Starter_Code'; sys.path.insert(0,str(START))

os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))

import numpy as np
import torch,h5py

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score,f1_score
from utils import load_config,seed_everything
from dataloader import build_HDF5_feat_dataset,mil_collate_fn,_labels_for_names
from mil_pipeline import ABMIL,train_one_epoch,evaluate

def atomic_json(path,value):
    tmp=Path(str(path)+'.tmp'); tmp.write_text(json.dumps(value,indent=2,allow_nan=False)); tmp.replace(path)

def save(path,obj):
    tmp=Path(str(path)+'.tmp'); torch.save(obj,tmp); tmp.replace(path)

def run(spec,out):

    # run one saved spec; checkpoints and metrics go in its own output folder

    atomic_json(out/'source_provenance.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'Project_A_Starter_Code/mil_pipeline.py',ROOT/'experiments/diagnostic_protocols.py']})
    if spec['variant'] not in ['E0','E1','E2','E3']: raise ValueError('Unknown variant')
    conf=load_config(str(START/'config'/spec['config'])); conf.data_dir=str((START/conf.data_dir).resolve()); conf.dataset_csv_dir=str(START/'dataset_csv')

    # Keep partition seed fixed across model seeds.

    # model seed changes below, after the course split has been built

    train,val,test=build_HDF5_feat_dataset(conf.data_dir,conf)

    if any(len(d)==0 for d in [train,val,test]): raise ValueError('Empty split')

    from diagnostic_protocols import apply_protocol,prevalence_subset

    protocol_info=apply_protocol(train,val,test,conf,spec)

    if 'train_positive_fraction' in spec:

        if conf.n_class!=2: raise ValueError('Binary prevalence experiment requires two classes')
        protocol_info['prevalence']=prevalence_subset(train,conf,spec,_labels_for_names(conf.data_dir,train.data_names,conf))

    conf.D_feat=train.feature_dim
    conf.top_fraction=float(spec.get('top_fraction',.1)) if spec['variant'] in ['E2','E3'] else 1.


    if spec['variant'] in ['E0','E1'] and spec.get('top_fraction',1.)!=1.: raise ValueError('Dense variant cannot override attention fraction')
    atomic_json(out/'protocol.json',protocol_info)
    seed=int(spec['seed']); seed_everything(seed); torch.set_num_threads(4)
    device=torch.device(spec.get('device','cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')))
    metadata={}; seen=set()

    for name,ds in [('train',train),('val',val),('test',test)]:
        ids=set(ds.data_names)

        if seen & ids: raise ValueError('Slide overlap between splits')
        seen |= ids
        labels=_labels_for_names(conf.data_dir,ds.data_names,conf)

        if min(labels)<0 or max(labels)>=conf.n_class: raise ValueError('Labels outside configured classes')

        with h5py.File(conf.data_dir) as h:
            shapes=[h[k]['feat'].shape for k in ds.data_names]

            if any(len(s)!=2 or s[0]<1 or s[1]!=conf.D_feat for s in shapes): raise ValueError('Invalid feature shape')

        metadata[name]={'ids':ds.data_names,'counts':np.bincount(labels,minlength=conf.n_class).tolist(),'instances':{'min':min(s[0] for s in shapes),'max':max(s[0] for s in shapes),'median':float(np.median([s[0] for s in shapes]))}}


    if conf.dataset=='CAMELYON17':
        import pandas as pd
        rows=pd.read_csv(START/'dataset_csv/camelyon17.csv').set_index('slide_id')
        a=set(rows.loc[train.data_names,'case_id']); b=set(rows.loc[val.data_names,'case_id']); metadata['train_val_shared_cases']=sorted(a&b)


    atomic_json(out/'data_audit.json',metadata)
    fingerprint=hashlib.sha256(json.dumps({'config':vars(conf),'variant':spec['variant'],'seed':seed,'splits':metadata,'train_positive_fraction':spec.get('train_positive_fraction'),'protocol':spec.get('protocol','course'),'sampling_seed':spec.get('sampling_seed',2026)},sort_keys=True).encode()).hexdigest()


    if (out/'fingerprint.txt').exists() and (out/'fingerprint.txt').read_text()!=fingerprint:
        raise ValueError('Configuration/data changed: use a new experiment ID')
    (out/'fingerprint.txt').write_text(fingerprint)

    atomic_json(out/'config.json',dict(vars(conf),experiment=spec,device=str(device)))
    t_load=time.perf_counter(); example=train[0]; loading_seconds=time.perf_counter()-t_load

    atomic_json(out/'benchmark.json',{'first_bag_load_seconds':loading_seconds,'first_bag_shape':list(example['input'].shape),'device':str(device),'real_epoch_estimate':'pending'})
    loaders=[DataLoader(ds,batch_size=1,shuffle=(i==0),collate_fn=mil_collate_fn,num_workers=0) for i,ds in enumerate([train,val,test])]
    model=ABMIL(conf).to(device); opt=torch.optim.AdamW(model.parameters(),lr=conf.lr,weight_decay=conf.wd)


    # class weights from train split only, not validation or test
    # standard inverse-frequency rule: w_c = N / (C * n_c)

    counts=np.array(metadata['train']['counts']); weights=None

    if spec['variant'] in ['E1','E3']:

        if (counts==0).any(): raise ValueError('Cannot weight absent training class')
        weights=torch.tensor(counts.sum()/(len(counts)*counts),dtype=torch.float32,device=device)


    atomic_json(out/'loss_config.json',{'class_weights':None if weights is None else weights.cpu().tolist(),'class_counts_training_only':counts.tolist(),'reduction':'unreduced weighted CE followed by minibatch mean','validation_loss':'unweighted CE'})
    
    # explicit bag mean in train_one_epoch keeps weights effective at batch size 1
    criterion=torch.nn.CrossEntropyLoss(weight=weights,reduction="none"); evaluation_loss=torch.nn.CrossEntropyLoss()
    last=out/'last.pt'; start=0; history=[]; best=float('-inf'); best_epoch=-1


    if spec.get('evaluation_source'):
        source=ROOT/'results'/spec['evaluation_source']
        shutil.copy2(source/'best.pt',out/'best.pt'); shutil.copy2(source/'history.json',out/'history.json')
        history=json.loads((out/'history.json').read_text()); best_epoch=max(range(len(history)),key=lambda i:history[i]['val_auc']+history[i]['val_f1']); start=int(spec.get('epochs',conf.train_epoch)); best=history[best_epoch]['val_auc']+history[best_epoch]['val_f1']

    if last.exists():
        state=torch.load(last,map_location=device,weights_only=False); model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); start=state['epoch']+1; history=state['history']; best=state['best']; best_epoch=state['best_epoch']; random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'].cpu())
        if device.type=='cuda' and state.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        if device.type=='mps' and state.get('mps_rng') is not None: torch.mps.set_rng_state(state['mps_rng'].cpu())

    for epoch in range(start,min(int(spec.get('epochs',conf.train_epoch)),int(spec.get('stop_after_epochs',spec.get('epochs',conf.train_epoch))))):

        if shutil.disk_usage(ROOT).free<10*1024**3: raise RuntimeError('Disk guard: less than 10 GiB free')
        t=time.perf_counter(); tr=train_one_epoch(model,criterion,loaders[0],opt,device); va=evaluate(model,evaluation_loss,loaders[1],device,conf)
        score=va['auc']+va['f1']

        if not np.isfinite(score): raise ValueError('Undefined validation selection metric')

        # save best based on validation only; strict comparison keeps earliest tie
        if score>best: best=score; best_epoch=epoch; save(out/'best.pt',model.state_dict())

        row={'epoch':epoch+1,'seconds':time.perf_counter()-t,**{'train_'+k:v for k,v in tr.items()},**{'val_'+k:v for k,v in va.items()}}; history.append(row)

        save(last,{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'history':history,'best':best,'best_epoch':best_epoch,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'mps_rng':torch.mps.get_rng_state() if device.type=='mps' else None})
        atomic_json(out/'history.json',history)
        atomic_json(out/'benchmark.json',{'first_bag_load_seconds':loading_seconds,'device':str(device),'measured_epoch_seconds':row['seconds'],'mean_epoch_seconds':float(np.mean([r['seconds'] for r in history])),'projected_training_seconds':float(np.mean([r['seconds'] for r in history]))*int(spec.get('epochs',conf.train_epoch)),'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,'mps_driver_bytes_observed':tr.get('mps_driver_bytes_observed'),'mps_tensor_bytes_observed':tr.get('mps_tensor_bytes_observed'),'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'memory_note':'MPS values sampled after backward per bag; not guaranteed instantaneous peak. macOS process RSS is high-water bytes.'})

        print(json.dumps(row),flush=True)


    if spec.get('benchmark_only'):
        return
    selected=history[best_epoch]
    atomic_json(out/'validation_selection.json',{'best_epoch':best_epoch+1,'selection_score':selected['val_auc']+selected['val_f1'],'val_auc':selected['val_auc'],'val_f1':selected['val_f1'],'val_loss':selected['val_loss'],'tie_rule':'earliest epoch','evaluation_policy':spec.get('evaluation_policy','final_test')})


    if spec.get('evaluation_policy')=='validation_only':
        if (out/'metrics.json').exists() or (out/'predictions.json').exists(): raise RuntimeError('Validation-only candidate unexpectedly contains test outputs')
        return


    # held-out test happens after checkpoint selection is finished
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True)); model.eval(); probabilities=[]; labels=[]; ids=[]; attention_examples={}

    with torch.no_grad():

        for batch in loaders[2]:
            logits,attention=model(batch['input'].to(device),mask=batch['mask'].to(device),return_attention=True); probabilities.extend(logits.softmax(1).cpu().tolist()); labels.extend(batch['label'].tolist()); ids.extend(batch['slide_id'])
            label=int(batch['label'][0])

            if label not in attention_examples:
                attention_examples[label]={'attention':attention[0,0].cpu().numpy(),'coords':None if batch['coords'][0] is None else batch['coords'][0].numpy(),'slide_id':batch['slide_id'][0]}

                arrays={'attention':attention_examples[label]['attention'],'slide_id':batch['slide_id'][0]}

                if batch['coords'][0] is not None: arrays['coords']=attention_examples[label]['coords']

                np.savez_compressed(out/f'attention_class{label}.npz',**arrays)

    y=np.array(labels); p=np.array(probabilities); pred=p.argmax(1); cm=confusion_matrix(y,pred,labels=list(range(conf.n_class))); tp=cm.diagonal(); fn=cm.sum(1)-tp; fp=cm.sum(0)-tp; tn=cm.sum()-tp-fn-fp
    aucs=[]; aps=[]


    for c in range(conf.n_class):
        binary=(y==c)
        aucs.append(float(roc_auc_score(binary,p[:,c])) if binary.any() and not binary.all() else None)
        aps.append(float(average_precision_score(binary,p[:,c])) if binary.any() else None)


    metrics={'accuracy':float((y==pred).mean()),'auc_macro_ovr':float(np.mean(aucs)) if all(v is not None for v in aucs) else None,'auc_per_class':aucs,'auprc_per_class':aps,'auprc_macro_ovr':float(np.mean(aps)) if all(v is not None for v in aps) else None,'recall_per_class':np.divide(tp,tp+fn,out=np.zeros_like(tp,dtype=float),where=tp+fn!=0).tolist(),'specificity_per_class':np.divide(tn,tn+fp,out=np.zeros_like(tp,dtype=float),where=tn+fp!=0).tolist(),'confusion_matrix':cm.tolist(),'f1_macro':float(f1_score(y,pred,labels=list(range(conf.n_class)),average='macro',zero_division=0)),'best_epoch':best_epoch+1,'epoch_seconds_mean':float(np.mean([r['seconds'] for r in history]))}
    metrics['test_loss']=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean())


    if conf.n_class==2:
        metrics.update(sensitivity=metrics['recall_per_class'][1],specificity=metrics['specificity_per_class'][1],auprc_positive=aps[1])
    atomic_json(out/'predictions.json',{'ids':ids,'labels':labels,'probabilities':probabilities}); atomic_json(out/'metrics.json',metrics)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    fig,ax=plt.subplots(figsize=(6,4)); ax.plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label='train objective'); ax.plot([r['epoch'] for r in history],[r['val_loss'] for r in history],label='validation CE'); ax.set(xlabel='Epoch',ylabel='Loss'); ax.legend(); fig.tight_layout(); fig.savefig(out/'loss_curves.png',dpi=220); fig.savefig(ROOT/'figures'/(spec['id']+'_loss.png'),dpi=220); plt.close(fig)


    for label,example in attention_examples.items():
        coords=example['coords']

        if coords is None: continue
        weights=example['attention']; fig,ax=plt.subplots(figsize=(6,5))

        # Midrank percentiles describe within-slide attention only.

        from scipy.stats import rankdata

        percentile=100*(rankdata(weights,method='average')-1)/max(1,len(weights)-1)
        image=ax.scatter(coords[:,0],coords[:,1],c=percentile,s=5,cmap='inferno',vmin=0,vmax=100,rasterized=True)
        ax.invert_yaxis(); ax.set_aspect('equal'); ax.set(xlabel='Patch x coordinate',ylabel='Patch y coordinate',title=example['slide_id']+' | class '+str(label))
        fig.colorbar(image,ax=ax,label='Attention percentile (not tumor probability)'); fig.tight_layout(); fig.savefig(ROOT/'figures'/(spec['id']+f'_attention_class{label}.png'),dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(5,4)); image=ax.imshow(cm,cmap='Blues'); ax.set(xlabel='Predicted class',ylabel='True class',xticks=range(conf.n_class),yticks=range(conf.n_class)); fig.colorbar(image,ax=ax); fig.tight_layout(); fig.savefig(ROOT/'figures'/(spec['id']+'_confusion.png'),dpi=220); plt.close(fig)



if __name__=='__main__':
    spec=json.loads(Path(sys.argv[1]).read_text()); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True); run(spec,out)

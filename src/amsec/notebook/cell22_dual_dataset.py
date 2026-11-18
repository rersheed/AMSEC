"""Cell 22 — Corrected TRIM-SEC dual-dataset CNN–BiLSTM benchmark (PyTorch).

Refactor of cell 22 of ``Final_phd-Experiment-android-malware_CORRECTED.ipynb``.

Fidelity rules
--------------
* ``PairDataset``, ``CNNBiLSTM``, ``DualInputModel``, ``apply_balancing`` and
  ``evaluate`` are copied verbatim (``tests/test_notebook_fidelity.py`` checks
  their AST against the notebook cell).
* ``run_cell22`` is the cell's top-level script turned into a function. The
  only changes are: dataset/output paths are parameters, progress/log hooks,
  and *additional* artifacts (test probabilities, model weights, MI table,
  history) are written. No step of the procedure is reordered or altered.

Pairing assumption (documented, not altered)
--------------------------------------------
The API-call file and the NF-UQ-NIDS file share **no sample identifier**.
For each API row, one NIDS row with the *same binary label* is drawn without
replacement (``SEED=42``). The paired NIDS row therefore carries the API
row's label by construction; the static branch learns NIDS separability, not
a property of the same application. See ``docs/NOTEBOOK_EXPERIMENT_AUDIT.md``.
"""
from __future__ import annotations

import gc
import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SEED = 42
API_FILE = "dynamic_api_call_sequence_per_malware_100_0_306.csv"
NIDS_FILE = "NF-UQ-NIDS-Dataset.csv"
ALL_METHODS = [
    "no_balancing",
    "smote",
    "adasyn",
    "smote_enn",
    "smote_tomek",
    "smote_ncr",
    "adasyn_enn",
    "adasyn_tomek",
    "adasyn_ncr",
]
# What each method key actually executes in cell 22 (names can mislead).
METHOD_SEMANTICS = {
    "no_balancing": "No resampling",
    "smote": "SMOTE(k_neighbors=3)",
    "adasyn": "ADASYN()",
    "smote_enn": "SMOTEENN()",
    "smote_tomek": "SMOTETomek()",
    "smote_ncr": "SMOTE() → NeighbourhoodCleaningRule()",
    "adasyn_enn": "ADASYN() → SMOTEENN()  (not ADASYN→ENN)",
    "adasyn_tomek": "ADASYN() → SMOTETomek()  (not ADASYN→Tomek)",
    "adasyn_ncr": "ADASYN() → NeighbourhoodCleaningRule()",
}
NOTEBOOK_HYPERPARAMETERS = {
    "seed": SEED,
    "torch_num_threads": 3,
    "api_benign_quota": 200,
    "api_malware_quota": 1000,
    "sequence_length": 80,
    "nids_nrows": 50_000,
    "split": "70/15/15 stratified (train_test_split 0.30 then 0.50)",
    "mi_sample_cap": 5000,
    "n_static_features": 5,
    "embedding_dim": 128,
    "lstm_hidden": 128,
    "lstm_layers": 2,
    "dropout": 0.72,
    "train_noise_std": 0.08,
    "batch_size": 512,
    "optimizer": "Adam(lr=0.0018, weight_decay=8e-4)",
    "scheduler": "ReduceLROnPlateau(min, patience=1, factor=0.5) on val loss",
    "epochs": 3,
    "grad_clip_norm": 1.0,
    "model_selection": "best validation accuracy across epochs",
    "train_shuffle": "only for no_balancing (resampled sets are NOT shuffled)",
}


# ---------------------------------------------------------------------------
# Verbatim notebook definitions (AST-checked against cell 22)
# ---------------------------------------------------------------------------
class PairDataset(Dataset):
    def __init__(self,X,S,y,noise=False): self.X=torch.tensor(X,dtype=torch.float32); self.S=torch.tensor(S,dtype=torch.long); self.y=torch.tensor(y,dtype=torch.long); self.noise=noise
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        x=self.X[i]
        if self.noise: x=x+torch.randn_like(x)*.08
        return x,self.S[i],self.y[i]
class CNNBiLSTM(nn.Module):
    def __init__(self,vocab_size,hidden=128):
        super().__init__(); self.embedding=nn.Embedding(vocab_size,128,padding_idx=0); self.conv1=nn.Sequential(nn.Conv1d(128,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2)); self.conv2=nn.Sequential(nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2)); self.lstm=nn.LSTM(128,hidden,2,batch_first=True,bidirectional=True,dropout=.3); self.attn=nn.Linear(hidden*2,1); self.drop=nn.Dropout(.72); self.fc=nn.Linear(hidden*2,hidden)
    def forward(self,x):
        x=self.embedding(x).transpose(1,2); x=self.conv2(self.conv1(x)).transpose(1,2); h,_=self.lstm(x); w=torch.softmax(self.attn(h),1); c=(w*h).sum(1); return self.drop(torch.relu(self.fc(c)))
class DualInputModel(nn.Module):
    def __init__(self,static_dim,vocab_size):
        super().__init__(); self.static=nn.Sequential(nn.Linear(static_dim,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.72),nn.Linear(128,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.72),nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.72)); self.seq=CNNBiLSTM(vocab_size); self.fuse=nn.Sequential(nn.Linear(256,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(.72),nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.72),nn.Linear(256,2))
    def forward(self,x,s): return self.fuse(torch.cat([self.static(x),self.seq(s)],1))

def apply_balancing(X,S,y,method):
    if method=='no_balancing': return X,S,y
    sampler={'smote':SMOTE(random_state=SEED,k_neighbors=3),'adasyn':ADASYN(random_state=SEED),'smote_enn':SMOTEENN(random_state=SEED),'smote_tomek':SMOTETomek(random_state=SEED),'smote_ncr':make_pipeline(SMOTE(random_state=SEED),NeighbourhoodCleaningRule()),'adasyn_enn':make_pipeline(ADASYN(random_state=SEED),SMOTEENN(random_state=SEED)),'adasyn_tomek':make_pipeline(ADASYN(random_state=SEED),SMOTETomek(random_state=SEED)),'adasyn_ncr':make_pipeline(ADASYN(random_state=SEED),NeighbourhoodCleaningRule())}[method]
    Xb,yb=sampler.fit_resample(X,y)
    buckets={c:np.where(y==c)[0] for c in (0,1)}; lookup={}
    for i,row in enumerate(X): lookup.setdefault(row.tobytes(),[]).append(i)
    sb=np.empty((len(yb),S.shape[1]),dtype=np.int64); rr=np.random.default_rng(SEED+len(method)+len(yb))
    for i,(row,lab) in enumerate(zip(Xb,yb)):
        inds=lookup.get(row.tobytes()); sb[i]=S[inds[0]] if inds else S[rr.choice(buckets[int(lab)])]
    return Xb.astype(np.float32),sb,yb.astype(np.int64)

def evaluate(model,loader):
    model.eval(); probs=[]; preds=[]; labs=[]
    with torch.no_grad():
        for xb,sb,yb in loader:
            o=model(xb,sb); probs.extend(torch.softmax(o,1)[:,1].numpy()); preds.extend(o.argmax(1).numpy()); labs.extend(yb.numpy())
    probs=np.array(probs); preds=np.array(preds); labs=np.array(labs); cm=confusion_matrix(labs,preds,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    return dict(accuracy=accuracy_score(labs,preds),precision=precision_score(labs,preds,zero_division=0),recall=recall_score(labs,preds,zero_division=0),f1_score=f1_score(labs,preds,zero_division=0),roc_auc=roc_auc_score(labs,probs),balanced_accuracy=balanced_accuracy_score(labs,preds),mcc=matthews_corrcoef(labs,preds),specificity=tn/(tn+fp) if tn+fp else 0.0,fpr=fp/(fp+tn) if fp+tn else 0.0,fnr=fn/(fn+tp) if fn+tp else 0.0,confusion_matrix=cm.tolist())


# ---------------------------------------------------------------------------
# Additive logging helper (does not affect metrics): test-set probabilities.
# ---------------------------------------------------------------------------
def _collect_test_outputs(model, loader) -> tuple[np.ndarray, np.ndarray]:
    # Creating a DataLoader iterator draws a base seed from torch's global RNG
    # (even with shuffle=False). Save/restore the state so this extra pass does
    # not shift the random stream of later methods (verified by
    # scripts/validate_cell22_refactor.py).
    rng_state = torch.get_rng_state()
    try:
        model.eval(); probs = []; labs = []
        with torch.no_grad():
            for xb, sb, yb in loader:
                o = model(xb, sb); probs.extend(torch.softmax(o, 1)[:, 1].numpy()); labs.extend(yb.numpy())
    finally:
        torch.set_rng_state(rng_state)
    return np.asarray(labs, dtype=np.int64), np.asarray(probs, dtype=np.float64)


def run_cell22(
    data_dir: Path | str,
    out_dir: Path | str,
    methods: list[str] | None = None,
    log: Callable[[str], None] = print,
    progress: Callable[[dict], None] | None = None,
    save_models: bool = True,
) -> dict:
    """Execute cell 22 end-to-end. Returns results, metadata and artifacts paths."""
    data_dir = Path(data_dir)
    OUT = Path(out_dir); OUT.mkdir(parents=True, exist_ok=True)
    API_PATH = data_dir / API_FILE
    NIDS_PATH = data_dir / NIDS_FILE
    methods = list(methods or ALL_METHODS)
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown cell-22 methods: {unknown}")

    def emit(payload: dict) -> None:
        if progress:
            progress(payload)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(3)
    log('='*90); log('CORRECTED TRIM-SEC DUAL-DATASET CNN-BiLSTM BENCHMARK'); log('='*90)
    log(f'CPU threads: {torch.get_num_threads()}')

    emit({"phase": "load_api", "message": "Loading API dataset"})
    log('\n[1/7] Loading API dataset (all rows)...')
    api_full=pd.read_csv(API_PATH, low_memory=False)
    api_cols=[c for c in api_full.columns if c.startswith('t_')]
    seq_len=80
    api_labels_full=(api_full['malware'].to_numpy()>0).astype(np.int64)
    rng_api=np.random.default_rng(SEED)
    quota=min(200, int((api_labels_full==0).sum()))
    mal_quota=1000
    mal_quota=min(mal_quota, int((api_labels_full==1).sum()))
    api_take=np.concatenate([rng_api.choice(np.where(api_labels_full==0)[0],size=quota,replace=False), rng_api.choice(np.where(api_labels_full==1)[0],size=mal_quota,replace=False)])
    rng_api.shuffle(api_take)
    api=api_full.iloc[api_take].reset_index(drop=True)
    api_labels=(api['malware'].to_numpy()>0).astype(np.int64)
    raw_seq=api[api_cols].to_numpy(dtype=np.int64)[:,:seq_len]
    log(f'API full rows: {len(api_full)} API rows used in rerun: {len(api)} sequence columns: {len(api_cols)} using: {seq_len}')
    log(f'API labels used: {dict(zip(*np.unique(api_labels,return_counts=True)))}')

    emit({"phase": "load_nids", "message": "Loading NIDS sample (first 50,000 rows)"})
    log('\n[2/7] Loading NIDS sample using notebook design (first 50,000 rows)...')
    sample=pd.read_csv(NIDS_PATH, nrows=50_000, low_memory=False)
    nids_y=(sample['Label'].to_numpy()>0).astype(np.int64)
    numeric_cols=[c for c in sample.select_dtypes(include=[np.number]).columns.tolist() if c!='Label']
    nids_X=sample[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
    log(f'NIDS sampled rows: {len(sample)} numeric features: {nids_X.shape[1]}')
    log(f'NIDS labels: {dict(zip(*np.unique(nids_y,return_counts=True)))}')

    emit({"phase": "pairing", "message": "Label-matched cross-dataset pairing"})
    log('\n[3/7] Building label-matched cross-dataset pairs...')
    rng=np.random.default_rng(SEED)
    nids_idx=np.empty(len(api_labels),dtype=np.int64)
    for cls in (0,1):
        api_count=int((api_labels==cls).sum()); avail=np.where(nids_y==cls)[0]
        if len(avail)<api_count: raise RuntimeError(f'Not enough NIDS class {cls}: {len(avail)} < {api_count}')
        picked=rng.choice(avail,size=api_count,replace=False); api_idx=np.where(api_labels==cls)[0]; rng.shuffle(picked); nids_idx[api_idx]=picked
    paired_static=nids_X[nids_idx]; paired_y=api_labels.copy(); paired_seq=raw_seq.copy()
    log(f'Paired samples: {len(paired_y)}'); log(f'Unique NIDS rows paired: {len(np.unique(nids_idx))}'); log(f'Paired class counts: {dict(zip(*np.unique(paired_y,return_counts=True)))}')

    idx=np.arange(len(paired_y))
    idx_train,idx_temp=train_test_split(idx,test_size=.30,random_state=SEED,stratify=paired_y)
    idx_val,idx_test=train_test_split(idx_temp,test_size=.50,random_state=SEED,stratify=paired_y[idx_temp])
    train_X=paired_static[idx_train]
    mi_sample_idx=np.random.default_rng(SEED).choice(len(train_X), size=min(5000,len(train_X)), replace=False)
    mi=mutual_info_classif(pd.DataFrame(train_X[mi_sample_idx],columns=numeric_cols),paired_y[idx_train][mi_sample_idx],random_state=SEED)
    mi_tbl=pd.DataFrame({'feature':numeric_cols,'mi_score':mi}).sort_values('mi_score',ascending=False)
    selected_features=mi_tbl.head(5)['feature'].tolist(); sel_idx=[numeric_cols.index(x) for x in selected_features]
    scaler=StandardScaler(); Xtr=scaler.fit_transform(paired_static[idx_train][:,sel_idx]).astype(np.float32); Xv=scaler.transform(paired_static[idx_val][:,sel_idx]).astype(np.float32); Xte=scaler.transform(paired_static[idx_test][:,sel_idx]).astype(np.float32)
    Ytr=paired_y[idx_train]; Yv=paired_y[idx_val]; Yte=paired_y[idx_test]; Str=paired_seq[idx_train]; Sv=paired_seq[idx_val]; Ste=paired_seq[idx_test]
    mi_tbl.to_csv(OUT/'nids_mutual_information_scores.csv',index=False)
    log(f'Splits: {len(idx_train)} {len(idx_val)} {len(idx_test)}'); log('Top NIDS features:')
    for k,row in mi_tbl.head(5).reset_index(drop=True).iterrows(): log(f"  {k+1}. {row['feature']}: {row['mi_score']:.6f}")

    unique_tokens=np.unique(raw_seq); unique_tokens=unique_tokens[unique_tokens!=0]; vocab={int(tok):i+1 for i,tok in enumerate(unique_tokens.tolist())}; vocab_size=len(vocab)+1
    def encode(a):
        out=np.zeros_like(a,dtype=np.int64)
        for j in range(a.shape[1]): out[:,j]=[vocab.get(int(v),0) if v!=0 else 0 for v in a[:,j]]
        return out
    Str=encode(Str); Sv=encode(Sv); Ste=encode(Ste)
    log(f'API vocabulary size: {vocab_size}')

    test_outputs: dict[str, dict] = {}

    def train_one(method):
        Xb,Sb,yb=apply_balancing(Xtr,Str,Ytr,method); log(f"  train distribution: benign={sum(yb==0)}, malware={sum(yb==1)}")
        tr=DataLoader(PairDataset(Xb,Sb,yb,True),batch_size=512,shuffle=(method=='no_balancing')); va=DataLoader(PairDataset(Xv,Sv,Yv),batch_size=512); te=DataLoader(PairDataset(Xte,Ste,Yte),batch_size=512)
        model=DualInputModel(Xtr.shape[1],vocab_size); params=sum(p.numel() for p in model.parameters() if p.requires_grad); opt=torch.optim.Adam(model.parameters(),lr=.0018,weight_decay=8e-4); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=1,factor=.5); best=-1; best_state=None; hist=[]; ts=time.perf_counter()
        for ep in range(3):
            model.train(); loss_sum=corr=tot=0
            for xb,sb,yb2 in tr:
                opt.zero_grad(); o=model(xb,sb); loss=nn.functional.cross_entropy(o,yb2); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); loss_sum+=loss.item(); corr+=(o.argmax(1)==yb2).sum().item(); tot+=len(yb2)
            model.eval(); vl=vc=vt=0
            with torch.no_grad():
                for xb,sb,yb2 in va:
                    o=model(xb,sb); vl+=nn.functional.cross_entropy(o,yb2).item(); vc+=(o.argmax(1)==yb2).sum().item(); vt+=len(yb2)
            vl/=len(va); vacc=vc/vt; sched.step(vl); hist.append((ep+1,loss_sum/len(tr),corr/tot,vl,vacc)); log(f"  epoch {ep+1}: train_acc={corr/tot:.4f} val_acc={vacc:.4f}")
            emit({"phase": "train", "method": method, "epoch": ep + 1, "epochs": 3, "val_acc": vacc})
            if vacc>best: best=vacc; best_state={k:v.clone() for k,v in model.state_dict().items()}
        train_sec=time.perf_counter()-ts
        if best_state: model.load_state_dict(best_state)
        ev=evaluate(model,te); ev.update(method=method,parameters=params,train_samples=len(yb),test_samples=len(Yte),training_seconds=train_sec,history=hist)
        # --- additive artifacts (not part of the notebook procedure) ---
        labs, probs = _collect_test_outputs(model, te)
        test_outputs[method] = {"y_true": labs, "proba": probs}
        if save_models:
            (OUT / "models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), OUT / "models" / f"cell22_{method}.pt")
        del model; gc.collect(); return ev

    results=[]; failures={}
    for m in methods:
        log('\n'+'='*90); log(f'RUNNING {m}'); log('='*90)
        emit({"phase": "method", "method": m})
        try:
            r=train_one(m); results.append(r); log(f"RESULT {m}: acc={r['accuracy']:.4f} f1={r['f1_score']:.4f} auc={r['roc_auc']:.4f} bal_acc={r['balanced_accuracy']:.4f}")
        except Exception as e:
            failures[m] = repr(e); log(f'FAILED {m} {e}'); log(traceback.format_exc())

    rows=[]
    for r in results: rows.append({k:r[k] for k in ['method','train_samples','test_samples','accuracy','precision','recall','f1_score','roc_auc','balanced_accuracy','mcc','specificity','fpr','fnr','parameters','training_seconds']})
    res_df=pd.DataFrame(rows)
    if len(res_df):
        res_df=res_df.sort_values(['f1_score','accuracy'],ascending=False).reset_index(drop=True); res_df.insert(0,'rank',range(1,len(res_df)+1))
    res_df.to_csv(OUT/'dual_dataset_balancing_results.csv',index=False)
    metadata={'seed':SEED,'api_rows_full':len(api_full),'api_rows_used':len(api),'nids_rows_loaded':len(sample),'unique_nids_rows_paired':int(len(np.unique(nids_idx))),'selected_features':selected_features,'sequence_length':seq_len,'vocab_size':vocab_size,'epochs':3,'batch_size':512,'methods':methods,'pairing':'one randomly selected NIDS row per API row, conditioned on the shared binary label'}
    with open(OUT/'dual_dataset_run_metadata.json','w') as f: json.dump(metadata,f,indent=2)
    log('\n'+'='*120); log('FINAL RESULTS'); log('='*120)
    if len(res_df): log(res_df.to_string(index=False,float_format=lambda x:f'{x:.5f}'))

    # Additive exports for the dashboard
    pd.DataFrame(
        [{"method": r["method"], "epoch": h[0], "train_loss": h[1], "train_acc": h[2], "val_loss": h[3], "val_acc": h[4]} for r in results for h in r["history"]]
    ).to_csv(OUT / "training_history.csv", index=False)
    pd.DataFrame(
        [{"method": m, "y_true": int(t), "proba_malware": float(p)} for m, d in test_outputs.items() for t, p in zip(d["y_true"], d["proba"])]
    ).to_csv(OUT / "test_predictions.csv", index=False)
    confusion = {r["method"]: r["confusion_matrix"] for r in results}
    (OUT / "confusion_matrices.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    split = {
        "train_idx": idx_train.tolist(), "val_idx": idx_val.tolist(), "test_idx": idx_test.tolist(),
        "api_rows_taken": api_take.tolist(), "nids_rows_paired": nids_idx.tolist(),
    }
    (OUT / "split_indices.json").write_text(json.dumps(split), encoding="utf-8")
    if save_models:
        import joblib
        joblib.dump({"scaler": scaler, "selected_features": selected_features, "vocab": vocab, "seq_len": seq_len, "vocab_size": vocab_size}, OUT / "models" / "cell22_preprocessing.joblib")

    return {"results": res_df, "metadata": metadata, "failures": failures, "out_dir": str(OUT)}


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Run notebook cell 22")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--methods", default=os.environ.get("METHODS", ",".join(ALL_METHODS)))
    a = ap.parse_args()
    run_cell22(a.data_dir, a.out_dir, a.methods.split(","))

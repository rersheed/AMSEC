import os, gc, json, time, random, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(3)
BASE=Path('/mnt/data/experiments_ii/Experiments II')
OUT=Path('/mnt/data/experiments_ii/results_dual_dataset'); OUT.mkdir(parents=True, exist_ok=True)
API_PATH=BASE/'dynamic_api_call_sequence_per_malware_100_0_306.csv'
NIDS_PATH=BASE/'NF-UQ-NIDS-Dataset.csv'
print('='*90); print('CORRECTED TRIM-SEC DUAL-DATASET CNN-BiLSTM BENCHMARK'); print('='*90)
print('CPU threads:', torch.get_num_threads())

print('\n[1/7] Loading API dataset (all rows)...')
api_full=pd.read_csv(API_PATH, low_memory=False)
api_cols=[c for c in api_full.columns if c.startswith('t_')]
seq_len=80
api_labels_full=(api_full['malware'].to_numpy()>0).astype(np.int64)
# Deterministic minority-floor subset for a CPU-feasible rerun while retaining both source datasets.
rng_api=np.random.default_rng(SEED)
quota=min(200, int((api_labels_full==0).sum()))
mal_quota=1000
mal_quota=min(mal_quota, int((api_labels_full==1).sum()))
api_take=np.concatenate([rng_api.choice(np.where(api_labels_full==0)[0],size=quota,replace=False), rng_api.choice(np.where(api_labels_full==1)[0],size=mal_quota,replace=False)])
rng_api.shuffle(api_take)
api=api_full.iloc[api_take].reset_index(drop=True)
api_labels=(api['malware'].to_numpy()>0).astype(np.int64)
raw_seq=api[api_cols].to_numpy(dtype=np.int64)[:,:seq_len]
print('API full rows:',len(api_full),'API rows used in rerun:',len(api),'sequence columns:',len(api_cols),'using:',seq_len)
print('API labels used:',dict(zip(*np.unique(api_labels,return_counts=True))))

print('\n[2/7] Loading NIDS sample using notebook design (first 50,000 rows)...')
sample=pd.read_csv(NIDS_PATH, nrows=50_000, low_memory=False)
nids_y=(sample['Label'].to_numpy()>0).astype(np.int64)
numeric_cols=[c for c in sample.select_dtypes(include=[np.number]).columns.tolist() if c!='Label']
nids_X=sample[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
print('NIDS sampled rows:',len(sample),'numeric features:',nids_X.shape[1])
print('NIDS labels:',dict(zip(*np.unique(nids_y,return_counts=True))))

print('\n[3/7] Building label-matched cross-dataset pairs...')
rng=np.random.default_rng(SEED)
nids_idx=np.empty(len(api_labels),dtype=np.int64)
for cls in (0,1):
    api_count=int((api_labels==cls).sum()); avail=np.where(nids_y==cls)[0]
    if len(avail)<api_count: raise RuntimeError(f'Not enough NIDS class {cls}: {len(avail)} < {api_count}')
    picked=rng.choice(avail,size=api_count,replace=False); api_idx=np.where(api_labels==cls)[0]; rng.shuffle(picked); nids_idx[api_idx]=picked
paired_static=nids_X[nids_idx]; paired_y=api_labels.copy(); paired_seq=raw_seq.copy()
print('Paired samples:',len(paired_y)); print('Unique NIDS rows paired:',len(np.unique(nids_idx))); print('Paired class counts:',dict(zip(*np.unique(paired_y,return_counts=True))))

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
print('Splits:',len(idx_train),len(idx_val),len(idx_test)); print('Top NIDS features:');
for k,row in mi_tbl.head(5).reset_index(drop=True).iterrows(): print(f"  {k+1}. {row['feature']}: {row['mi_score']:.6f}")

unique_tokens=np.unique(raw_seq); unique_tokens=unique_tokens[unique_tokens!=0]; vocab={int(tok):i+1 for i,tok in enumerate(unique_tokens.tolist())}; vocab_size=len(vocab)+1
def encode(a):
    out=np.zeros_like(a,dtype=np.int64)
    for j in range(a.shape[1]): out[:,j]=[vocab.get(int(v),0) if v!=0 else 0 for v in a[:,j]]
    return out
Str=encode(Str); Sv=encode(Sv); Ste=encode(Ste)
print('API vocabulary size:',vocab_size)

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

def train_one(method):
    Xb,Sb,yb=apply_balancing(Xtr,Str,Ytr,method); print(f"  train distribution: benign={sum(yb==0)}, malware={sum(yb==1)}")
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
        vl/=len(va); vacc=vc/vt; sched.step(vl); hist.append((ep+1,loss_sum/len(tr),corr/tot,vl,vacc)); print(f"  epoch {ep+1}: train_acc={corr/tot:.4f} val_acc={vacc:.4f}")
        if vacc>best: best=vacc; best_state={k:v.clone() for k,v in model.state_dict().items()}
    train_sec=time.perf_counter()-ts
    if best_state: model.load_state_dict(best_state)
    ev=evaluate(model,te); ev.update(method=method,parameters=params,train_samples=len(yb),test_samples=len(Yte),training_seconds=train_sec,history=hist)
    del model; gc.collect(); return ev

methods=os.environ.get('METHODS','no_balancing,smote,adasyn,smote_enn,smote_tomek,smote_ncr,adasyn_enn,adasyn_tomek,adasyn_ncr').split(',')
results=[]
for m in methods:
    print('\n'+'='*90); print('RUNNING',m); print('='*90)
    try:
        r=train_one(m); results.append(r); print(f"RESULT {m}: acc={r['accuracy']:.4f} f1={r['f1_score']:.4f} auc={r['roc_auc']:.4f} bal_acc={r['balanced_accuracy']:.4f}")
    except Exception as e: print('FAILED',m,e); traceback.print_exc()

rows=[]
for r in results: rows.append({k:r[k] for k in ['method','train_samples','test_samples','accuracy','precision','recall','f1_score','roc_auc','balanced_accuracy','mcc','specificity','fpr','fnr','parameters','training_seconds']})
res_df=pd.DataFrame(rows).sort_values(['f1_score','accuracy'],ascending=False).reset_index(drop=True); res_df.insert(0,'rank',range(1,len(res_df)+1)); res_df.to_csv(OUT/f'dual_dataset_balancing_results_{os.environ.get("METHODS","all")}.csv'.replace(',', '_'),index=False)
with open(OUT/f'dual_dataset_run_metadata_{os.environ.get("METHODS","all")}.json'.replace(',', '_'),'w') as f: json.dump({'seed':SEED,'api_rows_full':len(api_full),'api_rows_used':len(api),'nids_rows_loaded':len(sample),'unique_nids_rows_paired':int(len(np.unique(nids_idx))),'selected_features':selected_features,'sequence_length':seq_len,'vocab_size':vocab_size,'epochs':3,'batch_size':512,'methods':methods,'pairing':'one randomly selected NIDS row per API row, conditioned on the shared binary label'},f,indent=2)
print('\n'+'='*120); print('FINAL RESULTS'); print('='*120); print(res_df.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
fig,ax=plt.subplots(figsize=(12,6)); x=np.arange(len(res_df)); w=.38; ax.bar(x-w/2,res_df.accuracy,w,label='Accuracy'); ax.bar(x+w/2,res_df.f1_score,w,label='F1'); ax.set_xticks(x); ax.set_xticklabels(res_df.method,rotation=35,ha='right'); ax.set_ylim(0,1.02); ax.set_ylabel('Score'); ax.set_title('Corrected Dual-Dataset CNN-BiLSTM: Accuracy and F1'); ax.legend(); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(OUT/'accuracy_f1_comparison.png',dpi=180); plt.close(fig)
print('Saved:',OUT)

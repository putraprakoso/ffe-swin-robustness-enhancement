"""FFE Publication 2026 - shared, reproducible utilities for Google Colab.
Source design: user's 2026 thesis/notebooks. This is a revised research pipeline,
not a bit-identical reproduction of thesis figures or metrics.
"""
from __future__ import annotations
import os, re, io, json, hashlib, random, warnings, zipfile, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('FFE_ROOT', '/content/drive/MyDrive/tugas_akhir/FFE_Publication_2026'))
ZIP = ROOT / '01_data/raw/xzyx7pbr3w-1.zip'
LOCAL = Path('/content/ffe_publication_images')
CLASS_NAMES = ['Highly Fresh','Fresh','Not Fresh']
LABELS = {k:i for i,k in enumerate(CLASS_NAMES)}
IMG_EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
SEED=42


def paths():
    result={
      'root':ROOT, 'raw':ROOT/'01_data/raw', 'manifests':ROOT/'01_data/manifests',
      'splits':ROOT/'03_artifacts/splits', 'features':ROOT/'03_artifacts/features',
      'checkpoints':ROOT/'03_artifacts/checkpoints', 'models':ROOT/'03_artifacts/models',
      'metrics':ROOT/'04_results/metrics', 'figures':ROOT/'04_results/figures',
      'xai':ROOT/'04_results/xai', 'logs':ROOT/'06_logs'
    }
    for k,p in result.items():
        if k!='root':p.mkdir(parents=True,exist_ok=True)
    return result


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    except ImportError:pass


def ensure_dataset_local(force=False):
    """Extract safely to local Colab SSD; returns local root. ZIP remains on Drive."""
    if not ZIP.exists():raise FileNotFoundError(f'Dataset ZIP missing: {ZIP}')
    marker=LOCAL/'.ffe_extracted_ok'
    digest=f'{ZIP.stat().st_size}-{int(ZIP.stat().st_mtime)}'
    if not force and marker.exists() and marker.read_text()==digest:return LOCAL
    import shutil
    if LOCAL.exists():shutil.rmtree(LOCAL)
    LOCAL.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ZIP) as archive:
        bad=archive.testzip()
        if bad:raise ValueError(f'ZIP corrupt at: {bad}')
        files=archive.infolist()
        if sum(i.file_size for i in files)>6_000_000_000:
            raise ValueError('Uncompressed ZIP exceeds 6GB safety limit')
        for item in files:
            target=(LOCAL/item.filename).resolve()
            if not target.is_relative_to(LOCAL.resolve()):
                raise ValueError(f'Unsafe path traversal in ZIP: {item.filename}')
            if item.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(item) as src, open(target,'wb') as dst:shutil.copyfileobj(src,dst)
    marker.write_text(digest)
    return LOCAL


def parse_class_folder(folder):
    """Return species,freshness for FFE folders; reject unknown labels."""
    clean=re.sub(r'[_]+',' ',folder).strip()
    m=re.match(r'^(.*?)\s+-\s+(Highly Fresh|Not Fresh|Fresh)\s*$',clean,re.I)
    if not m:return None
    fresh=m.group(2).lower()
    label={'highly fresh':'Highly Fresh','not fresh':'Not Fresh','fresh':'Fresh'}[fresh]
    return m.group(1).strip(), label


def image_records(root=None):
    """Create stable relative-path manifest; never silently skip unreadable images."""
    import cv2
    from PIL import Image
    root = LOCAL if root is None else Path(root)
    rows=[]; failed=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:continue
        parsed=None
        for parent in p.parents:
            if parent==root.parent:break
            parsed=parse_class_folder(parent.name)
            if parsed:break
        if not parsed:failed.append((str(p),'class_folder_not_found'));continue
        try:
            data=p.read_bytes(); digest=hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            image=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError('OpenCV cannot decode image')
            h,w=image.shape[:2]
            small=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(32,32))
            dct=cv2.dct(np.float32(small))[:8,:8]
            med=np.median(dct[1:,:].ravel())
            phash=format(sum(int(v>=med)<<i for i,v in enumerate(dct.ravel())),'016x')
            rel=p.relative_to(root).as_posix()
            rows.append(dict(sample_id=hashlib.sha256(rel.encode()).hexdigest()[:20],rel_path=rel,
                             species=parsed[0],freshness=parsed[1],label=LABELS[parsed[1]],
                             sha256=digest,phash=phash,width=w,height=h,bytes=len(data)))
        except Exception as e:failed.append((str(p),str(e)))
    if failed:
        pd.DataFrame(failed,columns=['path','reason']).to_csv(paths()['manifests']/'unreadable_or_unparsed.csv',index=False)
        raise ValueError(f'{len(failed)} images failed audit; see unreadable_or_unparsed.csv')
    if not rows:raise ValueError('No FFE images recognized: inspect ZIP class folder names')
    df=pd.DataFrame(rows).sort_values('rel_path').reset_index(drop=True)
    if df.sample_id.duplicated().any():raise ValueError('Duplicate sample_id')
    if df.species.nunique()!=8 or df.label.nunique()!=3:
        raise ValueError(f'Unexpected taxonomy: {df.species.nunique()} species and {df.label.nunique()} labels; audit archive')
    return df


def load_manifest():
    f=paths()['manifests']/'dataset_manifest.csv'
    if not f.exists():raise FileNotFoundError('Run notebook 00 first')
    return pd.read_csv(f,dtype={'sample_id':'string','sha256':'string','phash':'string'})


def load_split(fold,part):
    f=paths()['splits']/f'fold_{fold:02d}_{part}.csv'
    if not f.exists():raise FileNotFoundError(f'Missing split {f}: run notebook 00')
    return pd.read_csv(f,dtype={'sample_id':'string'})


def prepare_splits(df,n_splits=5,seed=SEED):
    from sklearn.model_selection import StratifiedGroupKFold
    """Outer test folds and inner train/validation; exact duplicates stay grouped."""
    if df.groupby('sha256')['label'].nunique().max()>1:
        warnings.warn('Identical image bytes appear with contradictory freshness labels; audit labels.')
    df=df.copy().reset_index(drop=True)
    strata=df.species.astype(str)+'__'+df.freshness.astype(str)
    groups=df.sha256.astype(str)
    outer=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    all_tests=[]
    for fold,(train_val,test) in enumerate(outer.split(df,strata,groups),1):
        sub=df.iloc[train_val].reset_index(drop=True)
        strat2=sub.species.astype(str)+'__'+sub.freshness.astype(str)
        inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed+fold)
        a,b=next(iter(inner.split(sub,strat2,sub.sha256)))
        partitions={'train':sub.iloc[a].copy(),'val':sub.iloc[b].copy(),'test':df.iloc[test].copy()}
        ids=[set(s.sample_id) for s in partitions.values()]
        hashes=[set(s.sha256) for s in partitions.values()]
        assert all(not ids[i]&ids[j] and not hashes[i]&hashes[j] for i in range(3) for j in range(i+1,3))
        for part,data in partitions.items():
            data.to_csv(paths()['splits']/f'fold_{fold:02d}_{part}.csv',index=False)
        all_tests.extend(df.iloc[test].sample_id.to_list())
        print(f'Fold {fold}: train={len(sub.iloc[a])} val={len(sub.iloc[b])} test={len(test)}')
    assert len(all_tests)==len(df) and len(set(all_tests))==len(df)
    return True


def image_path(rel):return LOCAL/str(rel)


def handcrafted_feature_names():
    return [f'{ch}_{stat}' for ch in ('L','a','b') for stat in ('mean','std','skew','kurt')]+['cvr_a_b']+[f'lbp_{i}' for i in range(10)]+[f'hue_hist_{i}' for i in range(16)]+['sharpness']


def extract_handcrafted(bgr):
    import cv2
    from scipy.stats import skew,kurtosis
    from skimage.feature import local_binary_pattern
    bgr=cv2.resize(bgr,(224,224),interpolation=cv2.INTER_AREA)
    lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB).astype(np.float64)
    out=[]
    for ch in range(3):
        values=lab[:,:,ch].ravel()
        out += [float(np.mean(values)),float(np.std(values)),float(np.nan_to_num(skew(values))),
                float(np.nan_to_num(kurtosis(values)))]
    out.append(float(np.var(lab[:,:,1])/(np.var(lab[:,:,2])+1e-8)))
    # scikit-image LBP precision on integer pixels; match thesis P8 R1 uniform.
    lbp=local_binary_pattern(lab[:,:,2].astype(np.uint8),P=8,R=1,method='uniform')
    hist=np.histogram(lbp,bins=10,range=(0,10),density=True)[0]
    out.extend(np.nan_to_num(hist).tolist())
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    hue=cv2.calcHist([hsv],[0],None,[16],[0,180]).ravel()
    hue=hue/max(float(np.linalg.norm(hue)),1e-12)
    out.extend(hue.tolist())
    sharp=float(cv2.Laplacian(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
    out.append(sharp)
    assert len(out)==40
    arr=np.nan_to_num(np.asarray(out,dtype=np.float32),nan=0,posinf=0,neginf=0)
    return arr


def save_pack(file,X,df):
    file=Path(file);file.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(file,X=np.asarray(X,dtype=np.float32),y=df.label.to_numpy(dtype=np.int64),
                        ids=df.sample_id.to_numpy(dtype=str))


def read_pack(file,expected_df=None):
    with np.load(file,allow_pickle=False) as z:
        result={k:z[k] for k in ('X','y','ids')}
    if expected_df is not None:
        assert list(result['ids'])==expected_df.sample_id.astype(str).tolist(),f'Misaligned sample IDs: {file}'
        assert np.array_equal(result['y'],expected_df.label.to_numpy()),f'Misaligned labels: {file}'
    assert np.isfinite(result['X']).all(),f'Nonfinite features {file}'
    return result


def get_transform(training=False):
    from torchvision import transforms
    from PIL import Image
    # Replaces Albumentations CLAHE from thesis with reproducible PIL equalize (not method-identical).
    aug=[transforms.Resize((224,224))]
    if training:
        aug.extend([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.1),
                    transforms.RandomApply([transforms.RandomEqualize(p=1.0)],p=0.3)])
    aug.extend([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    return transforms.Compose(aug)


class ImageDataset:
    """PyTorch Dataset interface (duck-typed)."""
    def __init__(self,df,transform):self.df=df.reset_index(drop=True);self.transform=transform
    def __len__(self):return len(self.df)
    def __getitem__(self,idx):
        from PIL import Image
        import torch
        row=self.df.iloc[idx]
        with Image.open(image_path(row.rel_path)) as im:
            im=im.convert('RGB');tensor=self.transform(im)
        return tensor,torch.tensor(int(row.label),dtype=torch.long)


def make_model(pretrained=False,num_classes=3):
    import timm
    return timm.create_model('swin_tiny_patch4_window7_224',pretrained=pretrained,num_classes=num_classes,
                             **({'global_pool':'avg'} if num_classes==0 else {}))


def extract_feature_checkpoint(checkpoint,device):
    import torch
    m=make_model(pretrained=False,num_classes=0)
    obj=torch.load(checkpoint,map_location='cpu',weights_only=False)
    sd=obj.get('state_dict',obj)
    sd={k:v for k,v in sd.items() if not k.startswith('head.')}
    incompatible=m.load_state_dict(sd,strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f'Unexpected Swin weights: {incompatible.unexpected_keys}')
    return m.to(device).eval()


def predict_logits(net,Xd,Xh=None,bs=512,device=None):
    import torch
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(Xd),bs):
            a=torch.from_numpy(np.asarray(Xd[start:start+bs],dtype=np.float32)).to(device)
            if Xh is None:logits=net(a)
            else:
                b=torch.from_numpy(np.asarray(Xh[start:start+bs],dtype=np.float32)).to(device)
                logits=net(a,b)
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def save_prediction(file,df,preds,prob,model,fold):
    data=pd.DataFrame({'sample_id':df.sample_id,'fold':fold,'model':model,'y_true':df.label,
                       'y_pred':np.asarray(preds,dtype=int),'species':df.species})
    for i in range(3):data[f'p_{i}']=np.asarray(prob)[:,i]
    data.to_csv(file,index=False)
    return data


def metric_row(y,pred):
    from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,balanced_accuracy_score
    return {'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
            'mcc':matthews_corrcoef(y,pred),'balanced_acc':balanced_accuracy_score(y,pred)}

class DeepOnlyNetFactory:
    @staticmethod
    def create(deep_dim=768):
        import torch.nn as nn
        return nn.Sequential(nn.Linear(deep_dim,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.3),
                             nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,3))


def make_fusion_net(hand_dim=40,deep_dim=768):
    import torch.nn as nn
    class FusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.deep_proj=nn.Sequential(nn.Linear(deep_dim,256),nn.BatchNorm1d(256),nn.ReLU())
            self.hand_proj=nn.Sequential(nn.Linear(hand_dim,64),nn.BatchNorm1d(64),nn.ReLU())
            self.fusion=nn.Sequential(nn.Linear(320,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.3),
                                      nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,3))
        def forward(self,xd,xh):
            return self.fusion(__import__('torch').cat([self.deep_proj(xd),self.hand_proj(xh)],dim=1))
    return FusionNet()


def test_group_integrity():
    """No SHA256-identical samples between partitions and each sample once in outer test."""
    tests=[]
    for fold in range(1,6):
        a,b,c=[load_split(fold,p) for p in ('train','val','test')]
        for x,y in ((a,b),(a,c),(b,c)):
            assert set(x.sha256).isdisjoint(set(y.sha256))
        tests.extend(c.sample_id.to_list())
    df=load_manifest()
    assert len(tests)==len(df) and len(set(tests))==len(df)
    return True

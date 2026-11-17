# LayerCAM maps of the seed-11 models on the validation images; fills runs/seed11_layercam.
import os

STORAGE = os.environ.get("MSLS_STORAGE", "storage")
SCRATCH = os.environ.get("MSLS_SCRATCH", "scratch")


def display(obj):
    print(obj)


import numpy as np
import pandas as pd

print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("Environment: OK")


import json, os, random, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageClassification
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

CONFIG = {
    'seed': 11,
    'img_size': 384,
    'batch_size': 4,
    'num_workers': 2,
    'manifest': STORAGE + '/msls_vmamba/data/clean_protocol_v1/plantseg_optimization_validation.csv',
    'class_mapping': STORAGE + '/msls_vmamba/data/class_mapping.json',
    'plantseg_zip': STORAGE + '/plantseg.zip',
    'baseline_checkpoint': STORAGE + '/msls_vmamba/runs/clean_protocol_v1_384/vmamba_msls_11_55_v1/cleanv1_vmamba_baseline_strong_384_seed11/best_model.pth',
    'current_checkpoint': STORAGE + '/msls_vmamba/runs/clean_protocol_v1_384/vmamba_msls_11_55_v1/cleanv1_current_full_strong_384_seed11/best_model.pth',
    'model_repo': 'saurabhati/VMamba_ImageNet_82.6',
    'model_dir': SCRATCH + '/vmamba_model',
    'cache_root': SCRATCH + '/vmamba_layercam_validation',
    'output_root': STORAGE + '/msls_vmamba/runs/seed11_analyses/paired_layercam_seed11_v1',
}

assert torch.cuda.is_available(), 'Select a GPU runtime.'
DEVICE='cuda'
for key in ('manifest','class_mapping','plantseg_zip','baseline_checkpoint','current_checkpoint'):
    assert Path(CONFIG[key]).exists(), f'Missing {key}: {CONFIG[key]}'
assert 'validation' in CONFIG['manifest'].lower()
assert 'official_test' not in CONFIG['manifest'].lower()
assert 'plantdoc' not in CONFIG['manifest'].lower()

random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
OUT=Path(CONFIG['output_root']); OUT.mkdir(parents=True,exist_ok=True)
print('GPU:',torch.cuda.get_device_name(0))
print('Guard: optimization-validation only; no official test or PlantDoc.')


with open(CONFIG['class_mapping'],encoding='utf-8') as f: mapping=json.load(f)
CLASS_NAMES=list(mapping['class_names']); CLASS_TO_IDX={n:i for i,n in enumerate(CLASS_NAMES)}
NUM_CLASSES=len(CLASS_NAMES); assert NUM_CLASSES==16

frame=pd.read_csv(CONFIG['manifest'])
assert len(frame)==350 and frame.record_id.is_unique
frame['label']=frame.harmonized_class.map(CLASS_TO_IDX).astype(int)

cache=Path(CONFIG['cache_root']); cache.mkdir(parents=True,exist_ok=True)
local_images=[]; local_masks=[]
with zipfile.ZipFile(CONFIG['plantseg_zip']) as z:
    members=set(z.namelist())
    for row in frame.itertuples(index=False):
        assert row.image_reference in members, row.image_reference
        assert row.mask_reference in members, row.mask_reference
        ip=cache/row.image_reference; mp=cache/row.mask_reference
        ip.parent.mkdir(parents=True,exist_ok=True); mp.parent.mkdir(parents=True,exist_ok=True)
        if not ip.exists():
            with z.open(row.image_reference) as s, ip.open('wb') as d: shutil.copyfileobj(s,d)
        if not mp.exists():
            with z.open(row.mask_reference) as s, mp.open('wb') as d: shutil.copyfileobj(s,d)
        local_images.append(str(ip)); local_masks.append(str(mp))
frame['local_image']=local_images; frame['local_mask']=local_masks
print('Validation pairs cached:',len(frame))


def resize_and_pad(image,size,interpolation,fill):
    w,h=image.size; scale=float(size)/max(w,h); nw=max(1,round(w*scale)); nh=max(1,round(h*scale))
    image=TF.resize(image,[nh,nw],interpolation=interpolation,antialias=(interpolation!=InterpolationMode.NEAREST))
    left=(size-nw)//2; right=size-nw-left; top=(size-nh)//2; bottom=size-nh-top
    return TF.pad(image,[left,top,right,bottom],fill=fill)

class EvalSet(Dataset):
    def __init__(self,table): self.table=table.reset_index(drop=True)
    def __len__(self): return len(self.table)
    def __getitem__(self,i):
        row=self.table.iloc[i]
        with Image.open(row.local_image) as f: image=f.convert('RGB')
        with Image.open(row.local_mask) as f: mask=f.convert('L')
        image=resize_and_pad(image,384,InterpolationMode.BICUBIC,(124,116,104))
        mask=resize_and_pad(mask,384,InterpolationMode.NEAREST,0)
        shown=np.asarray(image).copy()
        image=TF.normalize(TF.to_tensor(image),[.485,.456,.406],[.229,.224,.225])
        mask=(TF.pil_to_tensor(mask).float()>0).float()
        return image,mask,int(row.label),row.record_id,shown

loader=DataLoader(EvalSet(frame),batch_size=CONFIG['batch_size'],shuffle=False,
                  num_workers=CONFIG['num_workers'],pin_memory=True)
print('Batches:',len(loader))


if not os.path.exists(os.path.join(CONFIG['model_dir'],'config.json')):
    snapshot_download(repo_id=CONFIG['model_repo'],local_dir=CONFIG['model_dir'],revision='3db342a1b00a91c350dca4df93a1645ad2ac9493')

def pretrained_vmamba():
    return AutoModelForImageClassification.from_pretrained(
        CONFIG['model_dir'],trust_remote_code=True,local_files_only=True).vmamba

class VMambaBaseline(nn.Module):
    def __init__(self,n):
        super().__init__(); self.vmamba=pretrained_vmamba(); self.classifier=nn.Linear(self.vmamba.num_features,n)
    def forward(self,x):
        x=self.vmamba.patch_embeddings(x)
        for layer in self.vmamba.layers: x=layer(x)
        x=self.vmamba.norm(x); vector=self.vmamba.avgpool(x).flatten(1)
        return self.classifier(vector)

class NativeMeanFusionHead(nn.Module):
    NAMES=('fine','mid','coarse')
    def __init__(self,channels=(192,384,768),hidden=128):
        super().__init__()
        self.projections=nn.ModuleList([nn.Sequential(nn.Conv2d(c,hidden,1,bias=False),nn.GroupNorm(8,hidden),nn.GELU()) for c in channels])
        self.mask_predictors=nn.ModuleList([nn.Sequential(nn.Conv2d(hidden,hidden,3,padding=1,bias=False),nn.GroupNorm(8,hidden),nn.GELU(),nn.Conv2d(hidden,1,1)) for _ in channels])
    def forward(self,features):
        vectors=[]
        for projection,predictor,feature in zip(self.projections,self.mask_predictors,features):
            projected=projection(feature); weight=torch.sigmoid(predictor(projected))
            vectors.append((projected*weight).sum((2,3))/weight.sum((2,3)).clamp_min(1e-6))
        return vectors

class CurrentFull(nn.Module):
    def __init__(self,n):
        super().__init__(); self.vmamba=pretrained_vmamba(); d=self.vmamba.num_features
        self.region_head=NativeMeanFusionHead(); self.lesion_projection=nn.Sequential(nn.Linear(384,d),nn.GELU(),nn.Dropout(.20))
        self.fusion_norm=nn.LayerNorm(d); self.classifier=nn.Linear(d,n)
    def forward(self,x):
        x=self.vmamba.patch_embeddings(x); features=[]
        for i in range(3): x=self.vmamba.layers[i](x); features.append(x)
        lesion=self.lesion_projection(torch.cat(self.region_head(features),dim=1))
        x=self.vmamba.layers[3](x); x=self.vmamba.norm(x); global_vector=self.vmamba.avgpool(x).flatten(1)
        return self.classifier(self.fusion_norm(global_vector+lesion))

def load_state(model,path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if isinstance(state,dict):
        for k in ('model_state_dict','model_state','state_dict'):
            if k in state and isinstance(state[k],dict): state=state[k]; break
    state={k.removeprefix('module.'):v for k,v in state.items()}
    missing,unexpected=model.load_state_dict(state,strict=False)
    assert not missing,missing; assert not unexpected,unexpected
    return model.to(DEVICE).eval()

baseline=load_state(VMambaBaseline(NUM_CLASSES),CONFIG['baseline_checkpoint'])
current=load_state(CurrentFull(NUM_CLASSES),CONFIG['current_checkpoint'])
print('Both checkpoints loaded.')


class LayerCAMCapture:
    def __init__(self,layer):
        self.activation=None; self.gradient=None
        self.handle=layer.register_forward_hook(self._forward_hook)
    def _forward_hook(self,module,inputs,output):
        self.activation=output
        output.register_hook(self._gradient_hook)
    def _gradient_hook(self,gradient): self.gradient=gradient
    def cam(self,target_size):
        assert self.activation is not None and self.gradient is not None
        cam=F.relu((F.relu(self.gradient)*self.activation).sum(dim=1,keepdim=True))
        cam=F.interpolate(cam,size=target_size,mode='bilinear',align_corners=False)
        flat=cam.flatten(1); lo=flat.min(1).values[:,None,None,None]; hi=flat.max(1).values[:,None,None,None]
        return ((cam-lo)/(hi-lo+1e-8)).detach()
    def close(self): self.handle.remove()

def calculate_metrics(cams,masks):
    eps=1e-8; binary=masks>0.5; outside=~binary
    total=cams.sum((1,2,3)).clamp_min(eps)
    inside=(cams*binary).sum((1,2,3))
    energy=inside/total
    lesion_area=binary.float().mean((1,2,3))
    lesion_mean=inside/binary.sum((1,2,3)).clamp_min(1)
    global_mean=cams.mean((1,2,3)).clamp_min(eps)
    concentration=lesion_mean/global_mean
    flat_index=cams.flatten(1).argmax(1)
    mask_flat=binary.flatten(1)
    pointing=mask_flat.gather(1,flat_index[:,None]).squeeze(1).float()
    return energy,concentration,pointing,lesion_area

@torch.enable_grad()
def model_cams(model,images):
    capture=LayerCAMCapture(model.vmamba.layers[2])
    model.zero_grad(set_to_none=True)
    logits=model(images); predicted=logits.argmax(1)
    score=logits.gather(1,predicted[:,None]).sum(); score.backward()
    cams=capture.cam(images.shape[-2:]); capture.close()
    return logits.detach(),predicted.detach(),cams.cpu()

images,masks,labels,ids,shown=next(iter(loader)); images=images[:2].to(DEVICE)
_,_,b_cam=model_cams(baseline,images); _,_,c_cam=model_cams(current,images)
assert b_cam.shape==(2,1,384,384) and c_cam.shape==(2,1,384,384)
assert torch.isfinite(b_cam).all() and torch.isfinite(c_cam).all()
print('Paired LayerCAM smoke test: PASS',b_cam.shape)
del images,b_cam,c_cam; torch.cuda.empty_cache()


rows=[]; all_images=[]; all_masks=[]; all_baseline_cams=[]; all_current_cams=[]
for batch_index,(images,masks,labels,ids,shown) in enumerate(loader):
    images_gpu=images.to(DEVICE,non_blocking=True)
    b_logits,b_pred,b_cam=model_cams(baseline,images_gpu)
    c_logits,c_pred,c_cam=model_cams(current,images_gpu)
    be,bc,bp,area=calculate_metrics(b_cam,masks)
    ce,cc,cp,_=calculate_metrics(c_cam,masks)
    for j,rid in enumerate(ids):
        true=int(labels[j]); b=int(b_pred[j]); c=int(c_pred[j])
        rows.append({
            'record_id':rid,'true_class':CLASS_NAMES[true],
            'baseline_prediction':CLASS_NAMES[b],'current_prediction':CLASS_NAMES[c],
            'baseline_correct':b==true,'current_correct':c==true,
            'lesion_area_ratio':float(area[j]),
            'baseline_lesion_energy':float(be[j]),'current_lesion_energy':float(ce[j]),
            'delta_lesion_energy':float(ce[j]-be[j]),
            'baseline_concentration':float(bc[j]),'current_concentration':float(cc[j]),
            'delta_concentration':float(cc[j]-bc[j]),
            'baseline_pointing_hit':float(bp[j]),'current_pointing_hit':float(cp[j]),
        })
    all_images.extend(shown.numpy()); all_masks.extend(masks[:,0].numpy())
    all_baseline_cams.extend(b_cam[:,0].numpy()); all_current_cams.extend(c_cam[:,0].numpy())
    if (batch_index+1)%20==0: print('Processed',min((batch_index+1)*CONFIG['batch_size'],350),'/350')

results=pd.DataFrame(rows)
all_images=np.asarray(all_images); all_masks=np.asarray(all_masks)
all_baseline_cams=np.asarray(all_baseline_cams); all_current_cams=np.asarray(all_current_cams)
assert len(results)==350
results.to_csv(OUT/'paired_layercam_per_image.csv',index=False)
print('Saved complete paired table.')


results['transition']=np.select([
    ~results.baseline_correct & results.current_correct,
    results.baseline_correct & ~results.current_correct,
    results.baseline_correct & results.current_correct,
],['current_fixes_baseline','current_breaks_baseline','both_correct'],default='both_wrong')
results['extent_group']=pd.cut(results.lesion_area_ratio,bins=[-1e-9,.05,.15,.30,1.01],
                               labels=['tiny_0_5%','small_5_15%','medium_15_30%','large_30_100%'])

def paired_summary(table,label):
    delta=table.delta_concentration.to_numpy()
    return {'subset':label,'images':len(table),
            'baseline_energy_pct':100*table.baseline_lesion_energy.mean(),
            'current_energy_pct':100*table.current_lesion_energy.mean(),
            'delta_energy_pp':100*table.delta_lesion_energy.mean(),
            'baseline_concentration':table.baseline_concentration.mean(),
            'current_concentration':table.current_concentration.mean(),
            'delta_concentration':table.delta_concentration.mean(),
            'baseline_pointing_pct':100*table.baseline_pointing_hit.mean(),
            'current_pointing_pct':100*table.current_pointing_hit.mean()}

summary=pd.DataFrame([
    paired_summary(results,'all_images'),
    paired_summary(results[results.transition=='both_correct'],'both_correct'),
    paired_summary(results[results.transition=='current_fixes_baseline'],'current_fixes_baseline'),
    paired_summary(results[results.transition=='current_breaks_baseline'],'current_breaks_baseline'),
])
extent_summary=pd.DataFrame([paired_summary(g,str(k)) for k,g in results.groupby('extent_group',observed=False)])
display(summary.round(4)); display(extent_summary.round(4))
summary.to_csv(OUT/'paired_layercam_summary.csv',index=False)
extent_summary.to_csv(OUT/'paired_layercam_by_extent.csv',index=False)


sns.set_theme(style='whitegrid',context='talk')
fig,axes=plt.subplots(1,3,figsize=(19,5.5))

plot=pd.DataFrame({'Baseline':100*results.baseline_lesion_energy,'Current full':100*results.current_lesion_energy}).melt(var_name='Model',value_name='Saliency energy inside lesion (%)')
sns.violinplot(data=plot,x='Model',y='Saliency energy inside lesion (%)',inner='quartile',cut=0,ax=axes[0],palette=['#A5A5A5','#4472C4'])
axes[0].set_title('Lesion saliency energy')

plot2=pd.DataFrame({'Baseline':results.baseline_concentration,'Current full':results.current_concentration}).melt(var_name='Model',value_name='Lesion concentration ratio')
sns.boxplot(data=plot2,x='Model',y='Lesion concentration ratio',showfliers=False,ax=axes[1],palette=['#A5A5A5','#4472C4'])
axes[1].axhline(1,color='black',lw=1); axes[1].set_title('Area-normalized concentration')

sns.barplot(data=extent_summary,x='subset',y='delta_concentration',color='#70AD47',ax=axes[2])
axes[2].axhline(0,color='black',lw=1); axes[2].set_xlabel('Ground-truth lesion extent'); axes[2].set_ylabel('Current − baseline concentration')
axes[2].set_title('Change by lesion extent'); axes[2].tick_params(axis='x',rotation=25)

plt.tight_layout(); path=OUT/'paired_layercam_aggregate.png'; plt.savefig(path,dpi=300,bbox_inches='tight'); plt.show(); print('Saved:',path)


selected=[]; used=set()
def select(category,mask,score,n=2,ascending=False):
    candidates=results.loc[mask & ~results.index.isin(used)].sort_values(score,ascending=ascending).head(n)
    for idx in candidates.index: selected.append((idx,category)); used.add(idx)

select('Current fixes baseline',results.transition=='current_fixes_baseline','delta_concentration',2)
select('Both correct; focus moves toward lesion',results.transition=='both_correct','delta_concentration',2)
select('Counterexample; focus moves away',results.transition=='both_correct','delta_concentration',2,ascending=True)
select('Small visible lesion',results.extent_group=='small_5_15%','delta_concentration',2)

selected_df=pd.DataFrame([{'index':i,'selection_category':c} for i,c in selected]).merge(results,left_on='index',right_index=True)
selected_df.to_csv(OUT/'selected_layercam_examples.csv',index=False)
display(selected_df[['selection_category','record_id','true_class','baseline_prediction','current_prediction','lesion_area_ratio','baseline_concentration','current_concentration','delta_concentration']].round(3))


def heat_overlay(image,cam,alpha=.48):
    cmap=plt.get_cmap('jet')(np.clip(cam,0,1))[...,:3]*255
    return np.clip((1-alpha)*image+alpha*cmap,0,255).astype(np.uint8)

n=len(selected_df); fig,axes=plt.subplots(n,4,figsize=(15,3.7*n),squeeze=False)
for r,row in selected_df.reset_index(drop=True).iterrows():
    i=int(row['index']); image=all_images[i]; mask=all_masks[i]
    axes[r,0].imshow(image); axes[r,0].set_title(f"Original | true: {row.true_class}")
    axes[r,1].imshow(image); axes[r,1].imshow(mask,cmap='Greens',alpha=.50,vmin=0,vmax=1); axes[r,1].set_title(f"GT lesion | area {100*row.lesion_area_ratio:.1f}%")
    axes[r,2].imshow(heat_overlay(image,all_baseline_cams[i])); axes[r,2].set_title(f"Baseline: {row.baseline_prediction}\nconcentration {row.baseline_concentration:.2f}")
    axes[r,3].imshow(heat_overlay(image,all_current_cams[i])); axes[r,3].set_title(f"Current: {row.current_prediction}\nconcentration {row.current_concentration:.2f}")
    for a in axes[r]: a.axis('off')
    axes[r,0].set_ylabel(row.selection_category,fontsize=10)
plt.tight_layout(); path=OUT/'paired_layercam_examples.png'; plt.savefig(path,dpi=300,bbox_inches='tight'); plt.show(); print('Saved:',path)

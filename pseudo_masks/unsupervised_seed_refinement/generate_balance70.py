"""Annotation-free scale/flip consistency and automatic-seed candidates."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,json,sys,time,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
OFFICIAL_SHA='08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'

METHODS={}
for scale in [1024,1536,2048]:
    for k in [3,4,5]:METHODS[f's{scale}_k{k}']=dict(family='single',scale=scale,k=k)
for ensemble in ['flip1024','multi3','tta6','feature6']:
    for k in [4,5]:
        for mode in ['hard','soft']:METHODS[f'{ensemble}_k{k}_{mode}']=dict(family=ensemble,k=k,mode=mode,offset=0.)
for cut in [-.01,.01,.02]:METHODS[f'tta6_k4_cut{cut:+.2f}']=dict(family='tta6',k=4,mode='soft',offset=cut)
METHODS['tta6_vote4']=dict(family='vote',k=4)
METHODS['tta6_median4']=dict(family='median',k=4)
METHODS['auto_seed_logistic']=dict(family='auto_seed')
METHODS['auto_seed_top4_consensus']=dict(family='auto_seed_consensus')
METHODS['rules_majority']=dict(family='rules_majority')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guided(g,p):
    box=lambda a:cv2.boxFilter(a,-1,(9,9),borderType=cv2.BORDER_REFLECT)
    mg,mp=box(g),box(p);a=(box(g*p)-mg*mp)/(box(g*g)-mg*mg+.001)
    return np.clip(box(a)*g+box(mp-a*mg),0,1)
def otsu(v):
    lo,hi=float(v.min()),float(v.max())
    if hi-lo<1e-8:return hi
    z=np.round((v-lo)/(hi-lo)*255).astype(np.uint8);t,_=cv2.threshold(z,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return lo+t/255*(hi-lo)
def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

def main():
    ap=argparse.ArgumentParser()
    for k in ['repo','weights','clusters','images','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--deps');ap.add_argument('--methods',nargs='+',default=list(METHODS));a=ap.parse_args()
    assert all(m in METHODS for m in a.methods)
    sys.path.insert(0,a.repo)
    if a.deps:sys.path.insert(0,a.deps)
    from dinov3.hub.backbones import dinov3_vits16
    from sklearn.mixture import GaussianMixture
    from sklearn.linear_model import LogisticRegression
    assert sha(a.weights)==OFFICIAL_SHA
    with np.load(a.clusters,allow_pickle=False) as fit:
        assert set(fit.files)=={'cluster_centers','cluster_seed_scores'}
        centers=fit['cluster_centers'].astype(np.float32);fg=fit['cluster_seed_scores']>.5
    cn=normalize(centers);assert fg.any() and not fg.all()
    paths=json.loads(Path(a.images).read_text());assert paths and all(isinstance(p,str) for p in paths)
    assert len(paths)==len({Path(p).stem for p in paths})
    out=Path(a.output);assert not out.exists() or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True)
    protocol=dict(started_unix=time.time(),generator_sha256=sha(__file__),checkpoint_sha256=OFFICIAL_SHA,clusters_sha256=sha(a.clusters),image_list_sha256=sha(a.images),methods={m:METHODS[m] for m in a.methods},human_masks_used=False,stage_weights_used=False,foreground_cluster_ids=np.where(fg)[0].tolist(),settings=dict(seed=42,guided_radius=4,guided_epsilon=.001,soft_temperature=.025,fusion_grid=128,auto_seed_positive='nearest of original16 clusters is foreground',auto_seed_negative='image-wise cosine-margin foreground posterior<=.05, excluding positives',logistic_C=.1,logistic_class_weight='balanced',logistic_max_iter=500))
    (out/'frozen_protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(42)
    model=dinov3_vits16(pretrained=False);model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True),strict=True);model=model.eval().cuda()
    for p in model.parameters():p.requires_grad_(False)
    def distances(norm):return np.sum(norm*norm,axis=1)[:,None]+np.sum(centers*centers,axis=1)[None,:]-2*(norm@centers.T)
    def margins(dist,grid):
        best=dist[:,fg].min(1);bg=np.sort(dist[:,~fg],axis=1)
        return {k:(bg[:,k-1]-best).reshape(grid,grid) for k in [3,4,5]}
    @torch.inference_mode()
    def feature(rgb,size,flip):
        if flip:rgb=np.ascontiguousarray(rgb[:,::-1])
        im=np.asarray(Image.fromarray(rgb).resize((size,size),Image.Resampling.BICUBIC)).copy()
        x=torch.from_numpy(im).cuda().float().permute(2,0,1).unsqueeze(0)/255
        x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):f=model.forward_features(x)['x_norm_patchtokens'][0].float().cpu().numpy()
        f=f.reshape(size//16,size//16,384)
        if flip:f=f[:,::-1].copy()
        return f
    records=[];start=time.monotonic()
    for i,path in enumerate(paths):
        tick=time.monotonic();rgb=np.array(Image.open(path).convert('RGB'));h,w=rgb.shape[:2];guide=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2GRAY)
        views={};feature_sum=np.zeros((128,128,384),np.float32);original=None
        for size in [1024,1536,2048]:
            for flip in [False,True]:
                raw=feature(rgb,size,flip);norm=normalize(raw);dist=distances(norm.reshape(-1,384))
                views[(size,flip)]=margins(dist,size//16)
                # OpenCV5 treats very high channel counts as N-D arrays.
                resized=torch.nn.functional.interpolate(torch.from_numpy(norm).permute(2,0,1)[None],size=(128,128),mode='bilinear',align_corners=False)[0].permute(1,2,0).numpy()
                feature_sum+=resized
                if size==1024 and not flip:original=(raw.reshape(-1,384),norm.reshape(-1,384),dist)
        feature_margins=margins(distances(normalize(feature_sum/6).reshape(-1,384)),128)
        ensembles={}
        for name,keys in [('flip1024',[(1024,False),(1024,True)]),('multi3',[(s,False) for s in [1024,1536,2048]]),('tta6',list(views))]:
            ensembles[name]={k:np.mean([cv2.resize(views[key][k],(128,128),interpolation=cv2.INTER_LINEAR) for key in keys],axis=0) for k in [4,5]}
        ensembles['feature6']=feature_margins
        f,norm,dist=original;sims=norm@cn.T;cosmargin=sims[:,fg].max(1)-sims[:,~fg].max(1)
        gm=GaussianMixture(n_components=2,n_init=3,random_state=42,reg_covar=1e-6,max_iter=100).fit(cosmargin[:,None]);component=int(gm.means_.ravel().argmax());post=gm.predict_proba(cosmargin[:,None])[:,component]
        means=np.sort(gm.means_.ravel());grid=np.linspace(means[0],means[1],4097);cut=float(grid[np.argmin(np.abs(gm.predict_proba(grid[:,None])[:,component]-.5))])
        otsu_mask=(cosmargin>.75*otsu(cosmargin)).reshape(64,64)
        gmm_mask=(cosmargin>.625*cut).reshape(64,64)
        top4=views[(1024,False)][4]>0
        positive=fg[np.argmin(dist,axis=1)];negative=(post<=.05)&~positive;keep=positive|negative
        diagnostics=dict(positive_auto_seeds=int(positive.sum()),negative_auto_seeds=int(negative.sum()),auto_seed_fallback=False)
        if positive.any() and negative.any():
            classifier=LogisticRegression(C=.1,class_weight='balanced',max_iter=500,random_state=42).fit(f[keep],positive[keep].astype(int))
            auto=classifier.predict_proba(f)[:,1].reshape(64,64).astype(np.float32)
        else:auto=top4.astype(np.float32);diagnostics['auto_seed_fallback']=True
        saved={}
        for method in a.methods:
            spec=METHODS[method];family=spec['family']
            if family=='single':soft=(views[(spec['scale'],False)][spec['k']]>0).astype(np.float32)
            elif family in ensembles:
                q=ensembles[family][spec['k']]-spec['offset'];soft=(q>0).astype(np.float32) if spec['mode']=='hard' else (1/(1+np.exp(-np.clip(q/.025,-60,60)))).astype(np.float32)
            elif family=='vote':soft=np.mean([cv2.resize((v[4]>0).astype(np.float32),(128,128),interpolation=cv2.INTER_LINEAR) for v in views.values()],axis=0)
            elif family=='median':soft=(np.median([cv2.resize(v[4],(128,128),interpolation=cv2.INTER_LINEAR) for v in views.values()],axis=0)>0).astype(np.float32)
            elif family=='auto_seed':soft=auto
            elif family=='auto_seed_consensus':soft=((auto>.5).astype(int)+top4.astype(int)+otsu_mask.astype(int)>=2).astype(np.float32)
            elif family=='rules_majority':soft=(top4.astype(int)+gmm_mask.astype(int)+otsu_mask.astype(int)>=2).astype(np.float32)
            else:raise ValueError(family)
            pred=guided(guide,cv2.resize(soft,(w,h),interpolation=cv2.INTER_LINEAR))>.5
            dest=out/'masks'/method/(Path(path).stem+'.png');dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(pred.astype(np.uint8)*255).save(dest)
            saved[method]=dict(file=str(dest.relative_to(out)),sha256=sha(dest),foreground_fraction=float(pred.mean()))
        records.append(dict(image=path,image_sha256=sha(path),masks=saved,seconds=time.monotonic()-tick,diagnostics=diagnostics))
        (out/'generation.json').write_text(json.dumps(dict(completed=i+1,planned=len(paths),elapsed_seconds=time.monotonic()-start,images=records),indent=2));print(f'{i+1}/{len(paths)} {records[-1]["seconds"]:.3f}s',flush=True)
    (out/'COMPLETE.json').write_text(json.dumps(dict(completed=len(paths),finished_unix=time.time(),generator_sha256=sha(__file__),generation_sha256=sha(out/'generation.json')),indent=2))
if __name__=='__main__':main()

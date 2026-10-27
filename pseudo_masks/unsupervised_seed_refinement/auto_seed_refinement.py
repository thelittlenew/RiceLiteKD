"""Frozen DINOv3, automatic prototype seeds only; no human annotation input."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from generate_balance70 import sha,guided,otsu,normalize,OFFICIAL_SHA
CS=[.01,.1,1.]
METHODS={'top4_control':dict(family='top4')}
for ci,c in enumerate(CS):
    for t in [.35,.5,.65,.8]:METHODS[f'global_c{ci}_p{round(t*100)}']=dict(family='global',ci=ci,threshold=t)
    for k in [4,5]:METHODS[f'global_c{ci}_vote{k}']=dict(family='global_vote',ci=ci,k=k,threshold=.5)
    for t in [.45,.55,.65]:
        for k in [4,5]:METHODS[f'local_c{ci}_p{round(t*100)}_vote{k}']=dict(family='local_vote',ci=ci,k=k,threshold=t)

def main():
    ap=argparse.ArgumentParser()
    for name in ['mode','repo','weights','clusters','images','output']:ap.add_argument('--'+name,required=True)
    ap.add_argument('--deps');ap.add_argument('--fit');ap.add_argument('--methods',nargs='+',default=list(METHODS));a=ap.parse_args()
    assert a.mode in ['fit','generate'];sys.path.insert(0,a.repo)
    if a.deps:sys.path.insert(0,a.deps)
    from dinov3.hub.backbones import dinov3_vits16
    from sklearn.mixture import GaussianMixture
    from sklearn.linear_model import LogisticRegression
    assert sha(a.weights)==OFFICIAL_SHA
    with np.load(a.clusters,allow_pickle=False) as data:
        assert set(data.files)=={'cluster_centers','cluster_seed_scores'}
        centers=data['cluster_centers'].astype(np.float32);fg=data['cluster_seed_scores']>.5
    cn=normalize(centers);paths=json.loads(Path(a.images).read_text());assert paths and all(isinstance(p,str) for p in paths)
    assert len(paths)==len({Path(p).stem for p in paths})
    out=Path(a.output);assert not out.exists() or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True)
    protocol=dict(mode=a.mode,started_unix=time.time(),generator_sha256=sha(__file__),shared_module_sha256=sha(Path(__file__).with_name('generate_balance70.py')),checkpoint_sha256=OFFICIAL_SHA,clusters_sha256=sha(a.clusters),image_list_sha256=sha(a.images),human_masks_used=False,stage_weights_used=False,foreground_cluster_ids=np.where(fg)[0].tolist(),methods={m:METHODS[m] for m in a.methods} if a.mode=='generate' else {},settings=dict(seed=42,C_values=CS,class_weight='balanced',max_iter=500,global_positive_seed='nearest prototype is foreground',global_negative_seed='none of the six closest prototypes is foreground',sample_cap_per_class_per_image=128,local_negative_seed='image-wise GMM posterior<=.05 excluding nearest-foreground positives',guided_radius=4,guided_epsilon=.001))
    if a.mode=='generate':protocol['auto_seed_model_sha256']=sha(a.fit)
    (out/'frozen_protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(42);rng=np.random.default_rng(42)
    model=dinov3_vits16(pretrained=False);model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True),strict=True);model=model.eval().cuda()
    for p in model.parameters():p.requires_grad_(False)
    @torch.inference_mode()
    def extract(rgb):
        im=np.asarray(Image.fromarray(rgb).resize((1024,1024),Image.Resampling.BICUBIC)).copy();x=torch.from_numpy(im).cuda().float().permute(2,0,1).unsqueeze(0)/255
        x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):return model.forward_features(x)['x_norm_patchtokens'][0].float().cpu().numpy()
    if a.mode=='generate':
        with np.load(a.fit,allow_pickle=False) as fitted:
            assert set(fitted.files)=={'coef','intercept','C_values'};coefs=fitted['coef'];intercepts=fitted['intercept'];assert fitted['C_values'].tolist()==CS
    records=[];xs=[];ys=[];start=time.monotonic()
    for i,path in enumerate(paths):
        tick=time.monotonic();rgb=np.array(Image.open(path).convert('RGB'));h,w=rgb.shape[:2];f=extract(rgb);norm=normalize(f)
        dist=np.sum(norm*norm,axis=1)[:,None]+np.sum(centers*centers,axis=1)[None,:]-2*(norm@centers.T);ranks=np.argsort(dist,axis=1)
        positive=fg[ranks[:,0]];negative=~fg[ranks[:,:6]].any(1)
        if a.mode=='fit':
            counts={}
            for label,candidates in [(1,np.where(positive)[0]),(0,np.where(negative)[0])]:
                idx=rng.choice(candidates,min(128,len(candidates)),replace=False);counts[str(label)]=len(idx)
                if len(idx):xs.append(f[idx]);ys.append(np.full(len(idx),label,np.int64))
            records.append(dict(image=path,image_sha256=sha(path),seed_counts=counts))
        else:
            guide=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2GRAY);top={k:fg[ranks[:,:k]].any(1).reshape(64,64) for k in [4,5]}
            sims=norm@cn.T;margin=sims[:,fg].max(1)-sims[:,~fg].max(1);o=(margin>.75*otsu(margin)).reshape(64,64)
            gm=GaussianMixture(n_components=2,n_init=3,random_state=42,reg_covar=1e-6,max_iter=100).fit(margin[:,None]);component=int(gm.means_.ravel().argmax());pp=gm.predict_proba(margin[:,None])[:,component]
            neg=(pp<=.05)&~positive;keep=positive|neg;local=[]
            for c in CS:
                if positive.any() and neg.any():
                    clf=LogisticRegression(C=c,class_weight='balanced',max_iter=500,random_state=42).fit(f[keep],positive[keep].astype(int));p=clf.predict_proba(f)[:,1].reshape(64,64).astype(np.float32)
                else:p=top[4].astype(np.float32)
                local.append(p)
            global_probs=[(1/(1+np.exp(-np.clip(f@coef+intercept,-60,60)))).reshape(64,64).astype(np.float32) for coef,intercept in zip(coefs,intercepts)]
            saved={}
            for method in a.methods:
                spec=METHODS[method];family=spec['family']
                if family=='top4':soft=top[4].astype(np.float32)
                elif family=='global':
                    prob=guided(guide,cv2.resize(global_probs[spec['ci']],(w,h),interpolation=cv2.INTER_LINEAR));pred=prob>spec['threshold'];soft=None
                else:
                    p=(global_probs if family=='global_vote' else local)[spec['ci']]
                    soft=((p>spec['threshold']).astype(int)+top[spec['k']].astype(int)+o.astype(int)>=2).astype(np.float32)
                if soft is not None:pred=guided(guide,cv2.resize(soft,(w,h),interpolation=cv2.INTER_LINEAR))>.5
                dest=out/'masks'/method/(Path(path).stem+'.png');dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(pred.astype(np.uint8)*255).save(dest)
                saved[method]=dict(file=str(dest.relative_to(out)),sha256=sha(dest),foreground_fraction=float(pred.mean()))
            records.append(dict(image=path,image_sha256=sha(path),masks=saved,seconds=time.monotonic()-tick))
        (out/'generation.json').write_text(json.dumps(dict(completed=i+1,planned=len(paths),elapsed_seconds=time.monotonic()-start,images=records),indent=2));print(f'{a.mode} {i+1}/{len(paths)} {time.monotonic()-tick:.3f}s',flush=True)
    if a.mode=='fit':
        x=np.concatenate(xs);y=np.concatenate(ys);cs=[];bs=[]
        for c in CS:
            print(f'FITTING C={c} on {len(y)} automatic seeds',flush=True)
            clf=LogisticRegression(C=c,class_weight='balanced',max_iter=500,random_state=42).fit(x,y);cs.append(clf.coef_[0]);bs.append(clf.intercept_[0])
        np.savez(out/'auto_seed_model.npz',coef=np.array(cs),intercept=np.array(bs),C_values=np.array(CS))
    (out/'COMPLETE.json').write_text(json.dumps(dict(completed=len(paths),finished_unix=time.time(),generator_sha256=sha(__file__),generation_sha256=sha(out/'generation.json')),indent=2))
if __name__=='__main__':main()

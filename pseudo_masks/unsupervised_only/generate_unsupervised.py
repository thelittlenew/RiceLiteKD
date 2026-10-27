"""RGB-only generators. No reference masks, segmentation checkpoints, or GT input.

The Selfment calls use official pseudo-label initialization/refinement only;
this is NOT a reproduction of its 7B + trained PatchHead pipeline.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import argparse,hashlib,json,sys,time,importlib.util,subprocess
import numpy as np
from PIL import Image
import cv2,torch
import torch.nn.functional as F

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def guided(guide,prob,r=4,eps=.001):
    box=lambda a:cv2.boxFilter(a,-1,(2*r+1,2*r+1),borderType=cv2.BORDER_REFLECT)
    mi,mp=box(guide),box(prob)
    a=(box(guide*prob)-mi*mp)/(box(guide*guide)-mi*mi+eps)
    return np.clip(box(a)*guide+box(mp-a*mi),0,1)

def skmeans(feat,k=8):
    x=F.normalize(feat,dim=1);best=None
    for restart in range(3):
        gen=torch.Generator(device=x.device).manual_seed(42+restart)
        ids=[int(torch.randint(len(x),(1,),generator=gen,device=x.device))]
        d=((x-x[ids[0]])**2).sum(1)
        for _ in range(1,k):
            idx=int(torch.multinomial(d.clamp_min(1e-10),1,generator=gen));ids.append(idx)
            d=torch.minimum(d,((x-x[idx])**2).sum(1))
        centers=x[ids].clone();old=None
        for _ in range(40):
            labels=(x@centers.T).argmax(1)
            if old is not None and torch.equal(labels,old):break
            old=labels.clone()
            for j in range(k):
                v=x[labels==j]
                if len(v):centers[j]=v.mean(0)
            centers=F.normalize(centers,dim=1)
        obj=float(((x-centers[labels])**2).sum(1).mean())
        if best is None or obj<best[0]:best=(obj,labels.clone())
    return best[1].cpu().numpy()

def color_posterior(rgb,space,GaussianMixture):
    small=cv2.resize(rgb.astype(np.float32)/255,(128,128),interpolation=cv2.INTER_AREA)
    if space=='lab':small=cv2.cvtColor(small,cv2.COLOR_RGB2LAB)
    x=small.reshape(-1,3).astype(np.float64)
    g=GaussianMixture(n_components=3,covariance_type='full',n_init=3,max_iter=100,random_state=42,reg_covar=1e-4).fit(x)
    # Brightness prior from the panicle literature; Lab variant also penalizes green.
    score=g.means_.mean(1) if space=='rgb' else g.means_[:,0]+g.means_[:,1]
    fg=int(np.argmax(score));post=g.predict_proba(x)[:,fg].reshape(128,128).astype(np.float32)
    return post,dict(space=space,means=g.means_.tolist(),weights=g.weights_.tolist(),foreground_component=fg,selection_scores=score.tolist(),converged=bool(g.converged_))

def main():
    ap=argparse.ArgumentParser()
    for key in ['repo','weights','selfment','deps','image-list','output']:ap.add_argument('--'+key,required=True)
    args=ap.parse_args();sys.path.insert(0,args.repo);sys.path.insert(0,args.deps)
    from dinov3.hub.backbones import dinov3_vits16
    from sklearn.mixture import GaussianMixture
    from sklearn.linear_model import LogisticRegression
    spec=importlib.util.spec_from_file_location('selfment_official_train',Path(args.selfment)/'train.py');sm=importlib.util.module_from_spec(spec);spec.loader.exec_module(sm)
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(42)
    paths=json.loads(Path(args.image_list).read_text());assert all(isinstance(p,str) for p in paths)
    assert len({Path(p).stem for p in paths})==len(paths)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    checkpoint_sha=sha(args.weights);assert checkpoint_sha=='08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'
    config=dict(official_checkpoint_sha256=checkpoint_sha,generator_sha256=sha(__file__),image_list_sha256=sha(args.image_list),input_count=len(paths),started_unix=time.time(),human_annotation_used=False,stage_weights_used=False,learned_fewshot_classifier_used=False,selfment_commit=subprocess.check_output(['git','-C',args.selfment,'rev-parse','HEAD'],text=True).strip(),methods={
      'rgb_gmm_0999':'RGB full-covariance GMM, K3; brightest component posterior > .999, guided. EM approximation, not original Gibbs sampler.',
      'lab_gmm_05':'Lab full-covariance GMM, K3; argmax(mean L + mean a), posterior > .5, guided. Fixed heuristic.',
      'dino_k8_lab_seeds':'DINO-S at1024, spherical K8; foreground clusters have mean Lab-GMM foreground posterior > .5, guided.',
      'dino_auto_seed_logistic':'Per-image DINO-S at1024; logistic classifier C=.1 balanced fitted solely to automatic Lab-GMM patch seeds >=.99 / <=.01; no human labels; guided.',
      'tokencut_s512':'Official Selfment NCut initializer, DINO-S at512, tau=.2, seed-connected component, guided.',
      'selfment_ipo_s512':'Official Selfment unsupervised pseudo-label generator, DINO-S at512, tau=.2,20 refinement iterations,sample_size10; no trained PatchHead; guided.'},refinement=dict(radius=4,epsilon=.001,binary_threshold=.5),evaluation='Generation has only RGB paths. Separate scoring process may run after COMPLETE.json exists.')
    (out/'frozen_protocol.json').write_text(json.dumps(config,indent=2));model=dinov3_vits16(weights=args.weights).eval().cuda()
    for p in model.parameters():p.requires_grad_(False)
    @torch.inference_mode()
    def features(rgb,size):
        im=np.asarray(Image.fromarray(rgb).resize((size,size),Image.Resampling.BICUBIC)).copy()
        x=torch.from_numpy(im).cuda().float().permute(2,0,1).unsqueeze(0)/255
        x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):return model.forward_features(x)['x_norm_patchtokens'][0].float()
    records=[];start=time.monotonic()
    for i,path in enumerate(paths):
        tick=time.monotonic();rgb=np.asarray(Image.open(path).convert('RGB'));h,w=rgb.shape[:2]
        gray=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2GRAY)
        rp,rd=color_posterior(rgb,'rgb',GaussianMixture);lp,ld=color_posterior(rgb,'lab',GaussianMixture)
        preds={}
        preds['rgb_gmm_0999']=(guided(gray,cv2.resize(rp,(w,h),interpolation=cv2.INTER_LINEAR))>.999).astype(np.uint8)
        preds['lab_gmm_05']=(guided(gray,cv2.resize(lp,(w,h),interpolation=cv2.INTER_LINEAR))>.5).astype(np.uint8)
        feat=features(rgb,1024);labels=skmeans(feat);seedprob=cv2.resize(lp,(64,64),interpolation=cv2.INTER_AREA).ravel()
        cscore=np.array([seedprob[labels==j].mean() if np.any(labels==j) else 0 for j in range(8)])
        soft=(cscore[labels]>.5).reshape(64,64).astype(np.float32)
        preds['dino_k8_lab_seeds']=(guided(gray,cv2.resize(soft,(w,h),interpolation=cv2.INTER_LINEAR))>.5).astype(np.uint8)
        keep=(seedprob<=.01)|(seedprob>=.99);y=(seedprob[keep]>=.99).astype(int)
        info=dict(positive_seeds=int(y.sum()),negative_seeds=int((y==0).sum()),fallback=False)
        if len(y)>2 and y.min()!=y.max():
            f=feat.cpu().numpy();clf=LogisticRegression(C=.1,class_weight='balanced',max_iter=1000,random_state=42).fit(f[keep],y)
            soft=clf.predict_proba(f)[:,1].reshape(64,64).astype(np.float32)
            preds['dino_auto_seed_logistic']=(guided(gray,cv2.resize(soft,(w,h),interpolation=cv2.INTER_LINEAR))>.5).astype(np.uint8)
        else:
            info['fallback']=True;preds['dino_auto_seed_logistic']=preds['lab_gmm_05'].copy()
        smallfeat=features(rgb,512)
        with torch.inference_mode():
            ncut=sm._tokencut_init_labels(smallfeat,32,32,tau=.2)
            ipo=sm._unsupervised_saliency_labels(smallfeat,32,32,n_iter=20,sample_size=10,tau=.2)
        for method,labels in [('tokencut_s512',ncut),('selfment_ipo_s512',ipo)]:
            soft=labels.cpu().numpy().reshape(32,32).astype(np.float32)
            preds[method]=(guided(gray,cv2.resize(soft,(w,h),interpolation=cv2.INTER_LINEAR))>.5).astype(np.uint8)
        saved={}
        for method,pred in preds.items():
            folder=out/'masks'/method;folder.mkdir(parents=True,exist_ok=True);dest=folder/(Path(path).stem+'.png')
            Image.fromarray(pred*255).save(dest)
            saved[method]=dict(file=str(dest.relative_to(out)),sha256=sha(dest),foreground_fraction=float(pred.mean()))
        records.append(dict(image=path,image_sha256=sha(path),seconds=time.monotonic()-tick,automatic_rgb_gmm=rd,automatic_lab_gmm=ld,cluster_seed_scores=cscore.tolist(),auto_seed_classifier=info,masks=saved))
        progress=dict(completed=i+1,planned=len(paths),elapsed_seconds=time.monotonic()-start,images=records)
        (out/'generation.json').write_text(json.dumps(progress,indent=2));print(f'{i+1}/{len(paths)} {Path(path).stem} {records[-1]["seconds"]:.2f}s',flush=True)
    (out/'COMPLETE.json').write_text(json.dumps(dict(completed=len(paths),finished_unix=time.time(),generator_sha256=sha(__file__),generation_sha256=sha(out/'generation.json')),indent=2))

if __name__=='__main__':main()

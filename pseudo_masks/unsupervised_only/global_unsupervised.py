"""Cross-image clustering/self-training from RGB-derived seeds only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import argparse,json,time,sys,hashlib
import numpy as np
from PIL import Image
import cv2,torch
from generate_unsupervised import color_posterior,guided,sha

def main():
    ap=argparse.ArgumentParser()
    for key in ['repo','weights','deps','unlabeled-train','image-list','output']:ap.add_argument('--'+key,required=True)
    args=ap.parse_args();sys.path.insert(0,args.deps);sys.path.insert(0,args.repo)
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.linear_model import LogisticRegression
    from dinov3.hub.backbones import dinov3_vits16
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(42);rng=np.random.default_rng(42)
    train=json.loads(Path(args.unlabeled_train).read_text());targets=json.loads(Path(args.image_list).read_text())
    assert all(isinstance(p,str) for p in train+targets) and not set(train)&set(targets)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    protocol=dict(generator_sha256=sha(__file__),shared_generator_sha256=sha(Path(__file__).with_name('generate_unsupervised.py')),started_unix=time.time(),unlabeled_training_images=len(train),target_images=len(targets),ground_truth_masks_used=False,reference_masks_used=False,stage_weights_used=False,official_checkpoint_sha256=sha(args.weights),train_list_sha256=sha(args.unlabeled_train),target_list_sha256=sha(args.image_list),settings=dict(input_size=1024,seed=42,cluster_samples_per_image=256,seed_samples_per_class_per_image=128,K=16,cluster_foreground='mean automatic Lab-GMM posterior >.5',C=.1,class_weight='balanced',patch_seed_thresholds=[.01,.99],max_iter=1000,guided_radius=4,guided_epsilon=.001,prediction_threshold=.5))
    assert protocol['official_checkpoint_sha256']=='08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'
    (out/'frozen_protocol.json').write_text(json.dumps(protocol,indent=2))
    model=dinov3_vits16(weights=args.weights).eval().cuda()
    for p in model.parameters():p.requires_grad_(False)
    @torch.inference_mode()
    def feature(rgb):
        im=np.asarray(Image.fromarray(rgb).resize((1024,1024),Image.Resampling.BICUBIC)).copy()
        x=torch.from_numpy(im).cuda().float().permute(2,0,1).unsqueeze(0)/255
        x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):return model.forward_features(x)['x_norm_patchtokens'][0].float().cpu().numpy()
    xs=[];ys=[];cx=[];cp=[];training=[]
    for i,path in enumerate(train):
        rgb=np.asarray(Image.open(path).convert('RGB'));f=feature(rgb)
        post,desc=color_posterior(rgb,'lab',GaussianMixture);pp=cv2.resize(post,(64,64),interpolation=cv2.INTER_AREA).ravel()
        idx=rng.choice(len(f),256,replace=False);cx.append(f[idx]);cp.append(pp[idx])
        counts={}
        for label,condition in [(0,pp<=.01),(1,pp>=.99)]:
            candidates=np.where(condition)[0];idx=rng.choice(candidates,min(128,len(candidates)),replace=False)
            if len(idx):xs.append(f[idx]);ys.append(np.full(len(idx),label,dtype=np.int64))
            counts[str(label)]=len(idx)
        training.append(dict(image=path,image_sha256=sha(path),automatic_seeds=counts))
        (out/'progress.json').write_text(json.dumps(dict(phase='unlabeled_fitting',completed=i+1,total=len(train))))
        print(f'FIT {i+1}/{len(train)}',flush=True)
    x=np.concatenate(xs);y=np.concatenate(ys);c=np.concatenate(cx);p=np.concatenate(cp)
    clf=LogisticRegression(C=.1,class_weight='balanced',max_iter=1000,random_state=42).fit(x,y)
    cn=c/np.maximum(np.linalg.norm(c,axis=1,keepdims=True),1e-8)
    km=MiniBatchKMeans(n_clusters=16,random_state=42,n_init=3,batch_size=2048,max_iter=100).fit(cn)
    labels=km.predict(cn);scores=np.bincount(labels,weights=p,minlength=16)/np.maximum(np.bincount(labels,minlength=16),1)
    np.savez(out/'unlabeled_fit.npz',coef=clf.coef_,intercept=clf.intercept_,cluster_centers=km.cluster_centers_,cluster_seed_scores=scores)
    (out/'training_rgb_provenance.json').write_text(json.dumps(dict(images=training,seed_total=len(y),positive=int(y.sum()),negative=int((y==0).sum()),clustering_samples=len(c),cluster_scores=scores.tolist()),indent=2))
    del xs,ys,cx,cp,x,y,c,cn
    records=[];start=time.monotonic()
    for i,path in enumerate(targets):
        tick=time.monotonic();rgb=np.asarray(Image.open(path).convert('RGB'));h,w=rgb.shape[:2];f=feature(rgb)
        pn=clf.predict_proba(f)[:,1].reshape(64,64).astype(np.float32)
        norm=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)
        pc=(scores[km.predict(norm)]>.5).reshape(64,64).astype(np.float32)
        gray=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2GRAY);saved={}
        for method,soft in [('global_auto_seed_logistic',pn),('global_k16_auto_semantics',pc)]:
            pred=(guided(gray,cv2.resize(soft,(w,h),interpolation=cv2.INTER_LINEAR))>.5).astype(np.uint8)
            folder=out/'masks'/method;folder.mkdir(parents=True,exist_ok=True);dest=folder/(Path(path).stem+'.png');Image.fromarray(pred*255).save(dest)
            saved[method]=dict(file=str(dest.relative_to(out)),sha256=sha(dest),foreground_fraction=float(pred.mean()))
        records.append(dict(image=path,image_sha256=sha(path),masks=saved,seconds=time.monotonic()-tick))
        (out/'generation.json').write_text(json.dumps(dict(completed=i+1,planned=len(targets),images=records),indent=2))
        (out/'progress.json').write_text(json.dumps(dict(phase='generation',completed=i+1,total=len(targets))))
        print(f'GEN {i+1}/{len(targets)}',flush=True)
    (out/'COMPLETE.json').write_text(json.dumps(dict(completed=len(targets),finished_unix=time.time(),generation_sha256=sha(out/'generation.json'),generator_sha256=sha(__file__)),indent=2))

if __name__=='__main__':main()

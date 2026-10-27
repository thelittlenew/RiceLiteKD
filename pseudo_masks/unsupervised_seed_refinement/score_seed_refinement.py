"""Evaluate completed immutable predictions. Not imported by the generator."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--split',choices=['val','test'],required=True);args=ap.parse_args()
root=Path('/root/autodl-tmp/work/pseudo_mask_probe_20261011/unsupervised_seed_refinement');run=root/args.run
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
done=json.loads((run/'COMPLETE.json').read_text());assert done['generation_sha256']==sha(run/'generation.json')
gen=json.loads((run/'generation.json').read_text());assert gen['completed']==gen['planned']
protocol=json.loads((run/'frozen_protocol.json').read_text());methods=list(protocol['methods'])
rows=json.loads((Path('/root/autodl-tmp/work/stage3_comparison_20261006/manifests')/f'RiceSEG_{args.split}.json').read_text())
lookup={r['image']:r for r in gen['images']}
totals={m:np.zeros(4,dtype=np.int64) for m in methods};records=[]
def metrics(c):
    tp,fp,fn,tn=map(int,c)
    return dict(counts=[tp,fp,fn,tn],panicle_iou=tp/(tp+fp+fn) if tp+fp+fn else None,panicle_dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
started=time.time();assert started>done['finished_unix']
for r in rows:
    entry=lookup[r['image']];gt=np.array(Image.open(r['mask']));valid=gt!=255;g=gt[valid]==1;per={}
    for m in methods:
        info=entry['masks'][m];p=run/info['file'];assert sha(p)==info['sha256']
        arr=np.array(Image.open(p));assert arr.shape==gt.shape and set(np.unique(arr))<={0,255}
        pred=arr[valid]>0;c=np.array([(pred&g).sum(),(pred&~g).sum(),(~pred&g).sum(),(~pred&~g).sum()])
        totals[m]+=c;per[m]=metrics(c)
    records.append(dict(image=r['image'],mask=r['mask'],metrics=per))
summary={m:metrics(c) for m,c in totals.items()}
res=dict(split=args.split,image_count=len(rows),scoring_started_unix=started,generation_finished_unix=done['finished_unix'],summary=summary,images=records)
(run/'evaluation.json').write_text(json.dumps(res,indent=2))
for m,v in sorted(summary.items(),key=lambda kv:kv[1]['panicle_iou'],reverse=True):print(m,'IoU',round(v['panicle_iou']*100,2),'P',round(v['precision']*100,2),'R',round(v['recall']*100,2))

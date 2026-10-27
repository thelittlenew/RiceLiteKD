"""Fixed student and downstream recipes for the two requested supplements."""
import hashlib,json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
OLD=WORK/'stage3_comparison_20261006'
LITE=WORK/'stage3_lite_20260926'
DATA=Path('/root/stage3_full_source_20260926')
for p in [WORK/'baseline_src/mmsegmentation',LITE,OLD]:sys.path.insert(0,str(p))
import torch
from comparison_common import Config,MODELS
from mmengine.runner import Runner
torch.set_num_threads(4)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write_json(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(data,indent=2,ensure_ascii=False));t.replace(p)

def save_torch(p,data):
 p=Path(p);t=p.with_name(p.name+'.tmp');torch.save(data,t);t.replace(p)

def verify_plan():
 plan=json.loads((ROOT/'plan.json').read_text())
 for p,digest in plan['hashes'].items():assert sha(p)==digest,p
 return plan

def get_config(job):
 cfg=Config.fromfile(str(ROOT/'base_configs'/f"CVRP_full_stage3_seed{job['seed']}.py"))
 cfg.work_dir=str(ROOT/'runs'/job['name'])
 cfg.load_from=str(ROOT/'source_full_initialization.pth')
 cfg.train_dataloader.dataset.ann_file=str(ROOT/'subsets'/f"{job['name']}.txt")
 assert cfg.randomness.seed==job['seed']
 assert cfg.train_cfg.max_epochs==120 and cfg.train_dataloader.batch_size==8
 return cfg

def measures(matrix):
 x=matrix.double();tp=x.diag();iou=tp/(x.sum(0)+x.sum(1)-tp);dice=2*tp/(x.sum(0)+x.sum(1))
 return dict(miou=iou.mean().item(),panicle_iou=iou[1].item(),mean_dice=dice.mean().item(),class_iou=iou.tolist(),class_dice=dice.tolist(),confusion_matrix=matrix.tolist())

@torch.inference_mode()
def score(model,cfg,split,job):
 loader=Runner.build_dataloader(cfg[split+'_dataloader'],seed=job['seed'])
 expected=json.loads((OLD/'manifests'/f"{job['dataset']}_{split}.json").read_text())
 actual={(str(Path(loader.dataset.get_data_info(i)['img_path']).resolve()),str(Path(loader.dataset.get_data_info(i)['seg_map_path']).resolve())) for i in range(len(loader.dataset))}
 assert actual=={(r['image'],r['mask']) for r in expected}
 matrix=torch.zeros(2,2,dtype=torch.int64);rows=[]
 for batch in loader:
  for sample in model.test_step(batch):
   gt=sample.gt_sem_seg.data.squeeze(0).long();pred=sample.pred_sem_seg.data.squeeze(0).long()
   assert gt.shape==pred.shape
   valid=gt!=255
   counts=torch.bincount(2*gt[valid]+pred[valid],minlength=4).reshape(2,2).cpu()
   matrix+=counts;rows.append(dict(image=sample.img_path,**measures(counts)))
 assert len(rows)==len(expected)
 return dict(images=len(rows),per_image=rows,**measures(matrix))

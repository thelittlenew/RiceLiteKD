from pathlib import Path
from copy import deepcopy
import hashlib,json,sys,os,time
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
SITE=WORK/'segmentation_comparison_audit_20261011/site'
sys.path[:0]=[str(SITE),str(WORK/'baseline_src/mmsegmentation')]
import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
register_all_modules()
import comparison_models
TEACHER_MODELS=['SegFormer_B2','DDRNet_23','PIDNet_M','HRNetV2_W32','UNet','CCASeg_S','GCNet_M']
STUDENT_MODELS=['DABNet','LEDNet','EDANet','ESPNetv2','Fast_SCNN','MK_UNet_M','FEM_Lite']
MODELS_TO_RUN=STUDENT_MODELS+TEACHER_MODELS
DATASETS=['RiceSEG','CVRP_full'];SEEDS=[42,7]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(p,obj):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)
def atomic_save(obj,p):
 p=Path(p);t=p.with_name(p.name+'.tmp');torch.save(obj,t);t.replace(p)
def original_config(dataset,seed):
 directory='cea_public_20260915' if dataset=='RiceSEG' else 'cea_cvrp_full_20260923'
 return WORK/directory/'configs'/f'{dataset}_b0_seed{seed}.py'
def make_config(model,dataset,seed):
 old=Config.fromfile(str(original_config(dataset,seed)));cfg=deepcopy(old)
 specs=json.loads((ROOT/'model_specs.json').read_text());entry=specs[model]
 cfg.custom_imports=dict(imports=['common'],allow_failed_imports=False)
 cfg.work_dir=str(ROOT/'runs'/f'{dataset}_{model}_seed{seed}')
 cfg.load_from=None;cfg.resume=False
 cfg.model=deepcopy(entry['model']);cfg.model.data_preprocessor=deepcopy(old.model.data_preprocessor)
 cfg.model.train_cfg={};cfg.model.test_cfg=dict(mode='whole')
 cfg.default_hooks.checkpoint.interval=1;cfg.default_hooks.checkpoint.max_keep_ckpts=1
 cfg.default_hooks.checkpoint.save_optimizer=True
 cfg.default_hooks.logger.interval=20
 cfg.log_processor=dict(by_epoch=True)
 cfg.custom_hooks=[dict(type='ProgressHook')]
 # ImageNet backbones use the original fine-tuning rate. Models trained from
 # scratch use 1e-3 uniformly and no arbitrary 'head' name multiplier.
 if entry['initialization']=='random':
  cfg.optim_wrapper.optimizer.lr=1e-3
  cfg.optim_wrapper.paramwise_cfg=dict(custom_keys={'norm':dict(decay_mult=0.)})
 if model=='PIDNet_M':
  pipeline=cfg.train_dataloader.dataset.pipeline
  pipeline.insert(len(pipeline)-1,dict(type='GenerateEdge',edge_width=4))
 assert cfg.train_cfg.max_epochs==120 and cfg.train_dataloader.batch_size==8
 return cfg
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
@HOOKS.register_module()
class ProgressHook(Hook):
 def after_train_epoch(self,runner):
  write_json(Path(runner.work_dir)/'progress.json',dict(epoch=runner.epoch+1,max_epochs=runner.max_epochs,iteration=runner.iter+1,time=time.time()))
 def after_val_epoch(self,runner,metrics=None):
  if metrics is not None:
   p=Path(runner.work_dir)/'epoch_validation.json';rows=json.loads(p.read_text()) if p.exists() else {}
   rows[str(runner.epoch)]=dict(epoch=runner.epoch,**{k:float(v) for k,v in metrics.items()})
   write_json(p,rows)
def verify_plan():
 plan=json.loads((ROOT/'plan.json').read_text())
 for path,digest in plan['frozen_files'].items():assert sha(ROOT/path)==digest,path
 for path,digest in plan['external_files'].items():assert sha(path)==digest,path
 return plan

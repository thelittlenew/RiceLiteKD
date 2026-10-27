"""Frozen downstream interfaces for the 2026-10-06 Stage3 replacement."""
from pathlib import Path
from copy import deepcopy
import dataclasses
import hashlib
import importlib.util
import json
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
LITE=WORK/'stage3_lite_20260926'
MM=WORK/'baseline_src/mmsegmentation'
sys.path.insert(0,str(MM))
sys.path.insert(0,str(LITE))
from mmengine.config import Config
from mmseg.models.backbones.cgnet import CGNet
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
register_all_modules()
from matched_cgnet_refine7_recipe import CGNetRefine7Head

SOURCE=LITE/'runs/SourceFull_refine7_distill_120ep_seed42/best.pt'
SOURCE_SHA='de3886554edfa8c04ce8d8d666dbb7f964e96394e9a4c10d150101538701ddec'
BACKBONE=dict(in_channels=3,num_channels=(32,64,128),num_blocks=(3,21),dilations=(2,4),
              reductions=(8,16),norm_cfg=dict(type='BN',eps=1e-3,requires_grad=True),
              act_cfg=dict(type='PReLU'),norm_eval=False,with_cp=False,pretrained=None,init_cfg=None)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp'); t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)

def atomic_save(value,path):
    path=Path(path);temp=path.with_name(path.name+'.tmp');torch.save(value,temp);temp.replace(path)

@MODELS.register_module()
class CGNetStage3Pyramid(CGNet):
    """Same parameter-free /4,/8,/16,/32 interface used in source distillation."""
    def forward(self,x):
        features=super().forward(x)
        return [features[1],features[2],F.avg_pool2d(features[2],2),F.avg_pool2d(features[2],4)]

def source_backbone():
    assert sha(SOURCE)==SOURCE_SHA
    ck=torch.load(SOURCE,map_location='cpu',weights_only=False)
    assert ck['architecture']=='CGNetRefine7NativeSeg'
    state={k[len('native.backbone.'):]:v for k,v in ck['model'].items() if k.startswith('native.backbone.')}
    assert state
    return state

def old_seg_config(dataset,arm,seed):
    p=(WORK/'cea_public_20260915/configs'/f'RiceSEG_{arm}_seed{seed}.py' if dataset=='RiceSEG'
       else WORK/'cea_cvrp_full_20260923/configs'/f'CVRP_full_{arm}_seed{seed}.py')
    return p

def seg_config(dataset,arm,seed):
    original=Config.fromfile(str(old_seg_config(dataset,arm,seed)))
    cfg=deepcopy(original)
    cfg.model.backbone=dict(type='CGNet',**deepcopy(BACKBONE))
    # Latest user direction: preserve the complete frozen Refine7 segmentation model.
    cfg.model.decode_head.type='CGNetRefine7Head'
    cfg.model.decode_head.in_channels=[35,131,256]
    cfg.model.decode_head.in_index=[0,1,2]
    cfg.model.decode_head.refine_blocks=2
    assert cfg.model.decode_head.channels==256
    cfg.load_from=str(ROOT/'source_full_initialization.pth') if arm=='stage3' else None
    cfg.work_dir=str(ROOT/'runs'/f'{dataset}_{arm}_seed{seed}')
    cfg.custom_imports=dict(imports=['comparison_common'],allow_failed_imports=False)
    cfg.resume=False
    for key in original:
        if key not in ('model','load_from','work_dir','custom_imports','resume'):
            assert cfg[key]==original[key],key
    for key in original.model:
        if key not in ('backbone','decode_head'):assert cfg.model[key]==original.model[key],key
    for key in original.model.decode_head:
        if key not in ('in_channels','in_index','type'):assert cfg.model.decode_head[key]==original.model.decode_head[key],key
    assert cfg.train_cfg.max_epochs==120 and cfg.train_dataloader.batch_size==8
    return cfg

def detector_module():
    if 'archived_detector' in sys.modules:return sys.modules['archived_detector']
    sp=importlib.util.spec_from_file_location('archived_detector',ROOT/'archived_detector.py')
    m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
    return m

def det_config(dataset,seed):
    m=detector_module()
    p=WORK/'ws60'/f'{dataset}_r3ws60/config.json'
    old=json.loads(p.read_text());allowed={f.name for f in dataclasses.fields(m.Config)}
    c={k:v for k,v in old.items() if k in allowed}
    c.update(seed=seed,output_dir=str(ROOT/'runs'/f'{dataset}_stage3_seed{seed}'),
             stage3_pth=str(SOURCE),stage3_model_py=str(ROOT/'comparison_common.py'),resume='',eval_only=False)
    cfg=m.Config(**c)
    assert cfg.epochs==60 and cfg.image_size==640 and cfg.batch_size==8
    assert cfg.neck_widths=='192,384,768' and cfg.arch=='yolov8'
    assert cfg.tta=='none' and cfg.conf_thres==.001 and cfg.nms_iou==.7
    return cfg

class DetectionExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Keep the archived student's parameter prefix and therefore its actual optimizer grouping.
        self.student=CGNetStage3Pyramid(**deepcopy(BACKBONE))
        self.student.load_state_dict(source_backbone(),strict=True)
    def forward(self,x):return self.student(x)

def build_detector(cfg):
    m=detector_module()
    extractor=DetectionExtractor()
    widths=[int(x) for x in cfg.neck_widths.split(',')]
    neck=m.YOLOv8Neck(in_channels=(256,256,256),c3=widths[0],c4=widths[1],c5=widths[2])
    head=m.YOLOv8Detect(channels=tuple(widths),num_classes=cfg.num_classes,reg_max=m.REG_MAX)
    return m.UnifiedDetector(extractor,neck,head)

def verify_plan():
    plan=json.loads((ROOT/'plan.json').read_text())
    for name,digest in plan['code_sha256'].items():assert sha(ROOT/name)==digest,name
    for path,digest in plan['dependencies_sha256'].items():assert sha(path)==digest,path
    for name,digest in plan['configs_sha256'].items():assert sha(ROOT/'configs'/name)==digest,name
    assert sha(SOURCE)==SOURCE_SHA
    return plan

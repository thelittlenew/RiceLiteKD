"""CGNet fixed-backbone, three-scale SegformerHead256 decoder comparison."""
from copy import deepcopy
from pathlib import Path
import hashlib
import torch
from matched_cgnet_recipe import ROOT, IMPLEMENTATION_SHA256 as BACKBONE_IMPLEMENTATION
from matched_cgnet_recipe import build_config as build_reference
from matched_recipe import MMSEG
from rice_lite import check_budget
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

IMPLEMENTATION_SHA256={**BACKBONE_IMPLEMENTATION,
    'mmseg/models/decode_heads/segformer_head.py':'47a503a6178ea9ee3cf4793b65aa5e4dd0308f05734db1e7bb16a95fdfc77b1b'}
INITIALIZATION='MMSeg CGNet native random Kaiming/BN/PReLU initialization; native SegformerHead initialization'


def build_config(seed=42):
    for name,expected in IMPLEMENTATION_SHA256.items():
        assert hashlib.sha256((Path(MMSEG)/name).read_bytes()).hexdigest()==expected,name
    cfg,baseline=build_reference(seed)
    reference=deepcopy(cfg)
    cfg.model.decode_head=dict(type='SegformerHead',in_channels=[35,131,256],in_index=[0,1,2],
        channels=256,dropout_ratio=0.1,num_classes=2,norm_cfg=dict(type='BN',requires_grad=True),
        align_corners=False,loss_decode=deepcopy(reference.model.decode_head.loss_decode))
    cfg.work_dir=str(ROOT/'runs'/f'RiceSEG_cgnet_fusion256_matched_random_seed{seed}')
    cfg.custom_imports=dict(imports=['matched_cgnet_fusion_recipe'],allow_failed_imports=False)
    for key in reference:
        if key not in ('model','work_dir','custom_imports'):
            assert cfg[key]==reference[key],key
    for key in reference.model:
        if key!='decode_head':assert cfg.model[key]==reference.model[key],key
    return cfg,baseline


class CGNetFusionNativeSeg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        register_all_modules()
        cfg,_=build_config(42)
        self.native=MODELS.build(cfg.model)
        assert check_budget(self)==798644

    def forward(self,x):
        size=tuple(x.shape[-2:])
        metas=[dict(img_shape=size,ori_shape=size,pad_shape=size) for _ in range(x.shape[0])]
        return {'logits':self.native.encode_decode(x,metas)}


def as_standalone(state):
    model=CGNetFusionNativeSeg()
    model.native.load_state_dict(state,strict=True)
    return model

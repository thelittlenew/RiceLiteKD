"""Fixed CGNet/fusion256 backbone and head, plus two residual spatial refiners."""
from copy import deepcopy
import torch
from torch import nn
from matched_cgnet_fusion_recipe import ROOT, IMPLEMENTATION_SHA256
from matched_cgnet_fusion_recipe import build_config as build_reference
from rice_lite import check_budget
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.segformer_head import SegformerHead
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

INITIALIZATION='MMSeg CGNet native random Kaiming/BN/PReLU; native SegformerHead and ConvModule random initialization'


class ResidualLocalRefine(nn.Module):
    def __init__(self, channels, norm_cfg):
        super().__init__()
        self.depthwise=ConvModule(channels,channels,3,padding=1,groups=channels,
                                  norm_cfg=deepcopy(norm_cfg),act_cfg=dict(type='ReLU'))
        self.pointwise=ConvModule(channels,channels,1,norm_cfg=deepcopy(norm_cfg),act_cfg=None)
        self.activation=nn.ReLU(inplace=False)

    def forward(self,x):
        return self.activation(x+self.pointwise(self.depthwise(x)))


@MODELS.register_module()
class CGNetRefineHead(SegformerHead):
    def __init__(self,refine_blocks=2,**kwargs):
        super().__init__(**kwargs)
        assert refine_blocks==2
        self.refine=nn.Sequential(*[ResidualLocalRefine(self.channels,self.norm_cfg)
                                    for _ in range(refine_blocks)])

    def cls_seg(self,feat):
        # Reuse the native fusion forward and its dropout/classifier unchanged.
        return super().cls_seg(self.refine(feat))


def build_config(seed=42):
    cfg,baseline=build_reference(seed)
    reference=deepcopy(cfg)
    cfg.model.decode_head.type='CGNetRefineHead'
    cfg.model.decode_head.refine_blocks=2
    cfg.work_dir=str(ROOT/'runs'/f'RiceSEG_cgnet_refine2_matched_random_seed{seed}')
    cfg.custom_imports=dict(imports=['matched_cgnet_refine_recipe'],allow_failed_imports=False)
    for key in reference:
        if key not in ('model','work_dir','custom_imports'):assert cfg[key]==reference[key],key
    for key in reference.model:
        if key!='decode_head':assert cfg.model[key]==reference.model[key],key
    for key in reference.model.decode_head:
        if key!='type':assert cfg.model.decode_head[key]==reference.model.decode_head[key],key
    return cfg,baseline


class CGNetRefineNativeSeg(nn.Module):
    def __init__(self):
        super().__init__()
        register_all_modules()
        cfg,_=build_config(42)
        self.native=MODELS.build(cfg.model)
        assert check_budget(self)==936372

    def forward(self,x):
        size=tuple(x.shape[-2:])
        metas=[dict(img_shape=size,ori_shape=size,pad_shape=size) for _ in range(x.shape[0])]
        return {'logits':self.native.encode_decode(x,metas)}


def as_standalone(state):
    model=CGNetRefineNativeSeg()
    model.native.load_state_dict(state,strict=True)
    return model

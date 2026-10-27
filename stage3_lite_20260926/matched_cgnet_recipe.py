"""CGNet M3/N21 architecture baseline under the archived RiceSEG recipe.

Uses the installed MMSeg implementation, not author-reported training settings.
The exported inference wrapper still requires MMSeg/MMEngine/MMCV.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import torch
from matched_recipe import ROOT, MMSEG, build_config as build_reference
from rice_lite import check_budget
from mmseg.registry import MODELS

IMPLEMENTATION_SHA256 = {
    'mmseg/models/backbones/cgnet.py': '6522971ea18fe0e911f5cd765cb60bd6ed11e8c0dcb8240e5eebbc6b6ca5cf61',
    'mmseg/models/decode_heads/fcn_head.py': 'c6fc931ac546d5d1f19bd2bf84679aa42be3245412b4d4c02e89bf2b16ba473a',
}
INITIALIZATION = 'MMSeg CGNet native random Kaiming/BN/PReLU initialization; native FCN head initialization'


def verify_implementation():
    for name, expected in IMPLEMENTATION_SHA256.items():
        assert hashlib.sha256((Path(MMSEG)/name).read_bytes()).hexdigest() == expected, name


def build_config(seed=42):
    verify_implementation()
    cfg, baseline = build_reference(seed)
    reference = deepcopy(cfg)
    head = cfg.model.decode_head
    cfg.model.backbone = dict(type='CGNet', in_channels=3, num_channels=(32,64,128),
        num_blocks=(3,21), dilations=(2,4), reductions=(8,16),
        norm_cfg=dict(type='BN',eps=1e-3,requires_grad=True),
        act_cfg=dict(type='PReLU'), norm_eval=False, with_cp=False,
        pretrained=None, init_cfg=None)
    cfg.model.decode_head = dict(type='FCNHead', in_channels=256, in_index=2,
        channels=256, num_convs=0, concat_input=False,
        dropout_ratio=head.dropout_ratio, num_classes=head.num_classes,
        norm_cfg=dict(type='BN',eps=1e-3,requires_grad=True),
        align_corners=head.align_corners, loss_decode=deepcopy(head.loss_decode))
    cfg.work_dir = str(ROOT/'runs'/f'RiceSEG_cgnet_matched_random_seed{seed}')
    cfg.custom_imports = dict(imports=['matched_cgnet_recipe'],allow_failed_imports=False)
    for key in reference:
        if key not in ('model','work_dir','custom_imports'):
            assert cfg[key] == reference[key], key
    for key in reference.model:
        if key not in ('backbone','decode_head'):
            assert cfg.model[key] == reference.model[key], key
    assert cfg.load_from is None and cfg.model.pretrained is None
    assert cfg.model.decode_head.num_classes == 2
    assert cfg.model.decode_head.dropout_ratio == 0.1
    return cfg, baseline


class CGNetNativeSeg(torch.nn.Module):
    """Receives already normalized RGB tensors; preserves native encode_decode."""
    def __init__(self):
        super().__init__()
        cfg, _ = build_config(42)
        self.native = MODELS.build(cfg.model)
        assert check_budget(self) == 491956

    def forward(self, x):
        size = tuple(x.shape[-2:])
        metas = [dict(img_shape=size,ori_shape=size,pad_shape=size) for _ in range(x.shape[0])]
        return {'logits': self.native.encode_decode(x,metas)}


def as_standalone(state):
    model = CGNetNativeSeg()
    model.native.load_state_dict(state,strict=True)
    return model

"""Register the <=1M model with the archived MMSeg random-student recipe."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MMSEG = '/root/autodl-tmp/work/baseline_src/mmsegmentation'
sys.path.insert(0, MMSEG)
from mmengine.config import Config
from mmengine.model import BaseModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from rice_lite import RiceLiteBackbone, AdditivePyramid, RiceLiteSeg, check_budget


@MODELS.register_module()
class RiceLiteMatchedBackbone(BaseModule):
    def __init__(self, variant='base', context=True, init_cfg=None):
        assert init_cfg is None, 'This experiment requires random initialization'
        super().__init__(init_cfg=None)
        self.core = RiceLiteBackbone(variant, context)

    def forward(self, x):
        return self.core(x)


@MODELS.register_module()
class RiceLiteMatchedHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.decoder = AdditivePyramid(self.in_channels, self.channels)

    def forward(self, inputs):
        features = self._transform_inputs(inputs)
        return self.cls_seg(self.decoder(features)[0])


def build_config(seed=42):
    assert seed in (42, 7, 3407)
    baselines = json.loads((ROOT / 'matched_baselines.json').read_text())
    baseline = next(row for row in baselines['records'] if row['seed'] == seed)
    source = Path(f'/root/autodl-tmp/work/cea_public_20260915/configs/RiceSEG_random_student_seed{seed}.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == baseline['config_sha256']
    original = Config.fromfile(str(source))
    assert original.load_from is None and original.model.pretrained is None
    cfg = deepcopy(original)
    cfg.model.backbone = dict(type='RiceLiteMatchedBackbone', variant='base', context=True)
    cfg.model.decode_head.type = 'RiceLiteMatchedHead'
    cfg.model.decode_head.in_channels = [24, 48, 96, 160]
    cfg.model.decode_head.channels = 32
    cfg.work_dir = str(ROOT / 'runs' / f'RiceSEG_base_matched_random_seed{seed}')
    cfg.custom_imports = dict(imports=['matched_recipe'], allow_failed_imports=False)
    # Keep the actual nested dataloader pipelines, CE, head dropout, AdamW,
    # per-name lr/decay rules, AMP and epoch-based schedulers verbatim.
    for key in original.keys():
        if key not in ('model', 'work_dir'):
            assert cfg[key] == original[key], key
    assert cfg.train_cfg.max_epochs == 120 and cfg.train_dataloader.batch_size == 8
    assert cfg.model.decode_head.dropout_ratio == 0.1
    assert cfg.model.decode_head.loss_decode == original.model.decode_head.loss_decode
    return cfg, baseline


def as_standalone(state):
    translated = {}
    for key, tensor in state.items():
        if key.startswith('backbone.core.'):
            name = 'backbone.' + key[len('backbone.core.'):]
        elif key.startswith('decode_head.decoder.'):
            name = 'decoder.' + key[len('decode_head.decoder.'):]
        elif key.startswith('decode_head.conv_seg.'):
            name = 'classifier.' + key[len('decode_head.conv_seg.'):]
        else:
            raise ValueError(f'Unexpected matched model state: {key}')
        translated[name] = tensor
    model = RiceLiteSeg('base', context=True)
    model.load_state_dict(translated, strict=True)
    assert check_budget(model) == 734186
    return model

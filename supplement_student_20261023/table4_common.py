"""Table 4 low-label controls, using each model's archived full-data recipe."""
from experiment_common import *
COMPARISON = WORK / 'segmentation_comparisons_v2_20261011'
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(COMPARISON))
import common as archived_comparison
from mmengine.model import revert_sync_batchnorm
from singleton_batch import RedistributeSingletonBatchSampler

def table4_config(job):
    cfg = Config.fromfile(job['original_config'])
    cfg.work_dir = str(ROOT / 'runs' / job['name'])
    cfg.resume = False
    cfg.train_dataloader.dataset.ann_file = str(ROOT / 'subsets' / f"{job['dataset']}_seed{job['seed']}.txt")
    if job.get('avoid_singleton_batch'):
        cfg.train_dataloader.batch_sampler = dict(type='RedistributeSingletonBatchSampler')
    cfg.default_hooks.checkpoint.interval = 1
    cfg.default_hooks.checkpoint.max_keep_ckpts = 1
    cfg.default_hooks.checkpoint.save_optimizer = True
    cfg.log_processor = dict(by_epoch=True)
    cfg.custom_hooks = []
    assert cfg.train_cfg.max_epochs == 120 and cfg.train_dataloader.batch_size == 8
    assert cfg.randomness.seed == job['seed']
    return cfg

def assert_data(cfg, job):
    counts = {}
    for split in ('train', 'val', 'test'):
        ds = Runner.build_dataloader(cfg[split + '_dataloader'], seed=job['seed']).dataset
        actual = {(str(Path(ds.get_data_info(i)['img_path']).resolve()), str(Path(ds.get_data_info(i)['seg_map_path']).resolve())) for i in range(len(ds))}
        path = ROOT / 'subsets' / f"{job['dataset']}_seed{job['seed']}.json" if split == 'train' else OLD / 'manifests' / f"{job['dataset']}_{split}.json"
        expected = {(r['image'], r['mask']) for r in json.loads(path.read_text())}
        assert actual == expected, (job['name'], split)
        assert len(ds) == len(expected)
        counts[split] = len(ds)
    return counts

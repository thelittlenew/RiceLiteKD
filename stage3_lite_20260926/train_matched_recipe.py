"""Train the new complete model using the old random-student MMSeg recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import torch
from torch.utils.data import DataLoader
from matched_recipe import ROOT, build_config, as_standalone
from mmengine.runner import Runner
import mmengine
import mmseg
from rice_lite import check_budget
from train_seg import SegDataset, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg, baseline = build_config(args.seed)
    output = Path(cfg.work_dir)
    output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        prior = json.loads((output / 'config.json').read_text())
        assert prior['seed'] == args.seed and prior['archived_config_sha256'] == baseline['config_sha256']
        assert (output/'last_checkpoint').exists()
    else:
        cfg.dump(str(output/'frozen_mmseg_config.py'))
    cfg.resume = args.resume
    runner = Runner.from_cfg(cfg)
    parameters = check_budget(runner.model)
    assert parameters == 734186
    settings = {'seed': args.seed, 'parameters': parameters, 'counts': {'train':318,'val':82},
                'teacher_loaded_strictly': False, 'init': None, 'epochs':120,
                'recipe': 'archived_random_student_native_mmseg',
                'archived_config_sha256': baseline['config_sha256'],
                'frozen_config_sha256': hashlib.sha256((output/'frozen_mmseg_config.py').read_bytes()).hexdigest(),
                'versions': {'torch':torch.__version__,'mmseg':mmseg.__version__,'mmengine':mmengine.__version__},
                'manifest_sha256': {kind:hashlib.sha256((ROOT/f'manifests/RiceSEG_{kind}.json').read_bytes()).hexdigest()
                                    for kind in ['train','val']},
                'dropout_ratio':0.1, 'resume_format':'native MMEngine checkpoint; use this trainer --resume',
                'baseline':baseline, 'test_evaluated':False}
    if not args.resume:
        (output/'config.json').write_text(json.dumps(settings, indent=2))
    started = time.time()
    runner.train()
    rows = {}
    for scalar_file in sorted(output.glob('*/vis_data/scalars.json')):
        for line in scalar_file.read_text().splitlines():
            row = json.loads(line)
            if 'mIoU' in row:
                rows[int(row['step'])] = {'epoch':int(row['step']), 'miou':row['mIoU']/100,
                    'panicle_iou':None, 'mmseg_metrics_percent':row}
    assert sorted(rows) == list(range(1,121)), 'Expected 120 validation epochs'
    candidates = list(output.glob('best_mIoU_epoch_*.pth'))
    assert len(candidates) == 1, candidates
    best_native = candidates[0]
    epoch = int(re.search(r'epoch_(\d+)\.pth', best_native.name).group(1))
    checkpoint = torch.load(best_native, map_location='cpu', weights_only=False)
    model = as_standalone(checkpoint['state_dict']).cuda().eval()
    loader = DataLoader(SegDataset(ROOT/'manifests/RiceSEG_val.json'), batch_size=8, num_workers=4)
    metrics = evaluate(model, loader, torch.device('cuda'))
    logged = rows[epoch]['miou']
    assert abs(metrics['miou']-logged) < 0.00015, (metrics, logged)
    best = {**rows[epoch], **metrics, 'mmseg_selected_miou':logged,
            'selection':'native MMSeg validation mIoU checkpoint hook; exact rescore only for selected weights'}
    rows[epoch] = best
    torch.save({'model':{name:tensor.cpu() for name,tensor in model.state_dict().items()},
                'variant':'base','context':True,'epoch':epoch,'validation':metrics,
                'training_recipe':settings['recipe'],'training_dropout_ratio':0.1}, output/'best.pt')
    last = Path((output/'last_checkpoint').read_text().strip())
    assert last.is_file()
    shutil.copyfile(last, output/'resume.pt')
    (output/'history.json').write_text(json.dumps([rows[e] for e in sorted(rows)], indent=2))
    result = {'seconds':time.time()-started, 'best_validation':best,
              'checkpoint':str(output/'best.pt'),
              'checkpoint_sha256':hashlib.sha256((output/'best.pt').read_bytes()).hexdigest(),
              'native_best_checkpoint':str(best_native),
              'native_best_sha256':hashlib.sha256(best_native.read_bytes()).hexdigest(),
              'test_evaluated':False,'parameters':parameters,
              'baseline_validation_miou_percent':baseline['validation_miou_percent'],
              'delta_vs_archived_seed_pp':100*metrics['miou']-baseline['validation_miou_percent']}
    (output/'training_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

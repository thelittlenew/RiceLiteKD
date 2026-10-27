"""Fully random teacher control; preserves the existing downstream recipe."""
from pathlib import Path
import argparse, contextlib, fcntl, hashlib, json, os, re, shutil, statistics
import subprocess, sys, time, traceback

ROOT = Path('/root/autodl-tmp/work/random_teacher_20261016')
WORK = ROOT.parent
MM = WORK / 'baseline_src/mmsegmentation'
sys.path.insert(0, str(MM))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_no_pretraining(cfg):
    assert cfg.load_from is None and not cfg.resume
    def visit(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ('pretrained', 'checkpoint'):
                    assert value is None, (key, value)
                if key == 'type':
                    assert value != 'Pretrained', obj
                visit(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                visit(value)
    visit(cfg.model)


def prepare():
    from mmengine.config import Config
    from mmseg.utils import register_all_modules
    from mmseg.registry import MODELS
    from mmengine.runner import set_random_seed
    import torch
    register_all_modules()
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT / 'plan.json').exists(), 'Existing plan must be reused'
    jobs = []
    expected_counts = {'RiceSEG': [318, 82, 63], 'CVRP_full': [1644, 183, 460]}
    for dataset in ['RiceSEG', 'CVRP_full']:
        source_root = WORK / ('cea_public_20260915' if dataset == 'RiceSEG' else 'cea_cvrp_full_20260923')
        for seed in [42, 7]:
            source = source_root / 'configs' / f'{dataset}_stage2_seed{seed}.py'
            cfg = Config.fromfile(str(source))
            name = f'{dataset}_random_teacher_seed{seed}'
            cfg.load_from = None
            cfg.resume = False
            cfg.work_dir = str(ROOT / 'runs' / name)
            check_no_pretraining(cfg)
            assert cfg.randomness.seed == seed
            assert cfg.train_cfg.max_epochs == 120
            assert cfg.train_dataloader.batch_size == 8
            counts = []
            for split in ['train', 'val', 'test']:
                dc = cfg[split + '_dataloader'].dataset
                image_dir = Path(dc.data_root) / dc.data_prefix.img_path
                mask_dir = Path(dc.data_root) / dc.data_prefix.seg_map_path
                files = sorted(image_dir.glob('*' + dc.img_suffix))
                assert all((mask_dir / (p.stem + dc.seg_map_suffix)).is_file() for p in files)
                counts.append(len(files))
            assert counts == expected_counts[dataset], (dataset, counts)
            target = ROOT / 'configs' / f'{name}.py'
            target.parent.mkdir(exist_ok=True)
            cfg.dump(str(target))
            copied = ROOT / 'source_configs' / source.name
            copied.parent.mkdir(exist_ok=True)
            shutil.copy2(source, copied)
            jobs.append(dict(name=name, dataset=dataset, seed=seed, epochs=120,
                             config=str(target), config_sha256=sha(target),
                             source_config=str(source), source_config_sha256=sha(source), counts=counts))
    set_random_seed(42)
    model = MODELS.build(Config.fromfile(jobs[0]['config']).model)
    model.init_weights()
    params = sum(p.numel() for p in model.parameters())
    assert 24_000_000 < params < 24_020_000, params
    assert all(p.requires_grad for p in model.parameters())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    plan = dict(created=time.time(), seeds=[42, 7], parameters=params, jobs=jobs,
                initialization='No checkpoint: random ViT, adapter and segmentation head; standard bias/norm initialization',
                training='Existing teacher downstream configs, only load_from and output directory changed',
                selection='Best validation mIoU checkpoint; test is never used for selection',
                source_training=False, official_dinov3_weights=False,
                gpu=torch.cuda.get_device_name(), free_disk_bytes=shutil.disk_usage(ROOT).free)
    write_json(ROOT / 'plan.json', plan)
    write_json(ROOT / 'preflight.json', dict(state='passed', parameters=params, all_trainable=True,
                                             all_finite=True, jobs=len(jobs), time=time.time()))
    print(json.dumps(dict(state='prepared', jobs=len(jobs), parameters=params)), flush=True)


def measures(matrix):
    import torch
    x = matrix.double()
    tp = x.diag()
    iou = tp / (x.sum(0) + x.sum(1) - tp)
    dice = 2 * tp / (x.sum(0) + x.sum(1))
    return dict(miou=iou.mean().item(), panicle_iou=iou[1].item(), mean_dice=dice.mean().item(),
                class_iou=iou.tolist(), class_dice=dice.tolist(), confusion_matrix=matrix.tolist())


def score(model, cfg, split, job, run):
    import torch
    import numpy as np
    from PIL import Image
    from mmengine.runner import Runner
    loader = Runner.build_dataloader(cfg[split + '_dataloader'], seed=job['seed'])
    matrix = torch.zeros(2, 2, dtype=torch.int64)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            for sample in model.test_step(batch):
                gt = sample.gt_sem_seg.data.squeeze(0).long()
                pred = sample.pred_sem_seg.data.squeeze(0).long()
                assert gt.shape == pred.shape
                valid = gt != 255
                assert torch.isin(gt, torch.tensor([0, 1, 255], device=gt.device)).all()
                counts = torch.bincount(2 * gt[valid] + pred[valid], minlength=4).reshape(2, 2).cpu()
                matrix += counts
                rows.append(dict(image=sample.img_path, **measures(counts)))
                if split == 'test' and job['seed'] == 42:
                    dst = run / 'predictions' / (Path(sample.img_path).stem + '.png')
                    dst.parent.mkdir(exist_ok=True)
                    Image.fromarray((pred.cpu().numpy() * 255).astype(np.uint8)).save(dst)
    expected = job['counts'][1 if split == 'val' else 2]
    assert len(rows) == expected, (split, len(rows), expected)
    write_json(run / f'{split}_per_image.json', rows)
    return dict(images=len(rows), **measures(matrix))


def run_one(name):
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmseg.utils import register_all_modules
    from mmseg.registry import MODELS
    torch.set_num_threads(4)
    register_all_modules()
    plan = json.loads((ROOT / 'plan.json').read_text())
    job = next(j for j in plan['jobs'] if j['name'] == name)
    assert sha(job['config']) == job['config_sha256']
    cfg = Config.fromfile(job['config'])
    check_no_pretraining(cfg)
    run = Path(cfg.work_dir)
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'result.json').exists():
        return
    started = time.time()
    if not (run / 'training_done.json').exists():
        cfg.resume = (run / 'last_checkpoint').exists()
        runner = Runner.from_cfg(cfg)
        assert sum(p.numel() for p in runner.model.parameters()) == plan['parameters']
        write_json(run / 'initialization_audit.json', dict(load_from=cfg.load_from, resume=cfg.resume,
                   model_pretrained=cfg.model.get('pretrained'), parameters=plan['parameters'], seed=job['seed']))
        runner.train()
        write_json(run / 'training_done.json', dict(time=time.time(), seconds=time.time() - started))
        del runner
        __import__('gc').collect()
        torch.cuda.empty_cache()
    best = list(run.glob('best_mIoU_epoch_*.pth'))
    assert len(best) == 1, best
    checkpoint = best[0]
    history = {}
    for scalar in sorted(run.glob('*/vis_data/scalars.json')):
        for line in scalar.read_text().splitlines():
            row = json.loads(line)
            if 'mIoU' in row:
                history[int(row['step'])] = row
    assert sorted(history) == list(range(1, 121)), sorted(history)
    epoch = int(checkpoint.stem.rsplit('_', 1)[1])
    assert history[epoch]['mIoU'] == max(r['mIoU'] for r in history.values())
    write_json(run / 'history.json', [history[i] for i in sorted(history)])
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = MODELS.build(cfg.model)
    model.load_state_dict(state['state_dict'], strict=True)
    model = model.cuda().eval()
    val = score(model, cfg, 'val', job, run)
    assert abs(val['miou'] - history[epoch]['mIoU'] / 100) < 0.00015
    selection = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                     epoch=epoch, validation_miou=val['miou'], criterion='validation mIoU only')
    write_json(run / 'selection.json', selection)
    test = score(model, cfg, 'test', job, run)
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, selection=selection),
               run / 'best_model.pth')
    result = dict(**job, state='complete', seconds=time.time() - started, time=time.time(),
                  validation=val, test=test, selection=selection, parameters=plan['parameters'],
                  plan_sha256=sha(ROOT / 'plan.json'), checkpoint_sha256=sha(run / 'best_model.pth'),
                  test_used_for_selection=False)
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


def summarize():
    plan = json.loads((ROOT / 'plan.json').read_text())
    results = [json.loads(p.read_text()) for p in sorted((ROOT / 'runs').glob('*/result.json'))]
    groups = []
    for dataset in ['RiceSEG', 'CVRP_full']:
        rows = [r for r in results if r['dataset'] == dataset]
        if rows:
            metrics = {k: dict(mean_percent=statistics.mean(r['test'][k] * 100 for r in rows),
                               sample_sd_pp=statistics.stdev(r['test'][k] * 100 for r in rows) if len(rows) > 1 else None)
                       for k in ['miou', 'panicle_iou', 'mean_dice']}
            groups.append(dict(dataset=dataset, n=len(rows), seeds=[r['seed'] for r in rows], metrics=metrics))
    summary = dict(planned=len(plan['jobs']), completed=len(results), groups=groups, results=results)
    write_json(ROOT / 'summary.json', summary)
    return summary


def queue():
    lock = open(ROOT / 'queue.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((ROOT / 'plan.json').read_text())
    failures = []
    for job in plan['jobs']:
        run = ROOT / 'runs' / job['name']
        if (run / 'result.json').exists():
            continue
        if shutil.disk_usage(ROOT).free < 2 * 1024 ** 3:
            write_json(ROOT / 'status.json', dict(state='blocked_disk', time=time.time()))
            return 2
        run.mkdir(exist_ok=True, parents=True)
        write_json(ROOT / 'status.json', dict(state='training', job=job['name'], time=time.time(),
                                              completed=summarize()['completed'], total=len(plan['jobs'])))
        with open(run / 'run.log', 'a') as output:
            rc = subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()), 'run', job['name']],
                                 cwd=str(MM), stdout=output, stderr=subprocess.STDOUT)
        if rc or not (run / 'result.json').exists():
            failures.append(job['name'])
            write_json(run / 'failed.json', dict(returncode=rc, time=time.time()))
        summarize()
    summary = summarize()
    write_json(ROOT / 'status.json', dict(state='complete' if not failures else 'failed',
               completed=summary['completed'], total=len(plan['jobs']), failures=failures, time=time.time()))
    return int(bool(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'queue', 'summary'])
    parser.add_argument('name', nargs='?')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'run': run_one(args.name)
    elif args.action == 'queue': sys.exit(queue())
    else: print(json.dumps(summarize()))

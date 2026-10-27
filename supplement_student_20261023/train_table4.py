import argparse, gc, time
from table4_common import *
from train_downstream import ProgressHook

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('--resume', action='store_true'); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--tail-smoke', action='store_true')
    args = ap.parse_args()
    smoke = args.smoke or args.tail_smoke
    plan = verify_plan()
    job = next(j for j in plan['jobs'] if j['name'] == args.name)
    cfg = table4_config(job)
    counts = assert_data(cfg, job)
    out = Path(cfg.work_dir)
    if smoke:
        out = ROOT / ('preflight_singleton' if args.tail_smoke else 'preflight_table4') / job['model']
        cfg.work_dir = str(out)
        cfg.train_cfg.max_epochs = 2
        if not args.tail_smoke: cfg.train_dataloader.dataset.indices = list(range(16))
        cfg.val_dataloader.dataset.indices = list(range(16))
    out.mkdir(parents=True, exist_ok=True)
    if (out / ('smoke_complete.json' if smoke else 'result.json')).exists(): return
    init = cfg.load_from
    probe = MODELS.build(cfg.model)
    parameters = sum(p.numel() for p in probe.parameters())
    assert parameters == job['parameters'], (parameters, job)
    if init:
        state = torch.load(init, map_location='cpu', weights_only=False)
        state = state.get('state_dict', state)
        assert state and all(k.startswith('backbone.') for k in state), 'Expected source backbone only'
        loaded = probe.load_state_dict(state, strict=False)
        assert not loaded.unexpected_keys
        assert not any(k.startswith('backbone.') and not k.endswith('num_batches_tracked') for k in loaded.missing_keys), loaded
    del probe
    cfg.resume = args.resume
    if args.resume:
        cfg.load_from = (out / 'last_checkpoint').read_text().strip()
        assert Path(cfg.load_from).is_file()
    cfg.dump(str(out / 'effective_config.py'))
    write_json(out / 'protocol.json', dict(job=job, counts=counts, smoke=smoke,
        plan_sha256=sha(ROOT / 'plan.json'), initialization=init or 'random',
        initialization_sha256=sha(init) if init else None,
        comparison='Archived model-specific full-data recipe; same 20% subset across models per seed'))
    runner = Runner.from_cfg(cfg)
    runner.register_hook(ProgressHook())
    if job.get('avoid_singleton_batch'):
        batches = list(runner.train_dataloader.batch_sampler)
        flat = [i for batch in batches for i in batch]
        assert len(flat) == len(set(flat)) == len(runner.train_dataloader.dataset)
        assert set(flat) == set(range(len(runner.train_dataloader.dataset)))
        assert min(map(len, batches)) >= 2 and max(map(len, batches)) <= 8
        assert len(batches) == len(runner.train_dataloader)
        if len(flat) == 329: assert list(map(len, batches[-2:])) == [7, 2] and len(batches) == 42
        write_json(out / 'batching.json', dict(images=len(flat),updates_per_epoch=len(batches),batch_sizes=list(map(len,batches)),no_omission=True,no_duplication=True))
    start = time.time()
    runner.train()
    checkpoints = list(out.glob('best_mIoU_epoch_*.pth'))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    history = {}
    for path in sorted(out.glob('*/vis_data/scalars.json')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if 'mIoU' in row: history[int(row['step'])] = row
    assert sorted(history) == list(range(1, 3 if smoke else 121)), sorted(history)
    write_json(out / 'history.json', [history[k] for k in sorted(history)])
    del runner
    gc.collect(); torch.cuda.empty_cache()
    if smoke:
        write_json(out / 'smoke_complete.json', dict(passed=True, model=job['model'], epochs=2, parameters=parameters, seconds=time.time()-start))
        return
    cfg = table4_config(job)
    model = revert_sync_batchnorm(MODELS.build(cfg.model))
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state, strict=True); model.cuda().float().eval()
    val = score(model, cfg, 'val', job)
    epoch = int(checkpoint.stem.rsplit('_', 1)[1])
    assert abs(val['miou'] - history[epoch]['mIoU']/100) < .00015
    selection = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), validation_miou=val['miou'], epoch=epoch, rule='validation mIoU only')
    write_json(out / 'selection.json', selection)
    test = score(model, cfg, 'test', job)
    save_torch(out / 'best_model.pth', dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()}, selection=selection, model=job['model']))
    exported = torch.load(out / 'best_model.pth', map_location='cpu', weights_only=False)['state_dict']
    assert exported.keys() == state.keys() and all(torch.equal(v.cpu(), exported[k]) for k,v in state.items())
    result = dict(**job, state='complete', time=time.time(), seconds=time.time()-start, validation=val, test=test, selection=selection,
        plan_sha256=sha(ROOT / 'plan.json'), test_used_for_selection=False, checkpoint_sha256=sha(out / 'best_model.pth'))
    write_json(out / 'result.json', result)
    write_json(out / 'progress.json', dict(state='complete', epoch=120, time=time.time()))
    # Retire only this completed run's duplicate optimizer snapshots after verifying the exported weights.
    retired = [dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in list(out.glob('epoch_*.pth')) + list(out.glob('best_mIoU_epoch_*.pth'))]
    write_json(out / 'retired_optimizer_checkpoints.json', retired)
    for row in retired:
        path = Path(row['path']); assert path.resolve().parent == out.resolve(); path.unlink()
    print('TABLE4_COMPLETE', job['name'], test['miou'], flush=True)

if __name__ == '__main__': main()

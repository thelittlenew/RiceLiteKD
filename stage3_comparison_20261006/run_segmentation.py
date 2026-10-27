"""Train or reuse a frozen run, then evaluate only its validation-selected checkpoint."""
from comparison_common import *
from mmengine.runner import Runner
from PIL import Image
import argparse,shutil,time,statistics
import numpy as np

def measures(matrix):
    x=matrix.double();tp=x.diag();iou=tp/(x.sum(0)+x.sum(1)-tp);dice=2*tp/(x.sum(0)+x.sum(1))
    return dict(miou=iou.mean().item(),panicle_iou=iou[1].item(),mean_dice=dice.mean().item(),
                class_iou=iou.tolist(),class_dice=dice.tolist(),confusion_matrix=matrix.tolist())

@torch.inference_mode()
def score(model,cfg,split,job):
    loader=Runner.build_dataloader(cfg[split+'_dataloader'],seed=job['seed'])
    manifest=json.loads((ROOT/'manifests'/f"{job['dataset']}_{split}.json").read_text())
    actual={(str(Path(loader.dataset.get_data_info(i)['img_path']).resolve()),str(Path(loader.dataset.get_data_info(i)['seg_map_path']).resolve())) for i in range(len(loader.dataset))}
    assert actual=={(r['image'],r['mask']) for r in manifest}
    matrix=torch.zeros(2,2,dtype=torch.int64);rows=[]
    for batch in loader:
        for sample in model.test_step(batch):
            gt=sample.gt_sem_seg.data.squeeze(0).long();pred=sample.pred_sem_seg.data.squeeze(0).long()
            assert gt.shape==pred.shape
            valid=gt!=255;assert torch.isin(gt,torch.tensor([0,1,255],device=gt.device)).all()
            counts=torch.bincount(2*gt[valid]+pred[valid],minlength=4).reshape(2,2).cpu()
            matrix+=counts;rows.append(dict(image=sample.img_path,shape=list(gt.shape),**measures(counts)))
            if split=='test' and job['seed']==42:
                dst=ROOT/'predictions'/job['name']/Path(sample.img_path).with_suffix('.png').name
                dst.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray((pred.cpu().numpy()*255).astype(np.uint8)).save(dst)
    assert len(rows)==len(manifest)
    return dict(images=len(rows),per_image=rows,**measures(matrix))

@torch.inference_mode()
def benchmark(model):
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x=torch.randn(1,3,512,512,device='cuda');meta=[dict(img_shape=(512,512),ori_shape=(512,512),pad_shape=(512,512))]
    for _ in range(50):model.encode_decode(x,meta)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        for _ in range(100):model.encode_decode(x,meta)
        end.record();torch.cuda.synchronize();times.append(start.elapsed_time(end)/100)
    return dict(gpu=torch.cuda.get_device_name(),protocol='FP32 TF32off batch1 512x512, 50 warmup, 5x100 CUDA-event timed encode_decode forwards; excludes data loading/preprocessing',
                repeat_mean_ms=times,median_ms=statistics.median(times),fps=1000/statistics.median(times),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('name');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    plan=verify_plan();job=next(j for j in plan['jobs'] if j['name']==args.name);assert job['kind']=='segmentation'
    torch.set_num_threads(4);output=ROOT/'runs'/job['name'];output.mkdir(exist_ok=True,parents=True)
    if (output/'result.json').exists():return
    cfg=Config.fromfile(str(ROOT/'configs'/f"{job['name']}.py"));start=time.time()
    if 'reuse_run' in job:
        checkpoint=Path(job['checkpoint']);assert sha(checkpoint)==job['checkpoint_sha256']
        validation_expected=job['validation_miou'];training_origin=job['reuse_run']
    else:
        cfg.resume=args.resume
        runner=Runner.from_cfg(cfg)
        assert sum(p.numel() for p in runner.model.parameters())==956852
        runner.train()
        best=list(output.glob('best_mIoU_epoch_*.pth'));assert len(best)==1
        checkpoint=best[0];training_origin=str(output)
        rows={}
        for p in sorted(output.glob('*/vis_data/scalars.json')):
            for line in p.read_text().splitlines():
                r=json.loads(line)
                if 'mIoU' in r:rows[int(r['step'])]=r
        assert sorted(rows)==list(range(1,121))
        epoch=int(checkpoint.stem.rsplit('_',1)[1]);validation_expected=rows[epoch]['mIoU']/100
        write_json(output/'history.json',[rows[i] for i in sorted(rows)])
        del runner;__import__('gc').collect();torch.cuda.empty_cache()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=MODELS.build(cfg.model);model.load_state_dict(state['state_dict'],strict=True);model=model.cuda().eval()
    params=sum(p.numel() for p in model.parameters());assert params==956852
    val=score(model,cfg,'val',job);assert abs(val['miou']-validation_expected)<.00015,(val['miou'],validation_expected)
    # Immutable selection evidence is written before accessing test labels.
    selection=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),validation_miou=val['miou'],selection='validation mIoU only')
    write_json(output/'selection.json',selection)
    test=score(model,cfg,'test',job)
    atomic_save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},selection=selection,architecture='CGNetRefine7NativeSeg'),output/'best_model.pth')
    speed=benchmark(model) if job['seed']==42 else None
    result=dict(state='complete',time=time.time(),seconds=time.time()-start,training_origin=training_origin,
                validation=val,test=test,selection=selection,parameters=params,benchmark=speed,plan_sha256=sha(ROOT/'plan.json'),
                checkpoint_sha256=sha(output/'best_model.pth'),test_used_for_selection=False)
    result={**job,**result}
    write_json(output/'result.json',result)
    print(json.dumps({k:result[k] for k in ['name','state','seconds','parameters']}),json.dumps({k:test[k] for k in ['miou','panicle_iou','mean_dice']}),flush=True)

if __name__=='__main__':main()

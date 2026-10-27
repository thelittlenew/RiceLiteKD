from common import *
from mmengine.runner import Runner
from mmengine.model import revert_sync_batchnorm
from PIL import Image
import argparse,gc,statistics,numpy as np

def metrics(matrix):
 x=matrix.double();tp=x.diag();iou=tp/(x.sum(0)+x.sum(1)-tp).clamp_min(1);dice=2*tp/(x.sum(0)+x.sum(1)).clamp_min(1)
 return dict(miou=iou.mean().item(),panicle_iou=iou[1].item(),mean_dice=dice.mean().item(),precision=(tp[1]/x[:,1].sum().clamp_min(1)).item(),recall=(tp[1]/x[1,:].sum().clamp_min(1)).item(),confusion_matrix=matrix.tolist())

@torch.inference_mode()
def score(model,cfg,split,job):
 loader=Runner.build_dataloader(cfg[split+'_dataloader'],seed=job['seed'])
 rows=json.loads((ROOT/'manifests'/f"{job['dataset']}_{split}.json").read_text())
 actual={(str(Path(loader.dataset.get_data_info(i)['img_path']).resolve()),str(Path(loader.dataset.get_data_info(i)['seg_map_path']).resolve())) for i in range(len(loader.dataset))}
 assert actual=={(r['image'],r['mask']) for r in rows}
 total=torch.zeros((2,2),dtype=torch.int64);per_image=[]
 for batch in loader:
  for sample in model.test_step(batch):
   gt=sample.gt_sem_seg.data.squeeze(0).long();pred=sample.pred_sem_seg.data.squeeze(0).long();assert gt.shape==pred.shape
   valid=gt!=255;counts=torch.bincount(2*gt[valid]+pred[valid],minlength=4).reshape(2,2).cpu();total+=counts
   per_image.append(dict(image=sample.img_path,**metrics(counts)))
   if split=='test' and job['seed']==42:
    dst=ROOT/'predictions'/job['name']/Path(sample.img_path).with_suffix('.png').name;dst.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray((pred.cpu().numpy()*255).astype('uint8')).save(dst)
 assert len(per_image)==len(rows)
 return dict(images=len(rows),per_image=per_image,**metrics(total))

@torch.inference_mode()
def benchmark(model):
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 x=torch.randn(1,3,512,512,device='cuda');meta=[dict(img_shape=(512,512),ori_shape=(512,512),pad_shape=(512,512))]
 for _ in range(50):model.encode_decode(x,meta)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();timings=[]
 for _ in range(5):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(100):model.encode_decode(x,meta)
  b.record();torch.cuda.synchronize();timings.append(a.elapsed_time(b)/100)
 return dict(gpu=torch.cuda.get_device_name(),protocol='FP32 TF32off batch1 512x512, 50 warmup then 5x100 CUDA-event encode_decode forwards; data loading, preprocessing and H2D excluded',repeat_mean_ms=timings,median_ms=statistics.median(timings),fps=1000/statistics.median(timings),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('name');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 plan=verify_plan();job=next(j for j in plan['jobs'] if j['name']==args.name);out=ROOT/'runs'/args.name;out.mkdir(parents=True,exist_ok=True)
 assert not (out/'result.json').exists()
 torch.set_num_threads(4);cfg=Config.fromfile(str(ROOT/'configs'/(args.name+'.py')));cfg.resume=args.resume;start=time.time()
 runner=Runner.from_cfg(cfg);runner.train()
 best=list(out.glob('best_mIoU_epoch_*.pth'));assert len(best)==1
 best=best[0];epoch=int(best.stem.rsplit('_',1)[1])
 history={int(k):v for k,v in json.loads((out/'epoch_validation.json').read_text()).items()}
 assert sorted(history)==list(range(1,121)),sorted(history)
 write_json(out/'validation_history.json',[history[i] for i in sorted(history)])
 del runner;gc.collect();torch.cuda.empty_cache()
 state=torch.load(best,map_location='cpu',weights_only=False)
 model=revert_sync_batchnorm(MODELS.build(cfg.model));model.load_state_dict(state['state_dict'],strict=True);model=model.cuda().float().eval()
 parameters=sum(p.numel() for p in model.parameters())
 val=score(model,cfg,'val',job);assert abs(val['miou']-history[epoch]['mIoU']/100)<.00015
 selection=dict(checkpoint=str(best),sha256=sha(best),epoch=epoch,validation_miou=val['miou'],rule='validation mIoU only',time=time.time())
 write_json(out/'selection.json',selection)
 test=score(model,cfg,'test',job)
 atomic_save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},selection=selection,model=job['model']),out/'best_model.pth')
 exported=torch.load(out/'best_model.pth',map_location='cpu',weights_only=False)
 assert exported['state_dict'].keys()==state['state_dict'].keys()
 assert all(torch.equal(v.cpu(),exported['state_dict'][k]) for k,v in state['state_dict'].items())
 del exported,state
 speed=None;deploy=None
 if job['seed']==42:
  speed=benchmark(model)
  if job['model']=='GCNet_M':
   model.backbone.switch_to_deploy();deploy_score=score(model,cfg,'val',job)
   assert abs(deploy_score['miou']-val['miou'])<.0001
   deploy=dict(parameters=sum(p.numel() for p in model.parameters()),benchmark=benchmark(model),validation_miou=deploy_score['miou'])
   atomic_save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},selection=selection,deploy=True),out/'deploy_model.pth')
 result=dict(**job,state='complete',seconds=time.time()-start,parameters=parameters,validation=val,test=test,selection=selection,benchmark=speed,deploy=deploy,checkpoint_sha256=sha(out/'best_model.pth'),plan_sha256=sha(ROOT/'plan.json'),test_used_for_selection=False,time=time.time())
 write_json(out/'result.json',result)
 # Retain verified final model weights and all metrics; completed jobs no longer
 # require duplicate optimizer snapshots. Never touch another run directory.
 intermediates=[]
 for path in list(out.glob('epoch_*.pth'))+list(out.glob('best_mIoU_epoch_*.pth')):
  assert path.resolve().parent==out.resolve()
  intermediates.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
 write_json(out/'retired_optimizer_checkpoints.json',intermediates)
 for record in intermediates:Path(record['path']).unlink()
 print('COMPLETE',args.name,json.dumps({k:test[k] for k in ['miou','panicle_iou','mean_dice']}),flush=True)
if __name__=='__main__':main()

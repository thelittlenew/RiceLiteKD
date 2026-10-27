import argparse,time,json,gc
from experiment_common import *
from mmengine.hooks import Hook

class ProgressHook(Hook):
 def after_train_epoch(self,runner):
  write_json(Path(runner.work_dir)/'progress.json',dict(state='training',epoch=runner.epoch+1,epochs=runner.max_epochs,time=time.time(),pid=os.getpid()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('name');ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 plan=verify_plan();job=next(j for j in plan['jobs'] if j['name']==a.name)
 cfg=get_config(job);out=ROOT/'runs'/job['name']
 if a.smoke:
  out=ROOT/'preflight/downstream';cfg.work_dir=str(out);cfg.train_cfg.max_epochs=2
  cfg.train_dataloader.dataset.indices=list(range(16));cfg.val_dataloader.dataset.indices=list(range(16))
  cfg.default_hooks.checkpoint.interval=1
 out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 assert Path(cfg.load_from).is_file()
 state=torch.load(cfg.load_from,map_location='cpu',weights_only=False)['state_dict']
 probe=MODELS.build(cfg.model);assert sum(p.numel() for p in probe.parameters())==956852
 probe.load_state_dict(state,strict=True);del probe,state
 cfg.resume=a.resume
 if a.resume:
  pointer=out/'last_checkpoint';assert pointer.exists();cfg.load_from=pointer.read_text().strip()
 runner=Runner.from_cfg(cfg);runner.register_hook(ProgressHook())
 actual={(str(Path(runner.train_dataloader.dataset.get_data_info(i)['img_path']).resolve()),str(Path(runner.train_dataloader.dataset.get_data_info(i)['seg_map_path']).resolve())) for i in range(len(runner.train_dataloader.dataset))}
 expected_path=ROOT/'subsets'/f"{job['dataset']}_seed{job['seed']}.json" if job['arm']=='distilled_20pct' else OLD/'manifests'/f"{job['dataset']}_train.json"
 expected={(r['image'],r['mask']) for r in json.loads(expected_path.read_text())}
 assert actual<=expected if a.smoke else actual==expected
 write_json(out/'protocol.json',dict(job=job,train_count=len(actual),plan_sha256=sha(ROOT/'plan.json'),init_sha256=sha(get_config(job).load_from),smoke=a.smoke))
 start=time.time();runner.train()
 best=list(out.glob('best_mIoU_epoch_*.pth'));assert len(best)==1
 checkpoint=best[0];history={}
 for p in sorted(out.glob('*/vis_data/scalars.json')):
  for line in p.read_text().splitlines():
   row=json.loads(line)
   if 'mIoU' in row:history[int(row['step'])]=row
 assert len(history)==(2 if a.smoke else 120),len(history)
 write_json(out/'history.json',[history[k] for k in sorted(history)])
 del runner;gc.collect();torch.cuda.empty_cache()
 if a.smoke:
  write_json(out/'smoke_complete.json',dict(state='passed',epochs=2,best=str(checkpoint)));return
 cfg=get_config(job)
 model=MODELS.build(cfg.model);state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 model.load_state_dict(state['state_dict'],strict=True);model.cuda().eval()
 val=score(model,cfg,'val',job);ep=int(checkpoint.stem.rsplit('_',1)[1])
 assert abs(val['miou']-history[ep]['mIoU']/100)<.00015
 selection=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),validation_miou=val['miou'],epoch=ep,rule='validation mIoU only')
 write_json(out/'selection.json',selection)
 test=score(model,cfg,'test',job)
 save_torch(out/'best_model.pth',dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},selection=selection,architecture='CGNetRefine7NativeSeg'))
 result=dict(**job,state='complete',time=time.time(),seconds=time.time()-start,parameters=956852,validation=val,test=test,selection=selection,plan_sha256=sha(ROOT/'plan.json'),test_used_for_selection=False)
 write_json(out/'result.json',result);write_json(out/'progress.json',dict(state='complete',epoch=120,time=time.time()))
 print('DOWNSTREAM_COMPLETE',job['name'],test['miou'],flush=True)

if __name__=='__main__':main()

"""Full fixed source budget, deterministic per-epoch sampling and mid-epoch recovery."""
import argparse,hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from full_distill_components import Refine7DistillStudent,FullSourceDataset,FullDistillationLoss,class_weights
from train_seg import build_teacher,teacher_forward,evaluate
ROOT=Path(__file__).resolve().parent
DATA=Path('/root/stage3_full_source_20260926')
RUN=ROOT/'runs/SourceFull_refine7_distill_120ep_seed42'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def save_torch(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
def permitted():
    scope=json.loads((ROOT/'scope.json').read_text())
    return scope.get('allow_full_source_distillation',False) and not (ROOT/'STOP_AFTER_CURRENT').exists()
def lr(epoch):
    return 3e-4*epoch/5 if epoch<=5 else 3e-6+.5*(3e-4-3e-6)*(1+math.cos(math.pi*(epoch-5)/115))

def main():
    global RUN,DATA
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--smoke-stop-after-step',type=int,default=0)
    parser.add_argument('--smoke-data-root')
    args=parser.parse_args()
    assert args.smoke or not args.smoke_stop_after_step
    assert args.smoke or not args.smoke_data_root
    if args.smoke_data_root:DATA=Path(args.smoke_data_root)
    max_epochs=2 if args.smoke else 120
    if args.smoke:RUN=ROOT/'preflight_runs/full_refine7_resume_chain'
    assert permitted()
    plan_path=ROOT/'full_distillation_jobs.json';plan=json.loads(plan_path.read_text());plan_sha=sha(plan_path)
    assert plan['source_epochs']==120 and plan['student_parameters']==956852
    for name,digest in plan['code_sha256'].items():assert sha(ROOT/name)==digest,name
    ready=json.loads((DATA/'DATA_READY.json').read_text())
    if not args.smoke:assert ready['paired_total']==103817 and ready['pairs']+ready['excluded_count']==103817 and ready['pairs']>=100000
    assert sha(DATA/'PREPARED.json')==ready['prepared_sha256']
    for name in ['teacher_source','teacher_weights']:assert sha(plan[name])==plan[name+'_sha256']
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    RUN.mkdir(parents=True,exist_ok=args.resume)
    counts={'train':32,'val':16} if args.smoke else ready['counts']
    settings={'seed':42,'architecture':'CGNetRefine7NativeSeg','parameters':956852,'counts':counts,
              'epochs':max_epochs,'batch_size':8,'init':None,'teacher_loaded_strictly':True,'test_evaluated':False,'not_formal_training':args.smoke,
              'recipe':'full_source_refine7_distillation_v1','data_prepared_sha256':ready['prepared_sha256'],
              'plan_sha256':plan_sha,'plan':plan,'data_root':str(DATA)}
    if args.resume:
        assert json.loads((RUN/'config.json').read_text())['plan_sha256']==plan_sha
    else:save_json(RUN/'config.json',settings)
    student=Refine7DistillStudent();student.native.init_weights();student=student.cuda().train()
    assert sum(p.numel() for p in student.parameters())==956852
    teacher=build_teacher(plan['teacher_source'],plan['teacher_weights'],'cuda')
    teacher.requires_grad_(False).eval()
    weights=class_weights(json.loads((DATA/'inventory.json').read_text())).cuda()
    criterion=FullDistillationLoss(weights).cuda()
    optimizer=torch.optim.AdamW(student.parameters(),lr=3e-4,weight_decay=.05)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    history=[];best=-1.;start_epoch=1;start_batch=0;global_step=0;totals={};seen=0;skipped_updates=0;consecutive_skips=0
    if args.resume:
        state=torch.load(RUN/'resume.pt',map_location='cpu',weights_only=False)
        assert state['plan_sha256']==plan_sha
        student.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        history=state['history'];best=state['best'];start_epoch=state['epoch'];start_batch=state['next_batch'];global_step=state['global_step']
        totals=state['loss_totals'];seen=state['seen']
        skipped_updates=state.get('skipped_updates',0)
        torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state_all(state['rng_cuda'])
    started=time.time();last_save=time.time();previous=started
    def checkpoint(epoch,next_batch):
        save_torch(RUN/'resume.pt',{'model':student.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'epoch':epoch,'next_batch':next_batch,'global_step':global_step,'history':history,'best':best,
            'loss_totals':totals,'seen':seen,'skipped_updates':skipped_updates,'plan_sha256':plan_sha,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()})
    if not args.resume:checkpoint(1,0)
    val_data=FullSourceDataset(DATA/'val.json')
    if args.smoke:val_data.rows=val_data.rows[:16]
    val_loader=DataLoader(val_data,batch_size=8,num_workers=4,pin_memory=True)
    for epoch in range(start_epoch,max_epochs+1):
        assert sha(plan_path)==plan_sha
        student.train()
        for param in optimizer.param_groups:param['lr']=lr(epoch)
        dataset=FullSourceDataset(DATA/'train.json',training=True,epoch=epoch)
        if args.smoke:dataset.rows=dataset.rows[:32]
        order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(42+epoch)).tolist()
        skip=start_batch if epoch==start_epoch else 0
        loader=DataLoader(Subset(dataset,order[skip*8:]),batch_size=8,num_workers=4,pin_memory=True)
        if not skip:totals={};seen=0
        total_batches=math.ceil(len(dataset)/8)
        for batch,(x,y) in enumerate(loader,start=skip):
            if not permitted():checkpoint(epoch,batch);print('STOP_REQUEST_SAVED',flush=True);return 75
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            with torch.no_grad():target=teacher_forward(teacher,x) # FP32 teacher; AMP only student.
            assert all(torch.isfinite(v).all() for v in [target['logits'],*target['features']]),'Nonfinite teacher'
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):losses=criterion(student(x),target,y)
            assert all(torch.isfinite(v) for v in losses.values()),'Nonfinite distillation loss'
            scaler.scale(losses['total']).backward();scaler.unscale_(optimizer)
            finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
            torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
            scaler.step(optimizer);scaler.update()
            if not finite:
                skipped_updates+=1;consecutive_skips+=1
                print('AMP_SKIPPED_UPDATE',json.dumps({'epoch':epoch,'batch':batch+1,'scale':scaler.get_scale(),'skipped_updates':skipped_updates}),flush=True)
                assert consecutive_skips<8,'Repeated nonfinite gradients; retained checkpoint requires diagnosis'
            else:consecutive_skips=0
            global_step+=1;seen+=len(x)
            for k,v in losses.items():totals[k]=totals.get(k,0.)+v.item()*len(x)
            if global_step%50==0:
                now=time.time();status={'state':'training','epoch':epoch,'epochs':120,'batch':batch+1,'batches':total_batches,
                    'global_step':global_step,'losses':{k:v.item() for k,v in losses.items()},'lr':lr(epoch),
                    'seconds_per_recent_step':(now-previous)/50,'time':now,'train_pid':os.getpid(),'best_source_miou':best,
                    'skipped_amp_updates':skipped_updates,'amp_scale':scaler.get_scale()}
                previous=now;save_json(ROOT/'full_distillation_progress.json',status);print(json.dumps(status),flush=True)
            if global_step%500==0 or time.time()-last_save>600:
                checkpoint(epoch,batch+1);last_save=time.time()
            if args.smoke and args.smoke_stop_after_step and global_step>=args.smoke_stop_after_step:
                checkpoint(epoch,batch+1);print('INTENTIONAL_PREFLIGHT_RESUME_BOUNDARY',flush=True);return 75
        checkpoint(epoch,total_batches)
        metrics=evaluate(student,val_loader,torch.device('cuda'))
        row={'epoch':epoch,**metrics,'losses':{k:v/seen for k,v in totals.items()},'lr':lr(epoch),'global_step':global_step,'skipped_amp_updates':skipped_updates}
        history.append(row)
        if row['miou']>best:
            best=row['miou'];save_torch(RUN/'best.pt',{'architecture':'CGNetRefine7NativeSeg','model':student.state_dict(),
                'epoch':epoch,'validation':metrics,'source_pretrained':True,'parameters':956852,'plan_sha256':plan_sha})
        save_json(RUN/'history.json',history);totals={};seen=0;checkpoint(epoch+1,0);last_save=time.time()
        print('SOURCE_EPOCH_COMPLETE',json.dumps(row),flush=True)
    assert len(history)==max_epochs
    result={'state':'preflight_only_complete' if args.smoke else 'full_source_training_complete','parameters':956852,'epochs':max_epochs,'counts':counts,
            'not_formal_training':args.smoke,'resume_format':'custom full-source trainer, not MMEngine',
            'job':{'name':RUN.name,'dataset':'RiceSEG','phase':'source_pretraining_for_RiceSEG','source_route_cancelled':False},
            'best_validation':max(history,key=lambda r:r['miou']),'checkpoint':str(RUN/'best.pt'),
            'checkpoint_sha256':sha(RUN/'best.pt'),'plan_sha256':plan_sha,'test_evaluated':False,
            'note':'Source pseudo-label validation is not RiceSEG accuracy','seconds_this_process':time.time()-started}
    save_json(RUN/'training_done.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    raise SystemExit(main())

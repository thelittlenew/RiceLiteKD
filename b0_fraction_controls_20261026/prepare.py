import shutil,time
from table4_common import *

assert not (ROOT/'plan.json').exists(),'Do not overwrite a frozen plan'
PREVIOUS=WORK/'supplement_student_20261023'
FRACTIONS={'RiceSEG':WORK/'riceseg_fraction_20261026','CVRP_full':WORK/'cvrp_fraction_20261026'}
for n in ['runs','logs','preflight','configs','base_configs','subsets','initialization','references','dependency_snapshot']:(ROOT/n).mkdir(exist_ok=True)
oldplan=json.loads((PREVIOUS/'plan.json').read_text());jobs=[];hashes={};reference_rows=[];baseline=json.loads((PREVIOUS/'table4_baseline_results.json').read_text())
for ds,parent in FRACTIONS.items():
 for seed in [42,7]:
  oldjob=next(j for j in oldplan['jobs'] if j.get('model')=='SegFormer-B0' and j['dataset']==ds and j['seed']==seed)
  original=Path(oldjob['original_config']);config=ROOT/'base_configs'/original.name;shutil.copy2(original,config)
  init=Path(oldjob['initialization']);target=ROOT/'initialization'/init.name
  if not target.exists():shutil.copy2(init,target)
  assert sha(init)==sha(target)
  old_protocol=json.loads((PREVIOUS/'runs'/oldjob['name']/'protocol.json').read_text())
  assert old_protocol['initialization_sha256']==sha(target)
  oldcfg=Config.fromfile(str(original));assert Path(oldcfg.load_from)==init
  for pct in [20,30,40]:
   name=f'{ds}_distilled{pct}_seed{seed}'
   src=parent/('prior20' if pct==20 else 'runs')/name/'result.json';result=json.loads(src.read_text());assert result['state']=='complete'
   rp=ROOT/'references'/('RiceLiteKD_'+name+'.json');shutil.copy2(src,rp)
   hashes[str(src)]=sha(src)
   reference_rows.append(dict(dataset=ds,model='RiceLiteKD',percent=pct,seed=seed,train_count=result['train_count'],**{k:result['test'][k]*100 for k in ['miou','panicle_iou','mean_dice']},origin=str(rp),origin_sha256=sha(rp)))
  src=PREVIOUS/'runs'/oldjob['name']/'result.json';result=json.loads(src.read_text())
  rp=ROOT/'references'/(oldjob['name']+'.json');shutil.copy2(src,rp);hashes[str(src)]=sha(src)
  reference_rows.append(dict(dataset=ds,model='SegFormer-B0',percent=20,seed=seed,train_count=result['train_count'],**{k:result['test'][k]*100 for k in ['miou','panicle_iou','mean_dice']},origin=str(rp),origin_sha256=sha(rp)))
  ref=next(b for b in baseline if b['dataset']==ds and b['model']=='SegFormer-B0' and b['seed']==seed)
  reference_rows.append(dict(dataset=ds,model='SegFormer-B0',percent=100,seed=seed,train_count=318 if ds=='RiceSEG' else 1644,**{k:ref['test'][k]*100 for k in ['miou','panicle_iou','mean_dice']},origin=ref['result_origin'],precision='rounded upstream to 0.01 percentage point'))
  for pct in [30,40]:
   name=f'{ds}_b0_{pct}pct_seed{seed}';matching=f'{ds}_distilled{pct}_seed{seed}'
   for suffix in ['.txt','.json']:
    src=parent/'subsets'/(matching+suffix);dest=ROOT/'subsets'/(name+suffix)
    shutil.copy2(src,dest);assert sha(src)==sha(dest);hashes[str(src)]=sha(src)
   chosen=json.loads((ROOT/'subsets'/(name+'.json')).read_text());n=len(chosen)
   assert n==round((318 if ds=='RiceSEG' else 1644)*pct/100)
   jobs.append(dict(name=name,kind='table4',model='SegFormer-B0',dataset=ds,seed=seed,percent=pct,arm=f'target_{pct}pct',train_count=n,parameters=3715682,original_config=str(config),initialization=str(target),matched_ricelite_run=matching,matched_subset_sha256=sha(ROOT/'subsets'/(name+'.txt'))))
 for split in ['train','val','test']:
  p=OLD/'manifests'/f'{ds}_{split}.json';hashes[str(p)]=sha(p)
 manifests={split:json.loads((OLD/'manifests'/f'{ds}_{split}.json').read_text()) for split in ['train','val','test']}
 sets={s:{r['image'] for r in rows} for s,rows in manifests.items()}
 assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
 assert all(Path(r[k]).is_file() for rows in manifests.values() for r in rows for k in ['image','mask'])
jobs.sort(key=lambda j:(['RiceSEG','CVRP_full'].index(j['dataset']),j['percent'],[42,7].index(j['seed'])))
write_json(ROOT/'reference_results.json',dict(units='percent',rows=reference_rows))
dependencies=[OLD/'comparison_common.py']+list(LITE.glob('*.py'))+list(COMPARISON.glob('*.py'))
for p in dependencies:
 target=ROOT/'dependency_snapshot'/p.parent.name/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);hashes[str(p)]=sha(p)
checks=[]
for job in jobs:
 cfg=table4_config(job);counts=assert_data(cfg,job)
 assert counts['train']==job['train_count']
 assert not cfg.train_dataloader.get('drop_last',False)
 assert counts['train']%8!=1
 cfg.dump(str(ROOT/'configs'/(job['name']+'.py')))
 checks.append(dict(name=job['name'],counts=counts,subset_sha256=job['matched_subset_sha256']))
for p in ROOT.rglob('*'):
 if p.is_file() and p.suffix in ['.py','.json','.txt','.pth'] and '__pycache__' not in p.parts:hashes[str(p)]=sha(p)
plan=dict(created=time.time(),seeds=[42,7],datasets=['RiceSEG','CVRP_full'],percentages=[30,40],jobs=jobs,hashes=hashes,protocol=dict(model='SegFormer-B0',parameters=3715682,epochs=120,batch_size=8,input_size=[512,512],initialization='Same source-trained B0 backbone as original B0 full-data and 20% experiments; fresh target decoder; no target full-data checkpoint',subsets='Exact existing RiceLiteKD 30/40 subsets, same seed, no resampling',selection='Validation mIoU only, final test after selecting checkpoint',budget='Fixed epochs; optimizer steps match RiceLiteKD at same dataset/fraction',interpretation='Model/pipeline comparison under matched target-label budgets; source pretraining differs, so not a pure architecture or distillation ablation'))
write_json(ROOT/'plan.json',plan)
write_json(ROOT/'preflight/data_verified.json',dict(passed=True,plan_sha256=sha(ROOT/'plan.json'),checks=checks))
print(json.dumps(dict(jobs=len(jobs),plan_sha256=sha(ROOT/'plan.json'),checks=checks)))

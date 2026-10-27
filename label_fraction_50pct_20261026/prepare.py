import random,shutil,time
from table4_common import *
assert not (ROOT/'plan.json').exists()
PREVIOUS=WORK/'supplement_student_20261023'
THIRD=WORK/'label_fraction_seed3407_20261026'
PARENTS={'RiceSEG':WORK/'riceseg_fraction_20261026','CVRP_full':WORK/'cvrp_fraction_20261026'}
for d in ['runs','logs','preflight','base_configs','configs','subsets','initialization','dependency_snapshot','references']:(ROOT/d).mkdir(exist_ok=True)
assert json.loads((THIRD/'status.json').read_text())['state']=='complete'
oldplan=json.loads((PREVIOUS/'plan.json').read_text())
jobs=[];hashes={};checks=[]
for ds,parent in PARENTS.items():
 manifests={s:json.loads((OLD/'manifests'/f'{ds}_{s}.json').read_text()) for s in ['train','val','test']}
 sets={s:{r['image'] for r in rows} for s,rows in manifests.items()}
 assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
 for split,rr in manifests.items():
  p=OLD/'manifests'/f'{ds}_{split}.json';hashes[str(p)]=sha(p)
  assert all(Path(r[k]).is_file() for r in rr for k in ['image','mask'])
 rows=sorted(manifests['train'],key=lambda r:r['image'])
 assert len(rows)=={'RiceSEG':318,'CVRP_full':1644}[ds]
 for seed in [42,7,3407]:
  rng=random.Random(seed);initial=rng.sample(range(len(rows)),round(len(rows)*.2))
  remaining=[i for i in range(len(rows)) if i not in set(initial)];rng.shuffle(remaining);order=initial+remaining
  chosen=[rows[i] for i in sorted(order[:round(len(rows)*.5)])]
  prior=THIRD/'subsets'/f'{ds}_ricelite_40pct_seed{seed}.json' if seed==3407 else parent/'subsets'/f'{ds}_distilled40_seed{seed}.json'
  previous=json.loads(prior.read_text())
  assert [rows[i] for i in sorted(order[:round(len(rows)*.4)])]==previous
  assert {r['image'] for r in previous}<={r['image'] for r in chosen}
  prior_copy=ROOT/'references'/prior.name;shutil.copy2(prior,prior_copy)
  write_json(ROOT/'subsets'/f'{ds}_ordering_seed{seed}.json',dict(seed=seed,indices=order,images=[rows[i]['image'] for i in order],previous_40pct=str(prior),previous_40pct_sha256=sha(prior),rule='Same nested sampling as 20/30/40%; sorted 50% prefix'))
  for model in ['RiceLiteKD','SegFormer-B0']:
   if model=='RiceLiteKD':
    original=OLD/'configs'/f'{ds}_stage3_seed{seed}.py';init=OLD/'source_full_initialization.pth';kind='downstream';tag='ricelite'
   else:
    oldjob=next(j for j in oldplan['jobs'] if j.get('model')==model and j['dataset']==ds and j['seed']==42)
    original=Path(oldjob['original_config'].replace('seed42',f'seed{seed}'));init=Path(oldjob['initialization']);kind='table4';tag='b0'
   config=ROOT/'base_configs'/original.name;shutil.copy2(original,config)
   init_copy=ROOT/'initialization'/init.name
   if not init_copy.exists():shutil.copy2(init,init_copy)
   assert sha(init)==sha(init_copy)
   if model=='RiceLiteKD':assert sha(init_copy)=='50a4868b7bb59c5b646260f6e984d70beca4ea21267d1408ea6f6189772d2989'
   cfg=Config.fromfile(str(original));assert cfg.randomness.seed==seed and Path(cfg.load_from)==init
   name=f'{ds}_{tag}_50pct_seed{seed}'
   write_json(ROOT/'subsets'/(name+'.json'),chosen)
   (ROOT/'subsets'/(name+'.txt')).write_text('\n'.join(Path(r['image']).stem for r in chosen)+'\n')
   job=dict(name=name,dataset=ds,model=model,seed=seed,percent=50,kind=kind,arm=f'{tag}_50pct',train_count=len(chosen),full_train=len(rows),original_config=str(config),initialization=str(init_copy))
   if model=='SegFormer-B0':job['parameters']=3715682
   jobs.append(job)
  assert sha(ROOT/'subsets'/f'{ds}_ricelite_50pct_seed{seed}.txt')==sha(ROOT/'subsets'/f'{ds}_b0_50pct_seed{seed}.txt')
for j in jobs:
 cfg=get_config(j) if j['kind']=='downstream' else table4_config(j)
 counts=assert_data(cfg,j);assert counts['train']==j['train_count']
 assert not cfg.train_dataloader.get('drop_last',False)
 cfg.dump(str(ROOT/'configs'/(j['name']+'.py')))
 checks.append(dict(name=j['name'],counts=counts,subset_sha256=sha(ROOT/'subsets'/(j['name']+'.txt'))))
shutil.copy2(THIRD/'summary.json',ROOT/'references'/'prior_36_summary.json')
assert len(json.loads((ROOT/'references/prior_36_summary.json').read_text())['rows'])==36
shutil.copy2(THIRD/'b0_full_three_seed_reference.json',ROOT/'b0_full_three_seed_reference.json')
for p in [OLD/'comparison_common.py']+list(LITE.glob('*.py'))+list(COMPARISON.glob('*.py')):
 q=ROOT/'dependency_snapshot'/p.parent.name/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);hashes[str(p)]=sha(p)
for p in ROOT.rglob('*'):
 if p.is_file() and p.suffix in ['.py','.json','.txt','.pth'] and '__pycache__' not in p.parts:hashes[str(p)]=sha(p)
plan=dict(created=time.time(),seeds=[42,7,3407],datasets=['RiceSEG','CVRP_full'],percentages=[50],jobs=jobs,hashes=hashes,protocol=dict(epochs=120,batch_size=8,input_size=[512,512],subsets='Same subsets for both models; nested within-seed expansion from verified 40% lists',initialization='Independent model-specific source initialization; never target-finetuned checkpoints',selection='Validation mIoU only; one final test per run',seed_scope='Target subsampling and fine-tuning; fixed source pretraining',aggregation='Mean and sample standard deviation across 42,7,3407',history='User requested exploratory 50% extension after examining 20/30/40%; old cancelled partial 50% run is preserved separately; these are fresh independent runs'))
write_json(ROOT/'plan.json',plan)
write_json(ROOT/'preflight/data_verified.json',dict(passed=True,plan_sha256=sha(ROOT/'plan.json'),checks=checks))
print(json.dumps(dict(jobs=len(jobs),plan_sha256=sha(ROOT/'plan.json'),checks=checks)),flush=True)

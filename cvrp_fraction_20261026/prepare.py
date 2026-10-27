import random,shutil,time
from experiment_common import *

assert not (ROOT/'plan.json').exists(),'Frozen plan already exists'
PREVIOUS=WORK/'supplement_student_20261023'
for d in ['base_configs','configs','subsets','runs','logs','preflight','prior20','dependency_snapshot']:
 (ROOT/d).mkdir(exist_ok=True)
seeds=[42,7];fractions=[30,40];jobs=[];hashes={}
manifests={s:json.loads((OLD/'manifests'/f'CVRP_full_{s}.json').read_text()) for s in ['train','val','test']}
sets={s:{r['image'] for r in rows} for s,rows in manifests.items()}
assert len(manifests['train'])==1644
assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
for split,rows in manifests.items():
 assert all(Path(r[k]).is_file() for r in rows for k in ['image','mask'])
 hashes[str(OLD/'manifests'/f'CVRP_full_{split}.json')]=sha(OLD/'manifests'/f'CVRP_full_{split}.json')
shutil.copy2(OLD/'source_full_initialization.pth',ROOT/'source_full_initialization.pth')
assert sha(ROOT/'source_full_initialization.pth')=='50a4868b7bb59c5b646260f6e984d70beca4ea21267d1408ea6f6189772d2989'
rows=sorted(manifests['train'],key=lambda r:r['image'])
for seed in seeds:
 shutil.copy2(OLD/'configs'/f'CVRP_full_stage3_seed{seed}.py',ROOT/'base_configs'/f'CVRP_full_stage3_seed{seed}.py')
 rng=random.Random(seed)
 initial=rng.sample(range(len(rows)),round(len(rows)*.2))
 previous=json.loads((PREVIOUS/'subsets'/f'CVRP_full_seed{seed}.json').read_text())
 assert [rows[i] for i in sorted(initial)]==previous
 remaining=[i for i in range(len(rows)) if i not in set(initial)];rng.shuffle(remaining)
 ordering=initial+remaining
 write_json(ROOT/'subsets'/f'ordering_seed{seed}.json',dict(seed=seed,indices=ordering,images=[rows[i]['image'] for i in ordering],rule='Original random.sample 20% retained; remainder shuffled by same RNG state; sorted prefixes written to training lists'))
 prev_set={r['image'] for r in previous}
 for pct in fractions:
  n=round(len(rows)*pct/100);chosen=[rows[i] for i in sorted(ordering[:n])]
  assert len({r['image'] for r in chosen})==n and prev_set<={r['image'] for r in chosen}
  prev_set={r['image'] for r in chosen}
  name=f'CVRP_full_distilled{pct}_seed{seed}'
  write_json(ROOT/'subsets'/f'{name}.json',chosen)
  (ROOT/'subsets'/f'{name}.txt').write_text('\n'.join(Path(r['image']).stem for r in chosen)+'\n')
  jobs.append(dict(name=name,dataset='CVRP_full',model='RiceLiteKD',seed=seed,percent=pct,arm=f'distilled_{pct}pct',train_count=n,full_train=1644,actual_fraction=n/1644,kind='downstream'))
 prior=PREVIOUS/'runs'/f'CVRP_full_distilled20_seed{seed}'
 out=ROOT/'prior20'/prior.name;out.mkdir()
 for fn in ['result.json','history.json','protocol.json','selection.json','best_model.pth']:
  shutil.copy2(prior/fn,out/fn)
  assert sha(prior/fn)==sha(out/fn)
 protocol=json.loads((out/'protocol.json').read_text())
 assert protocol['init_sha256']==sha(ROOT/'source_full_initialization.pth')
 write_json(out/'reuse_provenance.json',dict(origin=str(prior),retrained=False,files={p.name:sha(p) for p in out.iterdir()}))
jobs.sort(key=lambda j:(j['percent'],seeds.index(j['seed'])))
dependencies=[OLD/'comparison_common.py']+list(LITE.glob('*.py'))
for p in dependencies:
 target=ROOT/'dependency_snapshot'/p.parent.name/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 hashes[str(p)]=sha(p)
for j in jobs:
 cfg=get_config(j);cfg.dump(str(ROOT/'configs'/f"{j['name']}.py"))
 for split in ['train','val','test']:
  ds=Runner.build_dataloader(cfg[split+'_dataloader'],seed=j['seed']).dataset
  actual={(str(Path(ds.get_data_info(i)['img_path']).resolve()),str(Path(ds.get_data_info(i)['seg_map_path']).resolve())) for i in range(len(ds))}
  expected=json.loads((ROOT/'subsets'/f"{j['name']}.json").read_text()) if split=='train' else manifests[split]
  assert actual=={(r['image'],r['mask']) for r in expected},(j['name'],split)
for p in ROOT.rglob('*'):
 if p.is_file() and (p.suffix in ['.py','.json','.txt','.pth']) and '__pycache__' not in p.parts:
  hashes[str(p)]=sha(p)
plan=dict(created=time.time(),seeds=seeds,percentages=fractions,jobs=jobs,parameters=956852,hashes=hashes,
 protocol=dict(epochs=120,batch_size=8,input_size=[512,512],initialization='Same source-distilled checkpoint for every independent run; no target fine-tuned initialization',sampling='Nested subsets within seed retaining original 20%; no replacement, no discarded images',selection='Validation mIoU only; one final test evaluation per job',test_policy='Fixed CVRP 30/40 grid; chosen after exploratory RiceSEG results; no checkpoint selection from test results',budget='Fixed epochs, not matched optimizer steps',comparison='Same-seed B0 full-data reference; mean and sample SD; no equivalence or significance claim from two seeds'))
write_json(ROOT/'plan.json',plan)
write_json(ROOT/'preflight/data_verified.json',dict(passed=True,jobs=len(jobs),counts={str(p):round(1644*p/100) for p in fractions},val=len(manifests['val']),test=len(manifests['test']),plan_sha256=sha(ROOT/'plan.json')))
print(json.dumps(dict(jobs=len(jobs),plan_sha256=sha(ROOT/'plan.json'),data_verified=True)))

import random,json,shutil,time
from pathlib import Path
from experiment_common import *

assert not (ROOT/'plan.json').exists(),'Never overwrite a frozen experiment plan'
for n in ['runs','subsets','logs','preflight']: (ROOT/n).mkdir(exist_ok=True)
seeds=[42,7];datasets=['RiceSEG','CVRP_full'];jobs=[];hashes={}
baseline=json.loads((ROOT/'baseline_results.json').read_text())
baseline=[r for r in baseline if r['seed'] in seeds]
write_json(ROOT/'baseline_results.json',baseline)
subset_info=[]
for ds in datasets:
 manifests={s:json.loads((OLD/'manifests'/f'{ds}_{s}.json').read_text()) for s in ['train','val','test']}
 for s,rows in manifests.items():
  assert all(Path(r['image']).is_file() and Path(r['mask']).is_file() for r in rows)
  hashes[str(OLD/'manifests'/f'{ds}_{s}.json')]=sha(OLD/'manifests'/f'{ds}_{s}.json')
 sets={s:{r['image'] for r in rows} for s,rows in manifests.items()}
 assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
 for seed in seeds:
  rows=sorted(manifests['train'],key=lambda r:r['image'])
  n=round(len(rows)*.2);indices=sorted(random.Random(seed).sample(range(len(rows)),n))
  chosen=[rows[i] for i in indices]
  write_json(ROOT/'subsets'/f'{ds}_seed{seed}.json',chosen)
  p=ROOT/'subsets'/f'{ds}_seed{seed}.txt';p.write_text('\n'.join(Path(r['image']).stem for r in chosen)+'\n')
  hashes[str(p)]=sha(p)
  item=dict(dataset=ds,seed=seed,full_train=len(rows),selected=n,actual_fraction=n/len(rows),val=len(manifests['val']),test=len(manifests['test']))
  subset_info.append(item)
  jobs.append(dict(name=f'{ds}_distilled20_seed{seed}',kind='downstream',dataset=ds,seed=seed,arm='distilled_20pct',train_count=n))
  hashes[str(OLD/'configs'/f'{ds}_stage3_seed{seed}.py')]=sha(OLD/'configs'/f'{ds}_stage3_seed{seed}.py')
jobs.append(dict(name='SourceFull_direct_120ep_seed42',kind='source',seed=42))
for ds in datasets:
 for seed in seeds:jobs.append(dict(name=f'{ds}_source_direct_seed{seed}',kind='downstream',dataset=ds,seed=seed,arm='source_direct',train_count=len(json.loads((OLD/'manifests'/f'{ds}_train.json').read_text()))))
ready=json.loads((DATA/'DATA_READY.json').read_text());assert ready['counts']==dict(train=98422,val=5390)
assert ready['prepared_sha256']==sha(DATA/'PREPARED.json')
tr=json.loads((DATA/'train.json').read_text());va=json.loads((DATA/'val.json').read_text())
assert not {r['group'] for r in tr}&{r['group'] for r in va}
assert not {r['rgb_sha256'] for r in tr}&{r['rgb_sha256'] for r in va}
assert all(Path(p).is_file() for p in {r['archive'] for r in tr+va})
dependencies=[OLD/'source_full_initialization.pth',OLD/'comparison_common.py',LITE/'full_distill_components.py',LITE/'matched_cgnet_refine7_recipe.py',LITE/'matched_cgnet_refine_recipe.py',LITE/'matched_cgnet_fusion_recipe.py',LITE/'matched_cgnet_recipe.py',LITE/'matched_recipe.py',LITE/'rice_lite.py',DATA/'train.json',DATA/'val.json',DATA/'inventory.json',DATA/'DATA_READY.json',DATA/'PREPARED.json',ROOT/'baseline_results.json']
for p in dependencies:hashes[str(p)]=sha(p)
assert hashes[str(OLD/'source_full_initialization.pth')]=='50a4868b7bb59c5b646260f6e984d70beca4ea21267d1408ea6f6189772d2989'
for p in ROOT.glob('*.py'):
 if p.name not in ['prepare_experiment.py']:hashes[str(p)]=sha(p)
plan=dict(created=time.time(),seeds=seeds,student='CGNet + Refine7',parameters=956852,jobs=jobs,subsets=subset_info,
 source=dict(seed=42,epochs=120,batch_size=8,counts=ready['counts'],loss='class-weighted CE + 2 * two-class Dice',teacher_loaded=False,initialization='all random; no checkpoint or pretrained backbone',reference='identical data, split, architecture, augmentation, optimizer and schedule to full source distillation; remove KL and attention-transfer terms only'),
 downstream=dict(epochs=120,batch_size=8,selection='validation mIoU only; test once after selection',comparison='same archived full-data recipe; 20% of training images only; all validation/test images unchanged',budget_note='Fixed epochs, so 20% arms have fewer optimizer updates. No repeated sampling to match full-data updates.'),
 benchmark_policy='No new FPS claim; segmentation accuracy ablations only',hashes=hashes)
write_json(ROOT/'plan.json',plan)
for j in jobs:
 if j['kind']=='downstream':
  cfg=get_config(j);(ROOT/'configs').mkdir(exist_ok=True);cfg.dump(str(ROOT/'configs'/f"{j['name']}.py"))
print(json.dumps({'jobs':len(jobs),'seeds':seeds,'subsets':subset_info},indent=2))

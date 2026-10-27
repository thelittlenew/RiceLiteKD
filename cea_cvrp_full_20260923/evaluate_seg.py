from pathlib import Path
import sys,json,hashlib,argparse
R=Path('/root/autodl-tmp/work');sys.path.insert(0,str(R/'baseline_src/mmsegmentation'))
from mmengine.config import Config
from mmengine.runner import Runner
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('checkpoint');p.add_argument('output');p.add_argument('--save-predictions',action='store_true');a=p.parse_args()
O=Path(a.output);O.mkdir(parents=True,exist_ok=True)
cfg=Config.fromfile(a.config);cfg.work_dir=str(O);cfg.load_from=a.checkpoint;cfg.resume=False
if a.save_predictions:cfg.test_evaluator.output_dir=str(O/'predictions')
runner=Runner.from_cfg(cfg);result={k:float(v) for k,v in runner.test().items()}
(O/'metrics.json').write_text(json.dumps(dict(metrics=result,config=a.config,checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()),indent=2))
print('EVALUATION_COMPLETE',json.dumps(result),flush=True)

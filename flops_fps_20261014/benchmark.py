from pathlib import Path
import sys,json,time,hashlib,statistics,gc,traceback,subprocess,math
ROOT=Path(__file__).resolve().parent
W=ROOT.parent
sys.path[:0]=[str(W/'segmentation_comparison_audit_20261011/site'),str(W/'baseline_src/mmsegmentation'),str(W/'segmentation_comparisons_v2_20261011'),str(W/'stage3_comparison_20261006'),str(W/'stage3_lite_20260926')]
import torch
from torch import nn
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.analysis import FlopAnalyzer
from mmengine.analysis.jit_handles import get_shape
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
register_all_modules()
torch.set_num_threads(4);torch.set_num_interop_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.manual_seed(42)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def gpu():return subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
class Forward(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return self.model.encode_decode(x,[dict(img_shape=(512,512),ori_shape=(512,512),pad_shape=(512,512))])
def attention_count(inputs,outputs):
    q,k,v=[get_shape(x) for x in inputs[:3]]
    return math.prod(q[:-2])*q[-2]*k[-2]*(q[-1]+v[-1])

protocol=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,batch_size=1,input_shape=[1,3,512,512],precision='FP32',tf32=False,cudnn_benchmark=False,warmup=100,repeats=5,forwards_per_repeat=100,aggregation='FPS = 1000 / median of five repeat mean latencies (ms)',scope='GPU-resident input through complete evaluation-mode model encode_decode, including 512x512 logit interpolation; no disk I/O, preprocessing, host/device transfer or argmax',flops='MMEngine FlopAnalyzer, one multiply-add counted as one operation; per-image GFLOPs; fused attention matrix products counted explicitly',checkpoint_dataset='RiceSEG',no_training=True)
write(ROOT/'protocol.json',protocol)
jobs=json.loads((ROOT/'jobs.json').read_text())
if '--probe' in sys.argv:
    print(json.dumps([dict(id=j['id'],config=Path(j['config']).is_file(),checkpoint=Path(j['checkpoint']).is_file()) for j in jobs]));sys.exit(0)
selected=sys.argv[sys.argv.index('--only')+1] if '--only' in sys.argv else None
for index,job in enumerate(jobs):
    if selected and job['id']!=selected:continue
    out=ROOT/(job['id']+'.json')
    if out.exists() and '--force' not in sys.argv:continue
    write(ROOT/'status.json',dict(state='running',job=job['id'],index=index+1,total=len(jobs)))
    print('START',job['id'],flush=True)
    try:
        digest=sha(job['checkpoint'])
        if job.get('checkpoint_sha256'):assert digest==job['checkpoint_sha256']
        cfg=Config.fromfile(job['config'])
        model=revert_sync_batchnorm(MODELS.build(cfg.model))
        state=torch.load(job['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(state.get('state_dict',state),strict=True);del state
        params=sum(p.numel() for p in model.parameters());assert params==job['parameters'],(params,job['parameters'])
        model=model.float().cuda().eval().requires_grad_(False)
        forward=Forward(model)
        torch.manual_seed(42);x=torch.randn(1,3,512,512,device='cuda')
        before=gpu()
        with torch.inference_mode():
            y=forward(x);assert y.shape==(1,2,512,512) and torch.isfinite(y).all();del y
            for _ in range(100):forward(x)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            samples=[]
            for _ in range(5):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(100):forward(x)
                b.record();torch.cuda.synchronize();samples.append(a.elapsed_time(b)/100)
            peak=torch.cuda.max_memory_allocated()/2**20
            analysis=FlopAnalyzer(forward,(x,)).set_op_handle('aten::scaled_dot_product_attention',attention_count,'aten::_scaled_dot_product_flash_attention',attention_count,'aten::_scaled_dot_product_efficient_attention',attention_count)
            analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings('none')
            flops=analysis.total()
        median=statistics.median(samples)
        result={**job,'checkpoint_sha256':digest,'config_sha256':sha(job['config']),'parameters':params,'gflops':flops/1e9,'flops_by_operator':dict(analysis.by_operator()),'unsupported_ops':dict(analysis.unsupported_ops()),'repeat_mean_ms':samples,'median_ms':median,'fps':1000/median,'peak_allocated_mib':peak,'gpu_start':before,'gpu_end':gpu(),'backbone':str(cfg.model.get('backbone',{}).get('type')),'decode_head':str(cfg.model.get('decode_head',{}).get('type'))}
        write(out,result)
        print('DONE',job['id'],round(result['gflops'],3),round(result['fps'],2),'unsupported',result['unsupported_ops'],flush=True)
        del analysis,forward,model,x;gc.collect();torch.cuda.empty_cache()
    except Exception:
        error=traceback.format_exc();print(error,flush=True);write(ROOT/(job['id']+'_error.json'),dict(error=error));sys.exit(1)
write(ROOT/'status.json',dict(state='complete',total=len(jobs)))

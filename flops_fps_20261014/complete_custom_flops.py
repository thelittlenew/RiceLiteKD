from pathlib import Path
import json,math
from collections import Counter

ROOT=Path(__file__).resolve().parent
namespace={'__file__':str(ROOT/'benchmark.py')}
exec(compile((ROOT/'benchmark.py').read_text().split('protocol=dict')[0],'benchmark_setup','exec'),namespace)
globals().update({k:v for k,v in namespace.items() if not k.startswith('__')})

calls=[]
def conv_mode(inputs,outputs):
    w=get_shape(inputs[1]);out=get_shape(outputs[0])
    count=math.prod(out)*math.prod(w[1:])
    calls.append(dict(op='conv_mode',weight=w,output=out,flops=count))
    return Counter(conv=count)

def fft(inputs,outputs,inverse=False):
    shape=get_shape(inputs[0]);h,w=shape[-2:];n=h*w
    assert h&(h-1)==0 and w&(w-1)==0
    # Arithmetic estimate for a radix-2 complex FFT at the one-MAC convention:
    # complex twiddle product = two FMA + two multiplies; two complex butterfly
    # sums = four adds. Eight counted operations per butterfly -> 4*N*log2(N).
    # The inverse default normalization divides both real and imaginary parts.
    count=math.prod(shape)*(4*math.log2(n)+(2 if inverse else 0))
    calls.append(dict(op='ifft2' if inverse else 'fft2',shape=shape,flops=count))
    return Counter(fft=count)

def scan(inputs,outputs):
    b,d,l=get_shape(inputs[0]);a=get_shape(inputs[2]);n=a[1]
    assert a[0]==d
    # Mamba selective_scan_ref with real A and variable B/C:
    # delta*A: BDLN multiplies; delta*B*u: 2 BDLN multiplies;
    # recurrent state update: BDLN MACs; output C dot state: BDLN MACs.
    # Skip D*u+y: BDL MACs. Nonlinear exp/softplus and data movement are
    # excluded, consistently with unsupported elementwise activations below.
    count=5*b*d*l*n
    has_d=inputs[5].type().kind()!='NoneType'
    has_z=inputs[6].type().kind()!='NoneType'
    if has_d:count+=b*d*l
    if has_z:count+=b*d*l
    calls.append(dict(op='selective_scan',shape=[b,d,l],state=n,has_d=has_d,has_z=has_z,flops=count))
    return Counter(selective_scan=count)

job=next(j for j in json.loads((ROOT/'jobs.json').read_text()) if j['id']=='FEM_Lite')
cfg=Config.fromfile(job['config'])
model=revert_sync_batchnorm(MODELS.build(cfg.model))
state=torch.load(job['checkpoint'],map_location='cpu',weights_only=False)
model.load_state_dict(state['state_dict'],strict=True)
model=model.float().cuda().eval().requires_grad_(False)
torch.manual_seed(42);x=torch.randn(1,3,512,512,device='cuda')
with torch.inference_mode():
    analysis=FlopAnalyzer(Forward(model),(x,)).set_op_handle(
        'aten::_convolution_mode',conv_mode,
        'aten::fft_fft2',lambda i,o:fft(i,o,False),
        'aten::fft_ifft2',lambda i,o:fft(i,o,True),
        'prim::PythonOp.SelectiveScanFn',scan,
        'aten::scaled_dot_product_attention',attention_count)
    analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings('none')
    total=analysis.total()
result=json.loads((ROOT/'FEM_Lite.json').read_text())
result['base_tool_gflops_before_custom_ops']=result['gflops']
result['gflops']=total/1e9
result['flops_by_operator']=dict(analysis.by_operator())
result['unsupported_ops']=dict(analysis.unsupported_ops())
result['custom_operator_calls']=calls
assert len([c for c in calls if c['op']=='conv_mode'])==46
assert len([c for c in calls if c['op']=='selective_scan'])==20
assert len([c for c in calls if c['op']=='fft2'])==20
assert len([c for c in calls if c['op']=='ifft2'])==40
assert not any(k in result['unsupported_ops'] for k in ['aten::_convolution_mode','aten::fft_fft2','aten::fft_ifft2','prim::PythonOp.SelectiveScanFn'])
write(ROOT/'FEM_Lite.json',result)
protocol=json.loads((ROOT/'protocol.json').read_text())
protocol['custom_flops_note']='FEM-Lite: same-padding convolution counted as output elements times per-filter kernel size; real selective scan core 5*B*D*L*N plus B*D*L for the skip; radix-2 FFT estimate 4*N*log2(N) per complex transform plus 2*N for inverse normalization. One MAC = one counted operation. Elementwise activations and data movement not supported by MMEngine are excluded for all models.'
protocol['references']=['https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.analysis.FlopAnalyzer.html','https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py']
write(ROOT/'protocol.json',protocol)
jobs=json.loads((ROOT/'jobs.json').read_text());results=[json.loads((ROOT/(j['id']+'.json')).read_text()) for j in jobs]
write(ROOT/'results.json',results)
print(json.dumps(dict(label=result['label'],gflops=result['gflops'],flops_by_operator=result['flops_by_operator'],unsupported_ops=result['unsupported_ops']),indent=2))

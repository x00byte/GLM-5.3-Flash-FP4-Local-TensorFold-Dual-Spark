#!/usr/bin/env python3
"""Run one rank. Own-container lifecycle only; checkpoint mounts stay read-only."""
import argparse,json,os,shlex,socket,subprocess,time,urllib.request
from pathlib import Path
NAME=os.environ.get('CONTAINER_NAME','glm53-fp4-local-tensorfold')
IMAGE=os.environ.get('IMAGE','local/glm53-fp4-local-tensorfold:20261017-v3')
MODEL='nvidia/GLM-5.3-Flash-FP4-Local'
def need(key):
 value=os.environ.get(key,'').strip()
 if not value:raise SystemExit(f'Set {key}; see node.env.example')
 return value
def run(args,**kw):return subprocess.run(args,check=True,text=True,**kw)
def output(args):return run(args,capture_output=True).stdout.strip()
def environment():
 iface=need('FABRIC_IFACE');hcas=need('ROCE_HCAS')
 settings={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TENSORFOLD_NO_UPDATE_CHECK':'1','TF_GLM_COMM':'roce','TF_ROCE_HCA':hcas,'TF_ROCE_MAX_KB':'2048','TF_ROCE_WAIT_S':'300','TF_GLM_HC_SPLIT':'0','TF_GLM_PREFILL_OVERLAP':'0','TF_GLM_MULTI_WINDOW':'128','TF_GLM_MULTI_LONE':'0','TF_GLM_MULTI_SAMPLER':'packed','TF_GLM_MULTI_ASYNC':'1','TF_GLM_MULTI_PREFILL':'1','TF_GLM_SHARED_PREFIX':'0','TF_GLM_STREAM_SMOOTH':'0','TF_GLM_KEEP_REASONING':'1','TF_GLM_CLEAR_THINKING':'0','TF_GLM_EFFORT_TAIL':'0','TF_GLM_DFLASH_POLICY':'fnc7:0.3','TF_GLM_MTP':'0','TF_GLM_COPY_DRAFTS':'1','TF_GLM_COPY_SMART':'1','TF_GLM_COPY_MATCH':'8','TF_GLM_COPY_MAX':'15','TF_GLM_COPY_REPLY_MATCH':'32','TF_GLM_COPY_MISS_MAX':'0','TF_GLM_DRAFT_QUANT':'minmax','TF_GLM_KV':'fp8','TF_GLM_L2PF':'0','TF_GLM_PREFILL_ROWS':'2048','TF_GLM_CACHE_GIB':'8','TENSORFOLD_MEMORY_RESERVE_GIB':'14.5','TF_GLM_MAX_QUEUED':'16','TF_GLM_MULTI_WATCHDOG_S':'300','TF_GLM_MULTI_WATCHDOG_EXIT':'1','TF_GLM_EXIT_ON_BROKEN':'1','NCCL_IB_HCA':hcas,'NCCL_SOCKET_IFNAME':iface,'NCCL_IB_GID_INDEX':os.environ.get('GID_INDEX','3'),'NCCL_IB_DISABLE':'0','NCCL_IGNORE_CPU_AFFINITY':'1','NCCL_MAX_NCHANNELS':'4','GLOO_SOCKET_IFNAME':iface,'TORCH_EXTENSIONS_DIR':'/cache/extensions','TRITON_CACHE_DIR':'/cache/triton','CUDA_CACHE_PATH':'/cache/cuda','MAX_JOBS':'4','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':'0'}
 for key in ('TENSORFOLD_MEMORY_RESERVE_GIB','TF_GLM_CACHE_GIB','TF_GLM_MULTI_LONE','TF_GLM_MULTI_SAMPLER','TF_GLM_MULTI_ASYNC','TF_GLM_DFLASH_POLICY',
             'TF_GLM_DRAFT_QUANT','TF_GLM_COPY_DRAFTS','TF_GLM_COPY_MATCH','TF_GLM_COPY_MAX','TF_GLM_COPY_REPLY_MATCH','TF_GLM_COPY_MISS_MAX','TF_GLM_COPY_SMART','TF_GLM_MULTI_WINDOW','TF_ROCE_MAX_KB'):
  if key in os.environ:settings[key]=os.environ[key]
 return settings

def docker_args(plain=False):
 rank=int(need('RANK'))
 if rank not in (0,1):raise SystemExit('RANK must be 0 or 1')
 cache=str(Path(need('CACHE_DIR')).expanduser().resolve())
 a=['docker','run','-d','--name',NAME,'--gpus','all','--network','host','--ipc','host','--device','/dev/infiniband','--ulimit','memlock=-1','--ulimit','stack=67108864','--cap-add','IPC_LOCK','--security-opt','seccomp=unconfined','--security-opt','apparmor=unconfined','--log-opt','max-size=100m','--log-opt','max-file=3','-v',need('MODEL_SOURCE')+':/model:ro','-v',cache+':/cache','--entrypoint','tensorfold']
 if not plain:a+=['-v',need('DRAFTER_SOURCE')+':/draft:ro']
 for k,v in sorted(environment().items()):a+=['-e',k+'='+v]
 a+=[IMAGE,'serve','/model','--tp','2','--rank',str(rank),'--master',need('HEAD_ADDR'),'--master-port',os.environ.get('MASTER_PORT','29592'),'--parallel','8','--context','262144','--max-tokens','32768']
 a+=['--no-drafts'] if plain else ['--drafter','/draft']
 if rank==0:a+=['--name',MODEL,'--host','0.0.0.0','--port',os.environ.get('API_PORT','8888')]
 return a
def validate(plain=False):
 code="import runpy; runpy.run_path('/opt/glm-fp4-local/verify_snapshot.py')['verify'](__import__('pathlib').Path('/model'),__import__('pathlib').Path('/opt/glm-fp4-local/target-manifest.json')); import pathlib,json,torch; from tensorfold.families.glm5_next.cuda.fp4_local import validate; p=pathlib.Path('/model'); validate(json.loads((p/'config.json').read_text())); i=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']; assert len(set(i.values()))==42; assert all((p/f).is_file() for f in set(i.values())); print('FP4-Local storage and draft assets readable; torch',torch.__version__)"
 args=['docker','run','--rm','--network','none','--memory','4g','--cpus','2','-e','CUDA_VISIBLE_DEVICES=','-e','PYTHONDONTWRITEBYTECODE=1','-v',need('MODEL_SOURCE')+':/model:ro']
 if not plain:
  args+=['-v',need('DRAFTER_SOURCE')+':/draft:ro']
  code+="; assert pathlib.Path('/draft/config.json').is_file()"
 run(args+['--entrypoint','python',IMAGE,'-c',code])

def start(plain=False):
 args=docker_args(plain)
 existing=output(['docker','ps','-a','--filter','name=^/'+NAME+'$','--format','{{.State}}']).splitlines()
 if 'running' in existing:print(NAME+' is already running.');return
 validate(plain)
 if output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']):raise SystemExit('A GPU workload is running; stop it explicitly before starting this recipe.')
 if int(need('RANK'))==0:
  with socket.socket() as s:
   if s.connect_ex(('127.0.0.1',int(os.environ.get('API_PORT','8888'))))==0:raise SystemExit('API port is occupied')
 if existing:run(['docker','rm',NAME])
 Path(need('CACHE_DIR')).expanduser().mkdir(parents=True,exist_ok=True)
 run(args)
 if int(need('RANK')):print('Worker started. Start rank 0 on the head.');return
 for _ in range(300):
  try:
   with urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('API_PORT','8888')+'/health',timeout=2) as r:h=json.load(r)
   if h.get('backend')=='tensorfold' and h.get('ok'):print('TensorFold is ready.');return
  except OSError:pass
  if output(['docker','inspect','-f','{{.State.Running}}',NAME])!='true':raise SystemExit('Rank exited; inspect docker logs '+NAME)
  time.sleep(3)
 raise SystemExit('Readiness timed out; inspect both rank logs')
def stop():
 if output(['docker','ps','-a','--filter','name=^/'+NAME+'$','--format','{{.Names}}']):
  run(['docker','stop','--time','20',NAME]);run(['docker','rm',NAME])
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('start','stop','validate','args'));p.add_argument('--plain',action='store_true');a=p.parse_args()
 if a.action=='start':start(a.plain)
 elif a.action=='stop':stop()
 elif a.action=='validate':validate(a.plain)
 else:print(shlex.join(docker_args(a.plain)))

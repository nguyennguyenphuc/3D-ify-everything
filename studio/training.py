import json
import math
import os
import re
import subprocess
import shutil
import time
from pathlib import Path
import numpy as np
from plyfile import PlyData
from .config import OPENSPLAT
from .environment import inspect


def validate_splat(path):
    data = PlyData.read(str(path))['vertex']
    required = ['x','y','z','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','f_dc_0','f_dc_1','f_dc_2']
    if len(data) == 0 or any(n not in data.data.dtype.names for n in required): raise RuntimeError('PLY chưa phải Gaussian Splat hợp lệ')
    if not all(np.isfinite(data[n]).all() for n in required): raise RuntimeError('PLY có tham số không hữu hạn')
    return len(data)


def train(run,iterations,seconds,emit):
    env_check = inspect()
    for key in ('mps','opensplat','metal'):
        if not env_check[key]['ok']: raise RuntimeError(f'{key}: {env_check[key]["detail"]}')
    start = time.monotonic()
    output = run/'training'; output.mkdir(exist_ok=True)
    metrics = {'backend':'MPS','steps':0,'losses':[],'parameter_updates_verified':False,'budget_exhausted':False}
    def phase(target,resume=None):
        remaining = seconds-(time.monotonic()-start)
        if remaining <= 0: metrics['budget_exhausted']=True; return
        args = [str(OPENSPLAT),str(run),'-n',str(target),'-o',str(output/'working.ply'),'--save-every','50','--max-gaussians','300000','--sh-degree','1']
        if resume: args += ['--resume',str(resume)]
        emit(f'OpenSplat Metal: tối đa {target} bước',stage='training',progress=.65)
        (output/f'command_{target}.json').write_text(json.dumps(args,indent=2))
        using_mps = False
        # LibTorch's install name is patched during setup to share Homebrew's
        # libomp with OpenCV → OpenBLAS. Do not allow duplicate runtimes.
        child_env = dict(os.environ, STUDIO_TRAIN_SECONDS=str(remaining))
        child_env.pop('KMP_DUPLICATE_LIB_OK', None)
        p = subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=child_env)
        try:
            with (output/f'train_{target}.log').open('w') as log:
                for line in p.stdout:
                    log.write(line); log.flush()
                    if 'Discarding ' in line: raise RuntimeError('OpenSplat loại camera không thấy điểm; cảnh chưa đủ liên kết')
                    if 'Using CPU' in line or 'Using CUDA' in line: raise RuntimeError('OpenSplat dùng sai backend')
                    if 'Using MPS' in line: using_mps = True
                    match = re.search(r'Step (\d+):\s*(\S+)',line)
                    if match:
                        step,loss = int(match[1]),float(match[2])
                        if not using_mps or not math.isfinite(loss): raise RuntimeError('Loss/backend không hợp lệ')
                        metrics['steps']=step; metrics['losses'].append([step,loss])
                        (output/'metrics.json').write_text(json.dumps(metrics,indent=2))
                        emit(line.strip(),stage='training',progress=.65+.34*step/iterations,step=step,loss=loss)
                    update = re.search(r'STUDIO_UPDATE (\d+) 1',line)
                    if update: metrics['parameter_updates_verified']=True
                    budget = re.search(r'STUDIO_BUDGET (\d+)',line)
                    if budget: metrics['steps']=int(budget[1]); metrics['budget_exhausted']=True
                if p.wait() != 0: raise RuntimeError('OpenSplat thất bại; xem training log')
                if not using_mps: raise RuntimeError('Không xác nhận được OpenSplat thực thi MPS')
                if not metrics['budget_exhausted']: metrics['steps']=target
                metrics['splat_count']=validate_splat(output/'working.ply')
                shutil.copy2(output/'working.ply',output/'validated.tmp')
                (output/'validated.tmp').replace(output/'splat.ply')
        finally:
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
    try:
        if seconds <= 0:
            metrics['budget_exhausted']=True
            raise RuntimeError('Hết ngân sách trước khi training có kết quả hợp lệ')
        phase(100)
        metrics['splat_count'] = validate_splat(output/'splat.ply')
        if not metrics['parameter_updates_verified']: raise RuntimeError('Chưa xác nhận được cập nhật tham số trong smoke test')
        shutil.copy2(output/'splat.ply',output/'smoke.ply')
        if iterations > 100 and not metrics['budget_exhausted']:
            phase(iterations,output/'splat.ply')
            metrics['splat_count'] = validate_splat(output/'splat.ply')
        return metrics
    finally:
        metrics['seconds']=time.monotonic()-start
        (output/'metrics.json').write_text(json.dumps(metrics,indent=2))

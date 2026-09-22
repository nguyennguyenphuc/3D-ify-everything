"""Measure each level in a fresh process. Never updates the live job database."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from studio.config import DATA
from studio.capacity import fingerprint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--levels',type=int,nargs='+',default=[16,32,48,64],choices=[16,32,48,64])
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('--child',type=int)
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    paths=sorted(p for p in args.source.iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg'))
    if args.child:
        import numpy as np
        from studio.vggt_mps import reconstruct
        if len(paths)<args.child: raise RuntimeError('Not enough distinct source images')
        chosen=[paths[i] for i in np.linspace(0,len(paths)-1,args.child,dtype=int)]
        args.run.mkdir(parents=True,exist_ok=True)
        (args.run/'inputs.json').write_text(json.dumps([str(p) for p in chosen],indent=2))
        stats=reconstruct(chosen,args.run,lambda message,**kw: print(message,flush=True),max_points=10000)
        if stats['camera_count']!=args.child or not stats['finite_depth'] or stats['backend']!='mps':
            raise RuntimeError('Invalid geometry/backend')
        return
    stamp=str(int(time.time()))
    root=DATA/'capacity-benchmarks'/stamp;root.mkdir(parents=True)
    receipt_path=DATA/'capacity.json'
    try: receipt=json.loads(receipt_path.read_text())
    except (OSError,ValueError): receipt={}
    identity=fingerprint()
    if receipt.get('fingerprint')!=identity: receipt={'fingerprint':identity,'levels':{}}
    env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONUNBUFFERED='1')
    for count in args.levels:
        run=root/str(count);run.mkdir()
        started=time.monotonic()
        with (run/'console.log').open('w') as log:
            process=subprocess.Popen([sys.executable,__file__,'--source',str(args.source.resolve()),'--child',str(count),'--run',str(run)],stdout=log,stderr=subprocess.STDOUT,env=env)
            try: code=process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill();process.wait();code=-9
        result={'passed':code==0,'returncode':code,'seconds':time.monotonic()-started,'run':str(run)}
        for name in ('mps-memory.json','geometry/metrics.json'):
            if (run/name).exists(): result[name]=json.loads((run/name).read_text())
        receipt['levels'][str(count)]=result
        temporary=receipt_path.with_suffix('.tmp');temporary.write_text(json.dumps(receipt,indent=2));temporary.replace(receipt_path)
        print(json.dumps({'count':count,**result}),flush=True)


if __name__=='__main__': main()

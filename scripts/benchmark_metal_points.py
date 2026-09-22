"""Validate 100k initialization using a previously measured VGGT scene."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from studio.config import DATA
from studio.capacity import fingerprint
from studio.geometry import preprocess,export
from studio.training import train


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--geometry',type=Path,required=True)
    args=parser.parse_args()
    run=DATA/'capacity-benchmarks'/('metal-'+str(int(time.time())))
    (run/'images').mkdir(parents=True)
    inputs=json.loads((args.geometry/'inputs.json').read_text())
    cams=json.loads((args.geometry/'geometry/cameras.json').read_text())['cameras']
    arrays=[]; masks=[]; metas=[]; names=[]; depths=[]; confidences=[]
    for i,path in enumerate(inputs):
        a,m,im,meta=preprocess(Path(path))
        name=f'{i:02}.png';im.save(run/'images'/name)
        arrays.append(a);masks.append(m);metas.append(meta);names.append(name)
        with np.load(args.geometry/f'geometry/depth_{i:02}.npz') as data:
            depths.append(data['depth']);confidences.append(data['confidence'])
    stats=export(run,arrays,masks,metas,names,np.array(depths),np.array(confidences),
        np.array([c['w2c'] for c in cams]),np.array([c['model_K'] for c in cams]),max_points=100000)
    result={'passed':False,'run':str(run),'geometry':stats}
    try:
        if stats['point_count']!=100000: raise RuntimeError('Scene did not produce 100k points')
        result['training']=train(run,100,240,lambda message,**kw: print(message,flush=True))
        result['passed']=result['training']['steps']>=100 and result['training']['parameter_updates_verified']
    except Exception as e:
        result['error']=str(e)
    finally:
        path=DATA/'capacity.json'
        try: receipt=json.loads(path.read_text())
        except (ValueError,OSError): receipt={}
        if receipt.get('fingerprint')!=fingerprint(): receipt={'fingerprint':fingerprint(),'levels':{}}
        receipt['metal_100k']=result
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(receipt,indent=2));temporary.replace(path)
        (run/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)


if __name__=='__main__': main()

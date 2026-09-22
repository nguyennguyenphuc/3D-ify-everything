"""Tiny synthetic COLMAP scene that proves the built binary selects MPS, not CPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.config import DATA, OPENSPLAT
from studio.geometry import write_colmap


def main():
    root = DATA / 'opensplat-mps-smoke'
    (root / 'images').mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (64, 64), (120, 160, 190)).save(root / 'images' / '00.png')
    Image.new('RGB', (64, 64), (122, 162, 192)).save(root / 'images' / '01.png')
    K=[[80.,0.,32.],[0.,80.,32.],[0.,0.,1.]]
    # Two separated cameras avoid the pose normalization degeneracy of a
    # single-camera scene. They look along +Z in COLMAP/OpenCV coordinates.
    cameras=[
        {'name':'00.png','wh':[64,64], 'K':K, 'w2c':np.column_stack((np.eye(3),[-.1,0.,0.])).tolist()},
        {'name':'01.png','wh':[64,64], 'K':K, 'w2c':np.column_stack((np.eye(3),[ .1,0.,0.])).tolist()},
    ]
    points=np.array([[-.05,-.05,2.], [.05,-.05,2.],[-.05,.05,2.],[.05,.05,2.]],np.float32)
    colors=np.tile(np.array([[120,160,190]],np.uint8),(len(points),1))
    observations=np.array([[i%2,32.+(i%2)*2,32.] for i in range(len(points))])
    write_colmap(root,cameras,points,colors,observations)
    env = dict(os.environ, DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer', STUDIO_TRAIN_SECONDS='120')
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    p = subprocess.run([str(OPENSPLAT),str(root),'-n','10','-o',str(root/'splat.ply'),'--sh-degree','1'],text=True,capture_output=True,env=env,timeout=120)
    (root/'console.log').write_text(p.stdout+p.stderr)
    if p.returncode or 'Using MPS' not in p.stdout or 'Using CPU' in p.stdout:
        raise RuntimeError(p.stdout+p.stderr)
    if not (root/'splat.ply').exists(): raise RuntimeError('Không tạo được splat.ply')
    (root/'result.json').write_text(json.dumps({'backend':'MPS','returncode':p.returncode,'output':p.stdout},indent=2))
    print('OpenSplat MPS smoke passed')


if __name__ == '__main__': main()

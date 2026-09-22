"""Only measured levels are enabled. Receipts are local to this hardware/runtime."""
import hashlib
import importlib.metadata
import json
import platform
import psutil
from .config import DATA, ROOT, OPENSPLAT, AppError


def fingerprint():
    files=[ROOT/'studio/vggt_mps.py',ROOT/'studio/geometry.py',DATA/'models/revision.json',OPENSPLAT.parent/'metal-build.json']
    digest=hashlib.sha256()
    for path in files:
        if path.exists(): digest.update(path.read_bytes())
    return {'machine':platform.node(),'architecture':platform.machine(),
        'memory':psutil.virtual_memory().total,'torch':importlib.metadata.version('torch'),
        'implementation':digest.hexdigest()}


def capabilities():
    path=DATA/'capacity.json'
    try: receipt=json.loads(path.read_text())
    except (OSError,ValueError): receipt={}
    valid=receipt.get('fingerprint')==fingerprint()
    levels=sorted(int(n) for n,r in receipt.get('levels',{}).items() if valid and r.get('passed') and int(n) in (16,32,48,64))
    return {'verified_levels':levels,'max_frames':max([8]+levels),
        'max_points':100000 if valid and receipt.get('metal_100k',{}).get('passed') else 10000,
        'default_target':32,'room_ready':32 in levels,'receipt_valid':valid}


def require_capacity(count):
    maximum=capabilities()['max_frames']
    if count>maximum:
        raise AppError(f'MPS mới được kiểm chứng tối đa {maximum} ảnh; yêu cầu {count} ảnh. Chạy scripts/benchmark_capacity.py trước.',409)

import gc
import json
import os
import sys
import time
import threading
from pathlib import Path
import numpy as np
from .config import MODEL, VENDOR
from .geometry import preprocess, export


def reconstruct(paths, run: Path, emit, confidence=.5, smoke=False, max_points=10000):
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('CPU fallback đang bật. Phải khởi động lại với PYTORCH_ENABLE_MPS_FALLBACK=0')
    import torch
    if not torch.backends.mps.is_available(): raise RuntimeError('VGGT bắt buộc MPS; không tìm thấy MPS GPU')
    if not MODEL.exists(): raise RuntimeError('Thiếu trọng số VGGT. Chạy scripts/download_assets.py')
    sys.path.insert(0,str(VENDOR/'vggt'))
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    begin = time.monotonic()
    emit('Đọc model FP32; camera + depth heads', stage='model', progress=.03)
    model = VGGT(enable_point=False, enable_track=False)
    state = torch.load(MODEL,map_location='cpu',weights_only=True,mmap=True)
    state = {k:v for k,v in state.items() if not k.startswith(('point_head.','track_head.'))}
    model.load_state_dict(state,strict=True)
    del state
    model.eval().to(device='mps',dtype=torch.float32)
    assert all(p.device.type == 'mps' for p in model.parameters())
    arrays, masks, metas, names = [],[],[],[]
    (run/'images').mkdir(exist_ok=True)
    for i,path in enumerate(paths):
        a,m,train,meta = preprocess(path)
        name = f'{i:02}.png'
        train.save(run/'images'/name)
        arrays.append(a); masks.append(m); metas.append(meta); names.append(name)
    images = torch.from_numpy(np.stack(arrays)).permute(0,3,1,2).float().div_(255).to('mps')
    stats = {'backend':'mps','dtype':'float32','resolution':518,'images':len(paths),'model_load_seconds':time.monotonic()-begin}
    def infer(tensor):
        # Explicit MPS adapter: upstream's CUDA autocast context is not used.
        with torch.inference_mode():
            tokens,idx = model.aggregator(tensor.unsqueeze(0))
            pose = model.camera_head(tokens)[-1]
            depth, conf = model.depth_head(tokens,images=tensor.unsqueeze(0),patch_start_idx=idx)
            E,K = pose_encoding_to_extri_intri(pose,tensor.shape[-2:])
            torch.mps.synchronize()
            assert depth.device.type == conf.device.type == E.device.type == 'mps'
            return depth[0,...,0].cpu().numpy(),conf[0].cpu().numpy(),E[0].cpu().numpy(),K[0].cpu().numpy()
    if smoke:
        emit('Smoke test VGGT: 2 ảnh trên MPS',stage='vggt_smoke',progress=.1)
        start = time.monotonic()
        smoke_result = infer(images[:2])
        if not all(np.isfinite(x).all() for x in smoke_result): raise RuntimeError('Smoke test có giá trị không hữu hạn')
        del smoke_result
        torch.mps.empty_cache()
        stats['smoke_seconds'] = time.monotonic()-start
    emit(f'VGGT MPS: {len(paths)} ảnh trong cùng một inference',stage='vggt',progress=.2)
    start = time.monotonic()
    stopped=threading.Event()
    memory={'peak_mps_allocated_bytes':0,'peak_mps_driver_bytes':0}
    def sample_memory():
        while not stopped.wait(.1):
            memory['peak_mps_allocated_bytes']=max(memory['peak_mps_allocated_bytes'],torch.mps.current_allocated_memory())
            memory['peak_mps_driver_bytes']=max(memory['peak_mps_driver_bytes'],torch.mps.driver_allocated_memory())
    monitor=threading.Thread(target=sample_memory,daemon=True); monitor.start()
    try:
        result = infer(images)
        if not all(np.isfinite(x).all() for x in result): raise RuntimeError('VGGT có camera/depth/confidence không hữu hạn')
    finally:
        stopped.set(); monitor.join()
        stats.update(memory)
        (run/'mps-memory.json').write_text(json.dumps(stats,indent=2))
    stats['inference_seconds'] = time.monotonic()-start
    stats['mps_allocated_after_inference'] = torch.mps.current_allocated_memory()
    stats['mps_driver_after_inference'] = torch.mps.driver_allocated_memory()
    del images,model
    gc.collect(); torch.mps.empty_cache()
    emit('Xuất depth, cameras và COLMAP binary',stage='geometry',progress=.6)
    # Larger initialization is gated by a measured Metal training receipt.
    stats.update(export(run,arrays,masks,metas,names,*result,quantile=confidence,max_points=max_points))
    stats['geometry_total_seconds'] = time.monotonic()-begin
    (run/'geometry'/'metrics.json').write_text(json.dumps(stats,indent=2))
    emit('Geometry đã xuất; model được giải phóng khỏi GPU',stage='geometry',progress=.65,metrics=stats)
    return stats

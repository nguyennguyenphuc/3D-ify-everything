"""Device-generic VGGT reconstruction (CUDA on Colab, MPS on the Mac app).

The model call is isolated in ``load_model``/``run_model`` so tests can replace
it without torch. Everything after inference is ``studio.geometry.export``.
"""
import gc
import json
import sys
import threading
import time
from pathlib import Path
import numpy as np
from .config import MODEL, VENDOR
from .geometry import preprocess, export


def load_model(device, dtype, weights=None):
    import torch
    sys.path.insert(0, str(VENDOR/'vggt'))
    from vggt.models.vggt import VGGT
    weights = Path(weights or MODEL)
    if not weights.exists(): raise RuntimeError(f'Thiếu trọng số VGGT: {weights}')
    model = VGGT(enable_point=False, enable_track=False)
    state = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    state = {k: v for k, v in state.items() if not k.startswith(('point_head.', 'track_head.'))}
    model.load_state_dict(state, strict=True)
    del state
    # Keep FP32 parameters; CUDA uses bf16 autocast for the aggregator only.
    model.eval().to(device=device, dtype=torch.float32)
    return model


def run_model(model, arrays, device, dtype):
    """arrays: list of HxWx3 uint8 (already padded to the model size)."""
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float().div_(255).to(device)
    autocast = device.startswith('cuda') and dtype != 'float32'
    amp = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=amp, enabled=autocast):
            tokens, idx = model.aggregator(images.unsqueeze(0))
        with torch.autocast('cuda', enabled=False):
            pose = model.camera_head(tokens)[-1]
            depth, conf = model.depth_head(tokens, images=images.unsqueeze(0), patch_start_idx=idx)
        E, K = pose_encoding_to_extri_intri(pose, images.shape[-2:])
        synchronize(device)
        result = (depth[0, ..., 0].float().cpu().numpy(), conf[0].float().cpu().numpy(),
                  E[0].float().cpu().numpy(), K[0].float().cpu().numpy())
    del images
    return result


def synchronize(device):
    if device.startswith('cuda'):
        import torch; torch.cuda.synchronize()
    elif device == 'mps':
        import torch; torch.mps.synchronize()


def memory_probe(device):
    if device.startswith('cuda'):
        import torch
        return lambda: {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()}
    if device == 'mps':
        import torch
        return lambda: {'allocated': torch.mps.current_allocated_memory(), 'reserved': torch.mps.driver_allocated_memory()}
    return lambda: {'allocated': 0, 'reserved': 0}


def free(device):
    gc.collect()
    if device.startswith('cuda'):
        import torch; torch.cuda.empty_cache()
    elif device == 'mps':
        import torch; torch.mps.empty_cache()


def reconstruct(paths, run: Path, emit, confidence=.5, max_points=10000, device='cuda', dtype='bfloat16',
                size=518, train_max=1024, weights=None):
    begin = time.monotonic()
    emit(f'Đọc VGGT ({device}, {dtype}); camera + depth heads', stage='model', progress=.03)
    model = load_model(device, dtype, weights)
    arrays, masks, metas, names = [], [], [], []
    (run/'images').mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths):
        a, m, train, meta = preprocess(path, size=size, train_max=train_max)
        name = f'{i:02}.png' if len(paths) <= 100 else f'{i:03}.png'
        train.save(run/'images'/name)
        arrays.append(a); masks.append(m); metas.append(meta); names.append(name)
    stats = {'backend': device, 'dtype': dtype, 'resolution': size, 'train_max': train_max, 'images': len(paths),
             'model_load_seconds': time.monotonic()-begin}
    emit(f'VGGT {device}: {len(paths)} ảnh trong cùng một inference', stage='vggt', progress=.2)
    start = time.monotonic()
    probe = memory_probe(device)
    peak = {'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
    stopped = threading.Event()
    def sample():
        while not stopped.wait(.1):
            m = probe()
            peak['peak_allocated_bytes'] = max(peak['peak_allocated_bytes'], m['allocated'])
            peak['peak_reserved_bytes'] = max(peak['peak_reserved_bytes'], m['reserved'])
    monitor = threading.Thread(target=sample, daemon=True); monitor.start()
    try:
        result = run_model(model, arrays, device, dtype)
        if not all(np.isfinite(x).all() for x in result): raise RuntimeError('VGGT có camera/depth/confidence không hữu hạn')
    finally:
        stopped.set(); monitor.join()
        stats.update(peak)
        (run/'vggt-memory.json').write_text(json.dumps(stats, indent=2))
    stats['inference_seconds'] = time.monotonic()-start
    del model
    free(device)
    emit('Xuất depth, cameras và COLMAP binary', stage='geometry', progress=.6)
    stats.update(export(run, arrays, masks, metas, names, *result, quantile=confidence, max_points=max_points))
    stats['geometry_total_seconds'] = time.monotonic()-begin
    (run/'geometry'/'metrics.json').write_text(json.dumps(stats, indent=2))
    emit('Geometry đã xuất; model được giải phóng khỏi GPU', stage='geometry', progress=.65, metrics=stats)
    return stats

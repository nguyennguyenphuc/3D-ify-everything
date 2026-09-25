"""Full Courtyard Studio pipeline on a Colab GPU: VGGT -> 3DGUT -> ArtiFixer -> ArtiFixer3D.

    python -m colab.pipeline check
    python -m colab.pipeline setup                      # installs + weights (cached on Drive when mounted)
    python -m colab.pipeline run --source courtyard     # ETH3D 8-image smoke scene
    python -m colab.pipeline run --source video --input /content/drive/MyDrive/room.mp4 --name room
    python -m colab.pipeline selftest                   # CPU-only synthetic run of every non-model step
    python -m colab.pipeline tidy --dir out/courtyard/enhance   # crop a downloaded result (keeps *.full.ply)

Heavy imports stay inside functions so this module imports without torch/GPU.
When launched by colab.agent, small outputs go to $COLAB_PUBLISH_DIR (GitHub branch)
and large ones to $COLAB_RELEASE_DIR (GitHub release).
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO/'vendor'
ARTIFIXER = VENDOR/'ArtiFixer'
THREEDGRUT = ARTIFIXER/'thirdparty'/'3DGRUT-ArtiFixer'
ON_COLAB = Path('/content').is_dir()
MODEL_ID = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
CHECKPOINT = 'artifixer-1.3b.pt'
DEFAULT_CAPTION = ('A realistic, high-detail indoor space captured by a handheld camera moving slowly through the room, '
                   'sharp textures, natural lighting, consistent geometry, no motion blur.')
SETUP_VERSION = 1


# Paths --------------------------------------------------------------------
def cache_root():
    if os.environ.get('COURTYARD_CACHE'): return Path(os.environ['COURTYARD_CACHE'])
    drive = Path('/content/drive/MyDrive')
    if drive.is_dir(): return drive/'courtyard-cache'
    return Path('/content/cache') if ON_COLAB else REPO/'data'/'colab-cache'


def work_root():
    if os.environ.get('COURTYARD_WORK'): return Path(os.environ['COURTYARD_WORK'])
    return Path('/content/work') if ON_COLAB else REPO/'data'/'colab-work'


def model_env(cache):
    env = dict(os.environ, HF_HOME=str(cache/'hf'), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=os.pathsep.join([str(ARTIFIXER), str(THREEDGRUT), os.environ.get('PYTHONPATH', '')]))
    env.setdefault('WANDB_MODE', 'disabled')
    return env


class Log:
    def __init__(self, path=None):
        self.path = path

    def __call__(self, message, **fields):
        stage = fields.get('stage', '')
        print(f'[{time.strftime("%H:%M:%S")}] {stage and stage + ": "}{message}', flush=True)
        if self.path:
            event = dict(time=time.time(), message=message)
            event.update({k: v for k, v in fields.items() if k != 'metrics'})
            with open(self.path, 'a') as f: f.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')


def sh(cmd, log, cwd=None, env=None, check=True):
    """Run a command, streaming output; cmd is a list or a shell string."""
    shell = isinstance(cmd, str)
    log(f'$ {cmd if shell else " ".join(map(str, cmd))}', stage='cmd')
    p = subprocess.Popen(cmd, shell=shell, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         executable='/bin/bash' if shell else None)
    tail = []
    for line in p.stdout:
        print(line, end='', flush=True)
        tail = (tail + [line])[-40:]
    code = p.wait()
    if check and code != 0:
        raise RuntimeError(f'Lệnh thất bại (exit {code}): {cmd}\n' + ''.join(tail[-15:]))
    return code


# Check / setup --------------------------------------------------------------
def gpu():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap,driver_version',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout
        name, mem, cap, driver = [x.strip() for x in out.strip().splitlines()[0].split(',')]
        return {'name': name, 'memory_mib': int(mem), 'compute_capability': cap, 'driver': driver}
    except Exception:
        return None


def check(args=None):
    log = Log()
    info = {'gpu': gpu(), 'python': sys.version.split()[0], 'on_colab': ON_COLAB, 'cache': str(cache_root()),
            'work': str(work_root()), 'drive_mounted': Path('/content/drive/MyDrive').is_dir(),
            'disk_free_gb': round(shutil.disk_usage(REPO).free / 2**30, 1),
            'artifixer_cloned': (ARTIFIXER/'.git').exists(), 'vggt_cloned': (VENDOR/'vggt').exists(),
            'setup_marker': str(setup_marker()) if setup_marker().exists() else None}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
    except ImportError:
        info['torch'] = None
    log(json.dumps(info, indent=2, ensure_ascii=False), stage='check')
    g = info['gpu']
    if g and g['memory_mib'] < 39000:
        log('GPU dưới 40 GB: ArtiFixer 1.3B có thể thiếu VRAM; nên chọn A100 80 GB/H100 (High-RAM).', stage='check')
    return info


def setup_marker():
    return Path('/tmp')/f'courtyard-setup-v{SETUP_VERSION}.json'


def lock():
    return json.loads((REPO/'upstream.lock.json').read_text())


def clone(url, path, rev, log, submodules=False):
    if not (path/'.git').exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        sh(['git', 'clone', '--quiet', url, str(path)], log)
    current = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if current != rev:
        sh(['git', '-C', str(path), 'fetch', '--quiet', 'origin', rev], log)
        sh(['git', '-C', str(path), 'checkout', '--quiet', rev], log)
    if submodules:
        sh(['git', '-C', str(path), 'submodule', 'update', '--init', '--recursive', '--depth', '1'], log)


def download_vggt(cache, log):
    from huggingface_hub import HfApi, hf_hub_download
    target = cache/'models'/'vggt'/'model.pt'
    record = target.parent/'revision.json'
    if target.is_file() and record.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    revision = HfApi().model_info('facebook/VGGT-1B').sha
    log(f'Tải VGGT-1B @ {revision[:10]} (~5 GB)', stage='setup')
    path = Path(hf_hub_download('facebook/VGGT-1B', 'model.pt', revision=revision, local_dir=target.parent))
    if path != target: path.replace(target)
    digest = hashlib.sha256()
    with target.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''): digest.update(chunk)
    record.write_text(json.dumps({'repo': 'facebook/VGGT-1B', 'revision': revision, 'model_pt_sha256': digest.hexdigest()}, indent=2))
    return target


def setup(args):
    log = Log()
    cache = cache_root(); cache.mkdir(parents=True, exist_ok=True)
    info = check()
    pins = lock()
    py = sys.executable
    pip = [py, '-m', 'pip', 'install', '--quiet']
    sh(pip + ['plyfile', 'py7zr', 'einops', 'safetensors', 'huggingface_hub', 'hf_transfer', 'opencv-python-headless', 'scipy'], log)
    clone('https://github.com/facebookresearch/vggt.git', VENDOR/'vggt', pins['vggt'], log)
    env = model_env(cache); os.environ.update(HF_HOME=env['HF_HOME'])
    download_vggt(cache, log)
    if args.skip_artifixer:
        setup_marker().write_text(json.dumps({'artifixer': False, 'time': time.time()}))
        return
    clone('https://github.com/nv-tlabs/ArtiFixer.git', ARTIFIXER, pins['ArtiFixer'], log, submodules=True)
    torch_ok = False
    try:
        out = subprocess.run([py, '-c', 'import torch;print(torch.__version__)'], capture_output=True, text=True).stdout.strip()
        torch_ok = out.startswith(args.torch_version)
    except OSError:
        pass
    if not torch_ok:
        log(f'Cài torch=={args.torch_version} ({args.torch_index}) theo Dockerfile.cuda12 của ArtiFixer', stage='setup')
        sh(pip + [f'torch=={args.torch_version}', 'torchvision', '--index-url', args.torch_index], log)
    sh([py, '-m', 'pip', 'uninstall', '-y', 'flash-attn'], log, check=False)
    sh(pip + ['-r', str(THREEDGRUT/'requirements.txt')], log)
    sh(['bash', str(THREEDGRUT/'scripts'/'install_slangc.sh'), '/usr/local'], log)
    sh(pip + ['-e', str(THREEDGRUT)], log)
    sh(pip + ['accelerate==1.13.0', 'diffusers==0.37.1', 'transformers==5.5.0', 'ftfy', 'einops', 'scipy', 'wandb', 'tqdm',
              'Pillow', 'matplotlib', 'opencv-python-headless', 'pyyaml', 'torchmetrics', 'imageio-ffmpeg', 'h5py', 'av',
              'torch-fidelity', 'git+https://github.com/microsoft/MoGe.git'], log)
    major = int(float((info.get('gpu') or {}).get('compute_capability', '0')))
    if args.fa4 and major >= 9:
        sh(pip + ['--pre', 'flash-attn-4'], log)
        sh(pip + ['--force-reinstall', '--no-deps', 'cuda-python==12.6.2.post1'], log)
    # A fresh interpreter proves the stack imports together.
    sh([py, '-c', 'import torch, diffusers, transformers, h5py, threedgrut; from moge.model.v2 import MoGeModel; '
               'import model_training.net.transformer; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'],
       log, cwd=ARTIFIXER, env=env)
    from huggingface_hub import hf_hub_download, snapshot_download
    ckpt_dir = cache/'models'/'artifixer'
    log('Tải checkpoint ArtiFixer 1.3B và base Wan2.1-T2V-1.3B-Diffusers', stage='setup')
    hf_hub_download('nvidia/ArtiFixer', CHECKPOINT, local_dir=ckpt_dir)
    snapshot_download(MODEL_ID, cache_dir=Path(env['HF_HOME'])/'hub')
    setup_marker().write_text(json.dumps({'artifixer': True, 'time': time.time(), 'pins': pins}))
    log('Setup xong', stage='setup')


# Inputs --------------------------------------------------------------------
def fetch_input(value, dest, log):
    if re.match(r'^https?://', value or ''):
        dest.mkdir(parents=True, exist_ok=True)
        target = dest/Path(value.split('?')[0]).name
        if not target.exists(): sh(['curl', '-fL', '--retry', '5', '-o', str(target), value], log)
        return target
    path = Path(value).expanduser()
    if not path.exists(): raise FileNotFoundError(f'Không thấy input: {path}')
    return path


def prepare_inputs(cfg, layout, log):
    from studio.sources import courtyard, extract_video, import_images
    from studio.selection import select_frames
    project = layout['project']; project.mkdir(parents=True, exist_ok=True)
    run = layout['run']; run.mkdir(parents=True, exist_ok=True)
    manifest_path = project/'manifest.json'
    if not manifest_path.exists():
        if cfg['source'] == 'courtyard':
            courtyard(project, log)
        elif cfg['source'] == 'video':
            extract_video(fetch_input(cfg['input'], project/'input', log), project, cfg['fps'], log, cfg['target_frames'])
        elif cfg['source'] == 'images':
            import_images(fetch_input(cfg['input'], project/'input', log), project, limit=2000)
        else:
            raise ValueError(f'source không hỗ trợ: {cfg["source"]}')
    manifest = json.loads(manifest_path.read_text())
    if cfg['source'] == 'courtyard' or len(manifest) <= cfg['target_frames'] and cfg['source'] == 'images':
        selection = {'images': manifest, 'connected': True, 'method': 'all images'}
        (run/'selection.json').write_text(json.dumps(selection, indent=2, ensure_ascii=False))
    else:
        selection = select_frames(project, manifest, run, log, cfg['target_frames'], cfg['max_frames'])
    return [project/m['file'] for m in selection['images']], selection


def reconstruct(cfg, layout, paths, selection, log):
    from studio.vggt import reconstruct as vggt
    weights = cache_root()/'models'/'vggt'/'model.pt'
    stats = vggt(paths, layout['run'], log, confidence=cfg['confidence'], max_points=cfg['max_points'],
                 device='cuda', dtype=cfg['vggt_dtype'], size=518, train_max=cfg['train_image_max'], weights=weights)
    camera_path = layout['run']/'geometry'/'cameras.json'
    cameras = json.loads(camera_path.read_text())
    for camera, item in zip(cameras['cameras'], selection['images']):
        camera.update(source_id=item['id'], timestamp_seconds=item.get('timestamp_seconds'))
    camera_path.write_text(json.dumps(cameras, indent=2))
    return stats


# Pose refinement: VGGT's official BA (demo_colmap.py --use_ba) -------------------
BA_PYTHON = '3.11'  # demo_colmap needs pycolmap==3.10, which has no wheels for Colab's Python 3.13


def read_colmap(sparse):
    """Legacy COLMAP binary model (what demo_colmap/pycolmap 3.10 and studio.geometry.write_colmap write).

    Returns {image name: {'w2c': 3x4, 'K': 3x3, 'wh': [w, h]}}; PINHOLE / SIMPLE_PINHOLE cameras only.
    """
    import struct
    from scipy.spatial.transform import Rotation
    sparse = Path(sparse)
    cams = {}
    with (sparse/'cameras.bin').open('rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            cid, model, w, h = struct.unpack('<iiQQ', f.read(24))
            n = {0: 3, 1: 4}.get(model)
            if n is None: raise ValueError(f'Camera model {model} chưa hỗ trợ (chỉ PINHOLE/SIMPLE_PINHOLE)')
            p = struct.unpack(f'<{n}d', f.read(8 * n))
            fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if n == 3 else p
            cams[cid] = {'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), 'wh': [int(w), int(h)]}
    images = {}
    with (sparse/'images.bin').open('rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            _, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack('<i4d3di', f.read(64))
            name = b''
            while (ch := f.read(1)) != b'\0': name += ch
            f.seek(24 * struct.unpack('<Q', f.read(8))[0], 1)
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            images[name.decode()] = {'w2c': np.column_stack([R, [tx, ty, tz]]), **cams[cid]}
    return images


def umeyama(src, dst):
    """Similarity (s, R, t) minimising |s R src + t - dst|² (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def apply_ba(cameras, refined, min_registered=.8, max_residual=.05):
    """Move BA poses/intrinsics into the feed-forward VGGT frame (so depth maps and points still line up).

    cameras: cameras.json entries (mutated in place on success). refined: read_colmap() output keyed by image name.
    Returns stats; 'applied' is False when too few images registered or the alignment does not fit.
    """
    from studio.geometry import camera_to_viewer
    pairs = [(c, refined[c['name']]) for c in cameras if c['name'] in refined]
    stats = {'registered': len(pairs), 'images': len(cameras), 'applied': False}
    if len(pairs) < max(3, min_registered * len(cameras)): return dict(stats, reason='quá ít ảnh được BA đăng ký')
    center = lambda E: -E[:, :3].T @ E[:, 3]
    ff = np.array([center(np.asarray(c['w2c'], float)) for c, _ in pairs])
    ba = np.array([center(r['w2c']) for _, r in pairs])
    s, R, t = umeyama(ba, ff)
    spread = np.linalg.norm(ff - ff.mean(0), axis=1).max() or 1.0
    residual = float(np.sqrt(((s * ba @ R.T + t - ff) ** 2).sum(1).mean()) / spread)
    stats.update(scale=float(s), residual=residual)
    if residual > max_residual: return dict(stats, reason=f'pose BA lệch quá xa pose VGGT (residual {residual:.3f})')
    rot_change = []
    for c, r in pairs:
        E = r['w2c']
        # x_cam = E [x_ba;1] and x_ba = Rᵀ(x_ff - t)/s; scaling camera coords by s leaves projections unchanged.
        new = np.column_stack([E[:, :3] @ R.T, s * E[:, 3] - E[:, :3] @ R.T @ t])
        old = np.asarray(c['w2c'], float)
        rot_change.append(float(np.degrees(np.arccos(np.clip((np.trace(new[:, :3] @ old[:, :3].T) - 1) / 2, -1, 1)))))
        sx, sy = c['wh'][0] / r['wh'][0], c['wh'][1] / r['wh'][1]
        K = np.diag([sx, sy, 1.]) @ r['K']
        c.update(w2c=new.tolist(), K=K.tolist(), viewer_c2w=camera_to_viewer(new).tolist(), pose_source='vggt+ba')
    return dict(stats, applied=True, max_rotation_change_deg=max(rot_change), mean_rotation_change_deg=float(np.mean(rot_change)))


def ba_python(cache, log):
    """Python 3.11 venv with the VGGT demo stack (torch, pycolmap 3.10, pyceres, LightGlue); built once per runtime."""
    env = Path('/content/ba-env') if ON_COLAB else cache/'ba-env'
    py = env/'bin'/'python'
    if py.exists(): return py
    # uv needs file locks, which Google Drive does not provide: keep its cache on the runtime disk.
    uv_env = dict(os.environ, UV_CACHE_DIR='/content/uv-cache' if ON_COLAB else str(cache/'uv'))
    sh([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv'], log)
    sh([sys.executable, '-m', 'uv', 'venv', '--python', BA_PYTHON, str(env)], log, env=uv_env)
    pip = [sys.executable, '-m', 'uv', 'pip', 'install', '--python', str(py)]
    sh(pip + ['torch==2.5.1', 'torchvision==0.20.1', '--index-url', 'https://download.pytorch.org/whl/cu124'], log, env=uv_env)
    sh(pip + ['numpy<2', 'pillow', 'huggingface_hub', 'einops', 'safetensors', 'trimesh', 'scipy', 'opencv-python-headless',
              'hydra-core', 'omegaconf', 'requests', 'pycolmap==3.10.0', 'pyceres==2.3',
              'lightglue @ git+https://github.com/jytime/LightGlue.git'], log, env=uv_env)
    return py


def refine_poses(layout, log, cache=None):
    """Run VGGT's demo_colmap.py --use_ba on the training images and fold the result into cameras.json."""
    cache = cache or cache_root()
    run = layout['run']
    camera_path = run/'geometry'/'cameras.json'
    data = json.loads(camera_path.read_text())
    scene = run/'ba'
    if scene.exists(): shutil.rmtree(scene)
    (scene/'images').mkdir(parents=True)
    for c in data['cameras']: shutil.copy2(run/'images'/c['name'], scene/'images'/c['name'])
    env = dict(model_env(cache), TORCH_HOME=str(cache/'torch'), PYTHONPATH=str(VENDOR/'vggt'))
    try:
        py = ba_python(cache, log)
        log(f'VGGT + bundle adjustment trên {len(data["cameras"])} ảnh (demo_colmap.py --use_ba)', stage='pose-ba')
        sh([str(py), str(REPO/'colab'/'vggt_ba.py'), '--weights', str(cache/'models'/'vggt'/'model.pt'), '--',
            '--scene_dir', str(scene), '--use_ba', '--shared_camera', '--camera_type', 'PINHOLE'], log, cwd=VENDOR/'vggt', env=env)
        stats = apply_ba(data['cameras'], read_colmap(scene/'sparse'))
    except (RuntimeError, OSError, ValueError) as e:
        stats = {'applied': False, 'reason': str(e)[-500:]}
    if stats['applied']:
        camera_path.write_text(json.dumps(data, indent=2))
        log(f'Pose BA áp dụng: {stats["registered"]}/{stats["images"]} ảnh, xoay tối đa {stats["max_rotation_change_deg"]:.2f}°, '
            f'residual {stats["residual"]:.4f}', stage='pose-ba')
    else:
        log(f'Giữ pose VGGT gốc: {stats.get("reason")}', stage='pose-ba')
    (run/'geometry'/'pose-ba.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


# ArtiFixer scene + trajectory -----------------------------------------------
def load_cameras(run):
    data = json.loads((run/'geometry'/'cameras.json').read_text())['cameras']
    for c in data:
        c['K'] = np.asarray(c['K'], float); c['w2c'] = np.asarray(c['w2c'], float)
    return data


def read_ply_xyzrgb(path):
    from plyfile import PlyData
    v = PlyData.read(str(path))['vertex']
    return (np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64),
            np.stack([v['red'], v['green'], v['blue']], 1).astype(np.uint8))


def target_size(wh, long_side):
    w, h = wh
    s = min(1.0, long_side / max(w, h))
    return max(16, int(round(w * s / 16)) * 16), max(16, int(round(h * s / 16)) * 16)


def shared_intrinsics(cameras, size):
    """One calibration for every view (ArtiFixer prep requires it), scaled to ``size``."""
    W, H = size
    fx = np.mean([c['K'][0, 0] * W / c['wh'][0] for c in cameras])
    fy = np.mean([c['K'][1, 1] * H / c['wh'][1] for c in cameras])
    cx = np.mean([(c['K'][0, 2] + .5) * W / c['wh'][0] - .5 for c in cameras])
    cy = np.mean([(c['K'][1, 2] + .5) * H / c['wh'][1] - .5 for c in cameras])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])


def build_artifixer_scene(run, out, long_side, log):
    """COLMAP scene for ArtiFixer: resized images, one shared PINHOLE camera, VGGT poses and points."""
    from PIL import Image
    from studio.geometry import write_colmap, project
    cameras = load_cameras(run)
    aspect = [c['wh'][0] / c['wh'][1] for c in cameras]
    if max(aspect) - min(aspect) > .01: raise ValueError('Ảnh nguồn có tỉ lệ khung khác nhau; ArtiFixer cần một calibration chung')
    size = target_size(cameras[0]['wh'], long_side)
    K = shared_intrinsics(cameras, size)
    images = out/'images'; images.mkdir(parents=True, exist_ok=True)
    for c in cameras:
        with Image.open(run/'images'/c['name']) as im:
            im.convert('RGB').resize(size, Image.Resampling.LANCZOS).save(images/c['name'])
    points, colors = read_ply_xyzrgb(run/'geometry'/'points.ply')
    owner = np.full(len(points), -1); uv = np.zeros((len(points), 2))
    for i, c in enumerate(cameras):
        pix, z = project(points, K, c['w2c'])
        ok = (owner < 0) & np.isfinite(pix).all(1) & (z > 0) & (pix[:, 0] >= 0) & (pix[:, 0] < size[0]) & (pix[:, 1] >= 0) & (pix[:, 1] < size[1])
        owner[ok] = i; uv[ok] = pix[ok]
    keep = owner >= 0
    if keep.sum() < 100: raise ValueError('Quá ít điểm VGGT nhìn thấy được sau khi đổi calibration')
    shared = [{'name': c['name'], 'wh': list(size), 'K': K.tolist(), 'w2c': c['w2c'].tolist()} for c in cameras]
    obs = np.column_stack([owner[keep], uv[keep]])
    write_colmap(out, shared, points[keep], colors[keep], obs)
    info = {'size': list(size), 'K': K.tolist(), 'points': int(keep.sum()), 'views': len(cameras)}
    (out/'scene.json').write_text(json.dumps(info, indent=2))
    log(f'Scene ArtiFixer: {len(cameras)} ảnh {size[0]}x{size[1]}, {int(keep.sum())} điểm', stage='artifixer')
    return info


def camera_order(cameras):
    if all(c.get('timestamp_seconds') is not None for c in cameras):
        return sorted(range(len(cameras)), key=lambda i: cameras[i]['timestamp_seconds'])
    centers = np.array([-c['w2c'][:3, :3].T @ c['w2c'][:3, 3] for c in cameras])
    order, left = [0], set(range(1, len(cameras)))
    while left:
        nxt = min(left, key=lambda j: np.linalg.norm(centers[j] - centers[order[-1]]))
        order.append(nxt); left.remove(nxt)
    return order


def catmull_rom(points, t):
    """Uniform Catmull-Rom through ``points`` (N,3) at parameters t in [0, N-1]."""
    p = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]]) if len(points) > 1 else np.repeat(points, 4, 0)
    out = []
    for s in t:
        i = min(int(math.floor(s)), len(points) - 2) if len(points) > 1 else 0
        u = s - i
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        out.append(.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    return np.array(out)


def make_trajectory(cameras, K, size, frames=81, lateral=.04):
    """Smooth novel path through the source cameras, as ArtiFixer transforms-style JSON (OpenGL c2w)."""
    from scipy.spatial.transform import Rotation, Slerp
    from studio.geometry import camera_to_viewer
    if frames < 5 or (frames - 1) % 4: raise ValueError('trajectory_frames phải có dạng 4k+1 (vd 81)')
    order = camera_order(cameras)
    w2c = [np.vstack([cameras[i]['w2c'][:3], [0, 0, 0, 1]]) for i in order]
    c2w = [np.linalg.inv(E) for E in w2c]
    centers = np.array([T[:3, 3] for T in c2w])
    keys = np.arange(len(c2w), dtype=float)
    t = np.linspace(0, len(c2w) - 1, frames)
    pos = catmull_rom(centers, t) if len(c2w) > 1 else np.repeat(centers, frames, 0)
    rot = Slerp(keys, Rotation.from_matrix([T[:3, :3] for T in c2w]))(t).as_matrix() if len(c2w) > 1 else np.repeat([c2w[0][:3, :3]], frames, 0)
    radius = float(np.median(np.linalg.norm(centers - centers.mean(0), axis=1))) if len(c2w) > 1 else 1.0
    offset = lateral * radius * np.sin(np.linspace(0, 2 * np.pi, frames))
    out = []
    for p, R, o in zip(pos, rot, offset):
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p + o * R[:, 0]
        E = np.linalg.inv(T)
        # Target-only cameras: ArtiFixer rejects trajectory frames that carry file_path.
        out.append({'transform_matrix': camera_to_viewer(E[:3]).tolist()})
    W, H = size
    return {'camera_model': 'OPENCV', 'w': int(W), 'h': int(H), 'fl_x': float(K[0, 0]), 'fl_y': float(K[1, 1]),
            'cx': float(K[0, 2]), 'cy': float(K[1, 2]), 'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0,
            'camera_angle_x': float(2 * math.atan(W / (2 * K[0, 0]))), 'camera_angle_y': float(2 * math.atan(H / (2 * K[1, 1]))),
            'frames': out}


def look_rotation(forward, up):
    """OpenCV camera-to-world rotation (x right, y down, z forward) looking along ``forward`` with no roll."""
    z = forward / np.linalg.norm(forward)
    y = -up - (-up @ z) * z
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def make_coverage_trajectory(cameras, K, size, up, floor, frames=321, spins=4, spin_frames=48,
                             inward=.35, bob=.10, pitch_down=-35., pitch_up=15.):
    """A walk along the sweeps that swings ``inward`` toward the room centre, plus ``spins`` 360° turns made at the
    innermost points (looking down at the floor and up), as one continuous video.

    Phone captures usually walk the walls; this adds views from inside the room.

    The source-path trajectory only revisits views the photos already cover; ArtiFixer then has nothing to repair
    where the capture never looked (floor, behind the walker, between sweeps). Every frame here is a target for
    ArtiFixer and ArtiFixer3D. Cameras stay level (no roll) so the horizon is stable for the video model.
    """
    from studio.geometry import camera_to_viewer
    if frames < 5 or (frames - 1) % 4: raise ValueError('trajectory_frames phải có dạng 4k+1 (vd 321)')
    up = np.asarray(up, float); up /= np.linalg.norm(up)
    order = camera_order(cameras)
    c2w = [np.linalg.inv(np.vstack([cameras[i]['w2c'][:3], [0, 0, 0, 1]])) for i in order]
    eyes = np.array([T[:3, 3] for T in c2w]); fwd = np.array([T[:3, 2] for T in c2w])
    n = len(eyes)
    middle = eyes.mean(0)
    eye_height = max(float(np.median(eyes @ up)) - floor, 1e-6)
    e1 = np.cross(up, [1., 0, 0] if abs(up[0]) < .9 else [0, 0, 1.]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    yaw_of = lambda f: math.atan2(f @ e2, f @ e1)
    pitch_of = lambda f: math.asin(np.clip(f @ up / np.linalg.norm(f), -1, 1))
    direction = lambda y, p: math.cos(p) * (math.cos(y) * e1 + math.sin(y) * e2) + math.sin(p) * up
    # Heading along the walk: interpolated yaw/pitch of the source photos (unwrapped so it never spins backwards).
    yaws = np.unwrap([yaw_of(f) for f in fwd]); pitches = np.array([pitch_of(f) for f in fwd])
    spin_frames = max(8, min(spin_frames, int((frames - 1) * .6) // max(spins, 1))) if spins else 0
    walk = frames - spins * spin_frames
    stations = [int(round((j + .5) * walk / spins)) for j in range(spins)]
    out, k = [], 0
    for w in range(walk):
        s = w / max(walk - 1, 1) * (n - 1)
        u = s / max(n - 1, 1)
        pos = catmull_rom(eyes, [s])[0] if n > 1 else eyes[0]
        yaw, pitch = float(np.interp(s, np.arange(n), yaws)), float(np.interp(s, np.arange(n), pitches))
        toward = middle - pos; toward -= (toward @ up) * up
        # 0 on the captured path at the ends and between stations, deepest inside the room at each station.
        depth = inward * (1 - math.cos(2 * math.pi * max(spins, 1) * u)) / 2
        pos = pos + depth * toward + bob * eye_height * math.sin(2 * math.pi * 2 * u) * up
        views = [(pos, yaw, pitch)]
        if k < spins and w == stations[k]:
            # Turn a full circle on the spot; pitch dips to the floor at the middle and looks up at the quarters.
            for i in range(1, spin_frames + 1):
                t = i / (spin_frames + 1)
                target = math.radians(pitch_down + (pitch_up - pitch_down) * (1 - math.cos(4 * math.pi * t)) / 2)
                views.append((pos, yaw + 2 * math.pi * (3 * t * t - 2 * t ** 3), pitch + math.sin(math.pi * t) * (target - pitch)))
            k += 1
        for p, y, pt in views:
            R = look_rotation(direction(y, pt), up)
            out.append({'transform_matrix': camera_to_viewer(np.column_stack([R.T, -R.T @ p])).tolist()})
    W, H = size
    return {'camera_model': 'OPENCV', 'w': int(W), 'h': int(H), 'fl_x': float(K[0, 0]), 'fl_y': float(K[1, 1]),
            'cx': float(K[0, 2]), 'cy': float(K[1, 2]), 'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0,
            'camera_angle_x': float(2 * math.atan(W / (2 * K[0, 0]))), 'camera_angle_y': float(2 * math.atan(H / (2 * K[1, 1]))),
            'frames': out}


def draw_plan(cameras, points, trajectory, dest, size=720):
    """Top-down sketch: VGGT points (grey), source cameras (blue), trajectory (orange, darker where it looks down)."""
    from PIL import Image, ImageDraw
    box = scene_box(cameras, points)
    axes = np.asarray(box['axes'])
    c2w = [np.asarray(f['transform_matrix']) @ np.diag([1, -1, -1, 1]) for f in trajectory['frames']]
    path = np.array([T[:3, 3] for T in c2w]) @ axes.T
    looks_down = np.array([T[:3, 2] @ axes[2] < -.4 for T in c2w])
    eyes = np.array([-np.asarray(c['w2c'])[:3, :3].T @ np.asarray(c['w2c'])[:3, 3] for c in cameras]) @ axes.T
    pts = (points @ axes.T)[::max(1, len(points) // 20000)]
    lo = np.minimum(pts.min(0), path.min(0))[:2]; hi = np.maximum(pts.max(0), path.max(0))[:2]
    scale = (size - 40) / max(hi - lo)
    px = lambda p: tuple(20 + (np.asarray(p)[:2] - lo) * scale)
    im = Image.new('RGB', (size, size), (16, 19, 18)); d = ImageDraw.Draw(im)
    for p in pts: d.point(px(p), fill=(90, 96, 94))
    for a, b, down in zip(path, path[1:], looks_down[1:]):
        d.line([px(a), px(b)], fill=(236, 120, 40) if not down else (160, 60, 10), width=3)
    for e in eyes: x, y = px(e); d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(90, 160, 255), width=2)
    stops = [i for i in range(1, len(path) - 1) if np.allclose(path[i], path[i - 1]) and not np.allclose(path[i - 1], path[max(i - 2, 0)])]
    for i in stops: x, y = px(path[i]); d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(236, 120, 40), width=3)
    d.text((10, 8), 'xam: diem VGGT  xanh: camera da quay  cam: quy dao ArtiFixer, vong tron = xoay 360 (dam = nhin xuong san)', fill=(230, 230, 230))
    im.save(dest, quality=88)
    return dest


def trajectory_for(cfg, run, scene):
    cameras = load_cameras(run)
    if cfg.get('trajectory', 'coverage') == 'path':
        return make_trajectory(cameras, np.array(scene['K']), scene['size'], cfg['trajectory_frames'], cfg['trajectory_lateral'])
    points, _ = read_ply_xyzrgb(run/'geometry'/'points.ply')
    box = scene_box(cameras, points)
    return make_coverage_trajectory(cameras, np.array(scene['K']), scene['size'], box['up'], box['floor'],
                                    cfg['trajectory_frames'], cfg.get('coverage_spins', 4))


# ArtiFixer stages -------------------------------------------------------------
def af(cmd, log, cache, **kw):
    return sh(cmd, log, cwd=ARTIFIXER, env=model_env(cache), **kw)


def run_artifixer(cfg, layout, log):
    cache = cache_root()
    if not (ARTIFIXER/'.git').exists(): raise RuntimeError('Chưa setup ArtiFixer. Chạy: python -m colab.pipeline setup')
    root = layout['artifixer']; root.mkdir(parents=True, exist_ok=True)
    scene_dir = root/'colmap'
    if not (scene_dir/'scene.json').exists():
        build_artifixer_scene(layout['run'], scene_dir, cfg['artifixer_image_max'], log)
    scene = json.loads((scene_dir/'scene.json').read_text())
    trajectory = trajectory_for(cfg, layout['run'], scene)
    (root/'trajectory.json').write_text(json.dumps(trajectory, indent=2))
    scene_id = layout['scene_id']
    prep = root/'prep'/scene_id
    caption = prep/'captions'/scene_id/'caption.h5'
    py = sys.executable
    if not caption.exists():
        log('Mã hoá caption bằng text encoder Wan (umt5) — bỏ qua Qwen3-VL 30B', stage='caption')
        af([py, str(REPO/'colab'/'artifixer_tools.py'), 'caption', '--out', str(caption), '--text', cfg['caption'],
            '--model_id', MODEL_ID, '--images', str(scene_dir/'images')], log, cache)
    log('ArtiFixer prep: 3DGUT MCMC + render quỹ đạo + MoGe scale', stage='prepare')
    prep_cmd = [py, str(REPO/'colab'/'artifixer_tools.py'), 'prep', '--colmap_dir', str(scene_dir),
                '--output_root', str(prep), '--trajectory_path', str(root/'trajectory.json'),
                '--reconstruction_steps', str(cfg['reconstruction_steps']), '--text_encoder_model_id', MODEL_ID]
    if cfg.get('metric_scale'): prep_cmd += ['--metric_scale', str(cfg['metric_scale'])]
    af(prep_cmd, log, cache)
    infer = root/'infer'
    pred = find_frames(infer, scene_id, 'pred')
    if pred is None:
        log('ArtiFixer 1.3B: sửa artifact trên quỹ đạo mới', stage='inference')
        af([py, str(REPO/'colab'/'artifixer_tools.py'), 'infer', '--evalset', 'reconstructed_colmap',
            '--checkpoint_pt', str(cache/'models'/'artifixer'/CHECKPOINT), '--model_id', MODEL_ID, '--save_dir', str(infer),
            '--split_path', str(prep/'split.json'), '--render_trajectory', 'trajectory', '--save_frame_outputs_only']
           + (['--max_neighbors_per_encode', '1'] if cfg['low_memory'] else []), log, cache)
        pred = find_frames(infer, scene_id, 'pred')
    if pred is None: raise RuntimeError(f'Không thấy frame dự đoán của ArtiFixer trong {infer}')
    if cfg['artifixer3d_steps'] > 0:
        log(f'ArtiFixer3D: distill {cfg["artifixer3d_steps"]} bước (anchor thật + frame đã sửa)', stage='distill')
        af([py, '-m', 'data_processing.run_artifixer3d', '--scene_root', str(prep), '--artifixer_frames_dir', str(pred),
            '--artifixer3d_steps', str(cfg['artifixer3d_steps']), '--phases', 'distill,render'], log, cache)
    ckpts = {'baseline': json.loads((prep/'split.json').read_text())['test'][scene_id].get('reconstruction_checkpoint')}
    if ckpts['baseline'] and not Path(ckpts['baseline']).is_absolute(): ckpts['baseline'] = str(prep/ckpts['baseline'])
    distilled = sorted(prep.glob(f'artifixer3d/runs/**/ckpt_{cfg["artifixer3d_steps"]}.pt'))
    if distilled: ckpts['artifixer3d'] = str(distilled[-1])
    for name, ckpt in ckpts.items():
        target = root/f'{name}_3dgrut.ply'
        if ckpt and not target.exists():
            af([py, str(REPO/'colab'/'artifixer_tools.py'), 'export', '--checkpoint', ckpt, '--out', str(target)], log, cache)
    return {'prep': str(prep), 'pred': str(pred), 'checkpoints': ckpts, 'scene': scene}


def find_frames(infer, scene_id, kind):
    found = sorted(Path(infer).glob(f'**/{scene_id}/**/frames/batch_0000/{kind}')) + sorted(Path(infer).glob(f'**/{scene_id}/frames/batch_0000/{kind}'))
    found = [p for p in found if any(p.glob('*.png'))]
    return found[0] if found else None


# PLY conversion --------------------------------------------------------------
SH_REST = {0, 9, 24, 45}


def convert_3dgrut_ply(src, dst, min_opacity_logit=-8.0):
    """3DGRUT PLY (INRIA layout + placeholder normals) -> compact 3DGS PLY for Spark/validate_splat."""
    from plyfile import PlyData, PlyElement
    from studio.training import validate_splat
    v = PlyData.read(str(src))['vertex'].data
    names = v.dtype.names
    rest = sorted((n for n in names if n.startswith('f_rest_')), key=lambda n: int(n.split('_')[-1]))
    per_channel = len(rest) // 3
    keep_rest = max(k for k in SH_REST if k <= per_channel * 3 and k % 3 == 0)
    if keep_rest != len(rest):
        # f_rest is channel-major: R coeffs, then G, then B. Truncate each channel block.
        per = keep_rest // 3
        rest = [f'f_rest_{c * per_channel + i}' for c in range(3) for i in range(per)]
    fields = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + rest + ['opacity', 'scale_0', 'scale_1', 'scale_2',
                                                                   'rot_0', 'rot_1', 'rot_2', 'rot_3']
    missing = [f for f in fields if f not in names]
    if missing: raise ValueError(f'PLY thiếu trường {missing}')
    data = np.stack([np.asarray(v[f], np.float64) for f in fields], 1)
    ok = np.isfinite(data).all(1) & (data[:, fields.index('opacity')] >= min_opacity_logit)
    data = data[ok].astype(np.float32)
    out_names = fields[:6] + [f'f_rest_{i}' for i in range(len(rest))] + fields[6 + len(rest):]
    out = np.empty(len(data), dtype=[(n, 'f4') for n in out_names])
    for i, n in enumerate(out_names): out[n] = data[:, i]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, 'vertex')], text=False).write(str(dst))
    return {'splats': validate_splat(dst), 'dropped': int((~ok).sum()), 'sh_rest': len(rest)}


# Tidy: crop to the captured region --------------------------------------------
def scene_frame(cameras, points):
    """Up axis from the cameras (OpenCV y points down), horizontal axes from the VGGT points."""
    down = np.mean([np.asarray(c['w2c'])[1, :3] for c in cameras], 0)
    up = -down / np.linalg.norm(down)
    flat = points - np.outer(points @ up, up)
    flat = flat - flat.mean(0)
    a1 = np.linalg.svd(flat[:: max(1, len(flat) // 20000)], full_matrices=False)[2][0]
    a1 = a1 - (a1 @ up) * up; a1 /= np.linalg.norm(a1)
    return np.stack([a1, np.cross(up, a1), up])


def scene_box(cameras, points, margin=.1):
    axes = scene_frame(cameras, points)
    centers = np.array([-np.asarray(c['w2c'])[:3, :3].T @ np.asarray(c['w2c'])[:3, 3] for c in cameras])
    p, c = points @ axes.T, centers @ axes.T
    low = np.minimum(np.quantile(p, .01, 0), c.min(0)); high = np.maximum(np.quantile(p, .99, 0), c.max(0))
    pad = (high - low) * margin + 1e-6
    low, high = low - pad, high + pad
    return {'axes': axes.tolist(), 'up': axes[2].tolist(), 'center': (axes.T @ ((low + high) / 2)).tolist(),
            'half_size': ((high - low) / 2).tolist(), 'floor': float(np.quantile(p[:, 2], .02)),
            'ceiling': float(np.quantile(p[:, 2], .98)), 'sweeps': centers.tolist()}


def free_space(xyz, cameras, depth_dir, margin=.85, min_views=2):
    """True for Gaussians seen in front of the VGGT surface (z < margin·depth) by at least ``min_views`` cameras.

    Such Gaussians float in space every camera saw through, the typical fog between the viewer and a wall.
    depth_XX.npz and model_K are at VGGT model resolution, in the same world units as w2c.
    """
    from studio.geometry import project
    hits = np.zeros(len(xyz), np.int32)
    for i, c in enumerate(cameras):
        path = Path(depth_dir)/f'depth_{i:02}.npz'
        if not path.exists() or 'model_K' not in c: continue
        with np.load(path) as d: depth, valid = d['depth'], d['valid']
        pix, z = project(xyz, np.asarray(c['model_K'], float), np.asarray(c['w2c'], float))
        h, w = depth.shape
        ok = np.isfinite(pix).all(1) & (z > 0)
        uv = np.rint(np.where(ok[:, None], pix, -1)).astype(np.int64)
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx = np.flatnonzero(ok)
        u, v = uv[idx, 0], uv[idx, 1]
        front = valid[v, u] & (z[idx] < margin * depth[v, u])
        hits[idx[front]] += 1
    return hits >= min_views


def tidy_splat(src, dst, box, min_opacity=.05, max_scale=.05, neighbors=8, isolation=5.0, cameras=None, depth_dir=None):
    """Drop Gaussians outside ``box``, nearly transparent ones, oversized blobs, isolated and free-space floaters."""
    from plyfile import PlyData, PlyElement
    from scipy.spatial import cKDTree
    from studio.training import validate_splat
    v = PlyData.read(str(src), mmap=False)['vertex'].data
    xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    local = (xyz - np.asarray(box['center'])) @ np.asarray(box['axes']).T
    half = np.asarray(box['half_size'])
    inside = (np.abs(local) <= half).all(1)
    opaque = 1 / (1 + np.exp(-np.asarray(v['opacity'], np.float64))) >= min_opacity
    size = np.exp(np.stack([v[f'scale_{i}'] for i in range(3)], 1).max(1))
    small = size <= max_scale * 2 * np.linalg.norm(half)
    keep = inside & opaque & small
    floating = 0
    if cameras and depth_dir and Path(depth_dir).is_dir():
        idx = np.flatnonzero(keep)
        drop = free_space(xyz[idx], cameras, depth_dir)
        floating = int(drop.sum())
        keep[idx[drop]] = False
    isolated = 0
    if keep.sum() > neighbors * 4:
        pts = xyz[keep]
        dist = cKDTree(pts).query(pts, k=neighbors + 1, workers=-1)[0][:, -1]
        alone = dist > isolation * np.median(dist)
        isolated = int(alone.sum())
        keep[np.flatnonzero(keep)[alone]] = False
    if not keep.any(): raise ValueError(f'Cắt gọn bỏ hết Gaussian trong {src}; kiểm tra cameras.json/points.ply')
    kept, total = v[keep].copy(), len(v)
    del v
    # src may equal dst: write beside it, then swap.
    tmp = Path(dst).with_suffix('.tidy.tmp')
    PlyData([PlyElement.describe(kept, 'vertex')], text=False).write(str(tmp))
    tmp.replace(dst)
    return {'before': total, 'after': validate_splat(dst), 'outside_box': int((~inside).sum()),
            'transparent': int((~opaque).sum()), 'oversized': int((~small).sum()), 'floating': floating, 'isolated': isolated}


def tidy_dir(folder, keep_full=False, log=None, depth_dir=None):
    """Crop splat.ply/baseline.ply in an enhance folder in place and write scene.json for the viewer."""
    folder = Path(folder)
    cameras = json.loads((folder/'cameras.json').read_text())['cameras']
    points, _ = read_ply_xyzrgb(folder/'points.ply')
    box = scene_box(cameras, points)
    stats = {}
    for name in ('splat', 'baseline'):
        target, full = folder/f'{name}.ply', folder/f'{name}.full.ply'
        if not target.exists() and not full.exists(): continue
        if keep_full and not full.exists(): target.replace(full)
        stats[name] = tidy_splat(full if full.exists() else target, target, box, cameras=cameras, depth_dir=depth_dir)
        if log: log(f'{name}.ply: {stats[name]["before"]:,} → {stats[name]["after"]:,} Gaussian', stage='tidy')
    (folder/'scene.json').write_text(json.dumps(dict(box, tidy=stats), indent=2))
    return stats


# Packaging ------------------------------------------------------------------
def export_photos(layout, enhance, long_side=1920):
    """Source frames for the viewer's photo mode, named after cameras.json entries (00.png -> photos/00.jpg).

    Same framing as the training images (they are scaled copies), so cameras.json K still applies.
    """
    from PIL import Image, ImageOps
    run, project = layout['run'], layout['project']
    cameras = json.loads((run/'geometry'/'cameras.json').read_text())['cameras']
    selection = json.loads((run/'selection.json').read_text()) if (run/'selection.json').exists() else {}
    items = selection.get('images') or []
    out = enhance/'photos'; out.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(cameras):
        src = project/items[i]['file'] if i < len(items) and (project/items[i]['file']).exists() else run/'images'/c['name']
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            im.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
            im.save(out/(Path(c['name']).stem + '.jpg'), quality=90)
    return len(cameras)


def contact_sheet(pairs, dest, width=1600):
    from PIL import Image, ImageDraw
    rows = []
    for left, right in pairs:
        with Image.open(left) as a, Image.open(right) as b:
            a, b = a.convert('RGB'), b.convert('RGB').resize(a.size)
            row = Image.new('RGB', (a.width * 2 + 8, a.height), (16, 19, 18))
            row.paste(a, (0, 0)); row.paste(b, (a.width + 8, 0)); rows.append(row)
    if not rows: return None
    sheet = Image.new('RGB', (rows[0].width, sum(r.height for r in rows) + 36), (16, 19, 18))
    ImageDraw.Draw(sheet).text((10, 10), '3DGUT render (trái)  ·  ArtiFixer (phải)', fill=(230, 230, 230))
    y = 36
    for r in rows: sheet.paste(r, (0, y)); y += r.height
    scale = min(1.0, width / sheet.width)
    sheet = sheet.resize((int(sheet.width * scale), int(sheet.height * scale)))
    sheet.save(dest, quality=85)
    return dest


def package(cfg, layout, result, log):
    enhance = layout['enhance']
    if enhance.exists(): shutil.rmtree(enhance)
    enhance.mkdir(parents=True)
    run, root = layout['run'], layout['artifixer']
    metrics = {'name': layout['scene_id'], 'config': cfg, 'gpu': gpu(), 'stages': layout.get('timings', {})}
    for f in ('geometry/cameras.json', 'geometry/metrics.json', 'selection.json'):
        if (run/f).exists(): shutil.copy2(run/f, enhance/Path(f).name)
    if (run/'geometry'/'points.ply').exists(): shutil.copy2(run/'geometry'/'points.ply', enhance/'points.ply')
    if (root/'trajectory.json').exists(): shutil.copy2(root/'trajectory.json', enhance/'trajectory.json')
    for name in ('artifixer3d', 'baseline'):
        src = root/f'{name}_3dgrut.ply'
        if src.exists():
            metrics[f'{name}_ply'] = convert_3dgrut_ply(src, enhance/('splat.ply' if name == 'artifixer3d' else 'baseline.ply'))
    if not (enhance/'splat.ply').exists() and (enhance/'baseline.ply').exists():
        shutil.copy2(enhance/'baseline.ply', enhance/'splat.ply')
        metrics['note'] = 'Chưa có ArtiFixer3D; splat.ply là reconstruction 3DGUT gốc'
    if (enhance/'cameras.json').exists() and (enhance/'points.ply').exists():
        metrics['tidy'] = tidy_dir(enhance, log=log, depth_dir=run/'geometry')
    metrics['photos'] = export_photos(layout, enhance)
    pred = Path(result['pred']) if result.get('pred') else None
    pairs = []
    if pred and pred.exists():
        rendered = pred.parent/'rendered'
        for kind, src in (('fixed_frames', pred), ('rendered_frames', rendered)):
            if src.exists():
                shutil.copytree(src, enhance/kind)
        frames = sorted((enhance/'fixed_frames').glob('*.png'))
        pick = [frames[i] for i in np.linspace(0, len(frames) - 1, min(4, len(frames))).astype(int)] if frames else []
        pairs = [(enhance/'rendered_frames'/p.name, p) for p in pick if (enhance/'rendered_frames'/p.name).exists()]
        for video in sorted(pred.parents[2].glob('videos/*pred*.mp4'))[:1]:
            shutil.copy2(video, enhance/'preview.mp4')
    scale = Path(result.get('prep', ''))/'metric_alignment'/'scale_info.txt'
    if scale.exists(): metrics['metric_scale'] = scale.read_text().strip()
    contact_sheet(pairs, enhance/'compare.jpg')
    shutil.copy2(REPO/'colab'/'viewer.html', enhance/'index.html')
    metrics['files'] = sorted(p.relative_to(enhance).as_posix() for p in enhance.rglob('*') if p.is_file())
    (enhance/'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    archive = layout['base']/'results.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(enhance.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(enhance).as_posix())
    publish(enhance, archive, log)
    log(f'Kết quả: {enhance} · {archive} ({archive.stat().st_size / 2**20:.1f} MiB)', stage='package')
    return metrics


def publish(enhance, archive, log):
    pub, rel = os.environ.get('COLAB_PUBLISH_DIR'), os.environ.get('COLAB_RELEASE_DIR')
    if pub:
        pub = Path(pub); pub.mkdir(parents=True, exist_ok=True)
        for name in ('metrics.json', 'cameras.json', 'scene.json', 'selection.json', 'trajectory.json', 'compare.jpg'):
            if (enhance/name).exists(): shutil.copy2(enhance/name, pub/name)
        frames = sorted((enhance/'fixed_frames').glob('*.png'))
        for p in frames[::max(1, len(frames) // 4)][:4]:
            shutil.copy2(p, pub/f'fixed_{p.name}')
            if (enhance/'rendered_frames'/p.name).exists(): shutil.copy2(enhance/'rendered_frames'/p.name, pub/f'rendered_{p.name}')
    if rel:
        rel = Path(rel); rel.mkdir(parents=True, exist_ok=True)
        for name in ('splat.ply', 'baseline.ply', 'preview.mp4'):
            if (enhance/name).exists(): shutil.copy2(enhance/name, rel/name)
        shutil.copy2(archive, rel/'results.zip')
        log('Đã chuẩn bị file cho GitHub release', stage='package')
    drive = Path('/content/drive/MyDrive')
    if drive.is_dir():
        out = drive/'courtyard-results'/f'{enhance.parent.name}-{time.strftime("%Y%m%d-%H%M%S")}.zip'
        out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(archive, out)
        log(f'Đã lưu {out}', stage='package')


# Orchestration -----------------------------------------------------------------
STAGES = ('inputs', 'vggt', 'artifixer', 'package')


def layout_for(name):
    base = work_root()/name
    return {'base': base, 'project': base/'project', 'run': base/'run', 'artifixer': base/'artifixer',
            'enhance': base/'enhance', 'scene_id': name, 'stages': base/'stages', 'timings': {}}


def run(args):
    cfg = {k: v for k, v in vars(args).items() if k not in ('func', 'command')}
    cfg['name'] = re.sub(r'[^A-Za-z0-9_-]+', '-', cfg['name'] or cfg['source']).strip('-') or 'scene'
    os.environ.setdefault('STUDIO_DATA', str(cache_root()/'studio'))
    os.environ.setdefault('HF_HOME', str(cache_root()/'hf'))
    sys.path.insert(0, str(REPO))
    layout = layout_for(cfg['name'])
    if args.force and layout['base'].exists(): shutil.rmtree(layout['base'])
    for key in ('base', 'stages'): layout[key].mkdir(parents=True, exist_ok=True)
    log = Log(layout['base']/'progress.jsonl')
    (layout['base']/'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    stages = [s.strip() for s in cfg['stages'].split(',') if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown: raise SystemExit(f'Stage không hợp lệ: {unknown}')
    state, result = {}, {}
    marker = lambda s: layout['stages']/f'{s}.json'
    for stage in STAGES:
        if stage not in stages:
            # Later stages read earlier results, so load finished ones even when not requested.
            if marker(stage).exists(): state[stage] = json.loads(marker(stage).read_text())
            continue
        if marker(stage).exists() and stage != 'package':
            state[stage] = json.loads(marker(stage).read_text())
            log(f'Bỏ qua (đã xong ở lần chạy trước)', stage=stage)
            continue
        t0 = time.time()
        log('Bắt đầu', stage=stage)
        if stage == 'inputs':
            if cfg['fake']: paths, selection = synthetic_scene(layout, log), None
            else: paths, selection = prepare_inputs(cfg, layout, log)
            state[stage] = {'images': [str(p) for p in paths], 'selection': selection and {k: selection[k] for k in ('images', 'method') if k in selection}}
        elif stage == 'vggt':
            if cfg['fake']: state[stage] = {'fake': True}
            else:
                if 'inputs' not in state: raise SystemExit('Chưa có kết quả stage inputs; chạy --stages inputs trước')
                sel = state['inputs']['selection'] or {'images': [{'id': str(i)} for i in range(len(state['inputs']['images']))]}
                state[stage] = reconstruct(cfg, layout, [Path(p) for p in state['inputs']['images']], sel, log)
                if cfg.get('pose_ba'): state[stage]['pose_ba'] = refine_poses(layout, log)
        elif stage == 'artifixer':
            state[stage] = fake_artifixer(cfg, layout, log) if cfg['fake'] else run_artifixer(cfg, layout, log)
        elif stage == 'package':
            state[stage] = package(cfg, layout, state.get('artifixer', {}), log)
        layout['timings'][stage] = round(time.time() - t0, 1)
        if stage != 'package': marker(stage).write_text(json.dumps(state[stage], indent=2, default=str, ensure_ascii=False))
        log(f'Xong sau {time.time() - t0:.0f} giây', stage=stage)
    return state


# Synthetic CPU path (selftest / --fake) ---------------------------------------
def synthetic_scene(layout, log, n=6, size=(320, 240)):
    """Textured plane scene with known cameras; writes the same files studio.geometry.export does."""
    from PIL import Image
    from studio.geometry import write_colmap, write_ply, camera_to_viewer, project
    rng = np.random.default_rng(0)
    run = layout['run']; (run/'images').mkdir(parents=True, exist_ok=True); (run/'geometry').mkdir(parents=True, exist_ok=True)
    W, H = size
    K = np.array([[260., 0, W / 2 - .5], [0, 260., H / 2 - .5], [0, 0, 1]])
    points = np.column_stack([rng.uniform(-2, 2, 4000), rng.uniform(-1.5, 1.5, 4000), rng.uniform(3.5, 4.5, 4000)])
    colors = rng.integers(0, 255, (4000, 3)).astype(np.uint8)
    cameras, obs = [], []
    for i in range(n):
        angle = (i - n / 2) * .06
        R = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
        t = np.array([-.3 * (i - n / 2) * .5, 0, 0])
        E = np.column_stack([R, t])
        pix, z = project(points, K, E)
        img = np.full((H, W, 3), 40, np.uint8)
        ok = (z > 0) & (pix[:, 0] >= 0) & (pix[:, 0] < W - 1) & (pix[:, 1] >= 0) & (pix[:, 1] < H - 1)
        xy = np.rint(pix[ok]).astype(int); img[xy[:, 1], xy[:, 0]] = colors[ok]
        name = f'{i:02}.png'
        Image.fromarray(img).save(run/'images'/name)
        cameras.append({'name': name, 'wh': [W, H], 'K': K.tolist(), 'model_K': K.tolist(), 'w2c': E.tolist(),
                        'viewer_c2w': camera_to_viewer(E).tolist(), 'timestamp_seconds': float(i), 'source_id': str(i)})
        obs += [(i, *p) for p in pix[ok][:200]]
    write_ply(run/'geometry'/'points.ply', points, colors)
    write_colmap(run, cameras, points[:len(obs)], colors[:len(obs)], np.array(obs))
    (run/'geometry'/'cameras.json').write_text(json.dumps({'convention': 'OpenCV world-to-camera', 'cameras': cameras}, indent=2))
    (run/'geometry'/'metrics.json').write_text(json.dumps({'camera_count': n, 'synthetic': True}))
    log(f'Scene tổng hợp {n} ảnh', stage='inputs')
    return [run/'images'/c['name'] for c in cameras]


def synthetic_3dgrut_ply(path, n=500, sh_rest=45, seed=0):
    from plyfile import PlyData, PlyElement
    rng = np.random.default_rng(seed)
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{i}' for i in range(sh_rest)] + \
            ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    arr = np.empty(n, dtype=[(k, 'f4') for k in names])
    for k in names: arr[k] = rng.normal(0, 1, n)
    arr['opacity'] = rng.uniform(-3, 3, n); arr['nx'] = arr['ny'] = 0; arr['nz'] = 1
    # Small Gaussians on the synthetic_scene plane, like a trained splat, so tidy keeps them.
    arr['x'], arr['y'], arr['z'] = rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n), rng.uniform(3.5, 4.5, n)
    for i in range(3): arr[f'scale_{i}'] = rng.normal(-4, .3, n)
    arr['x'][0] = np.nan
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(str(path))


def fake_artifixer(cfg, layout, log):
    from PIL import Image, ImageFilter
    root = layout['artifixer']; root.mkdir(parents=True, exist_ok=True)
    scene = build_artifixer_scene(layout['run'], root/'colmap', cfg['artifixer_image_max'], log)
    traj = trajectory_for(cfg, layout['run'], scene)
    (root/'trajectory.json').write_text(json.dumps(traj, indent=2))
    frames = root/'infer'/'fake'/layout['scene_id']/'frames'/'batch_0000'
    for kind in ('pred', 'rendered'): (frames/kind).mkdir(parents=True, exist_ok=True)
    src = sorted((root/'colmap'/'images').glob('*.png'))
    for i in range(len(traj['frames'])):
        with Image.open(src[i % len(src)]) as im:
            im.save(frames/'pred'/f'{i:05d}.png')
            im.filter(ImageFilter.GaussianBlur(3)).save(frames/'rendered'/f'{i:05d}.png')
    for name in ('baseline', 'artifixer3d'): synthetic_3dgrut_ply(root/f'{name}_3dgrut.ply')
    log('ArtiFixer giả lập (không GPU)', stage='artifixer')
    return {'pred': str(frames/'pred'), 'prep': str(root), 'checkpoints': {}, 'scene': scene, 'fake': True}


def selftest(args):
    args.fake, args.source, args.name = True, 'synthetic', args.name or 'selftest'
    args.force = True
    state = run(args)
    enhance = layout_for(re.sub(r'[^A-Za-z0-9_-]+', '-', args.name))['enhance']
    need = ['splat.ply', 'baseline.ply', 'compare.jpg', 'metrics.json', 'trajectory.json', 'index.html', 'points.ply', 'cameras.json',
            'scene.json', 'photos/00.jpg']
    missing = [n for n in need if not (enhance/n).exists()]
    if missing: raise SystemExit(f'selftest thiếu {missing}')
    print(f'SELFTEST OK · {state["package"]["artifixer3d_ply"]}', flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check').set_defaults(func=check)
    p = sub.add_parser('tidy', help='cắt gọn splat.ply/baseline.ply trong thư mục enhance đã tải về')
    p.set_defaults(func=lambda a: tidy_dir(a.dir, keep_full=True, log=Log()))
    p.add_argument('--dir', required=True, type=Path)
    p = sub.add_parser('setup'); p.set_defaults(func=setup)
    p.add_argument('--skip-artifixer', action='store_true')
    p.add_argument('--torch-version', default='2.11.0')
    p.add_argument('--torch-index', default='https://download.pytorch.org/whl/cu128')
    p.add_argument('--fa4', action='store_true', help='cài flash-attn-4 trên Hopper (tuỳ chọn)')
    for name, func in (('run', run), ('selftest', selftest)):
        p = sub.add_parser(name); p.set_defaults(func=func)
        p.add_argument('--source', default='courtyard', choices=['courtyard', 'video', 'images', 'synthetic'])
        p.add_argument('--input', help='video/thư mục ảnh (đường dẫn Colab/Drive hoặc URL)')
        p.add_argument('--name', default=None)
        p.add_argument('--stages', default=','.join(STAGES))
        p.add_argument('--target-frames', dest='target_frames', type=int, default=32)
        p.add_argument('--max-frames', dest='max_frames', type=int, default=128)
        p.add_argument('--fps', type=float, default=2.0)
        p.add_argument('--confidence', type=float, default=.5)
        p.add_argument('--max-points', dest='max_points', type=int, default=200000)
        p.add_argument('--train-image-max', dest='train_image_max', type=int, default=1536)
        p.add_argument('--vggt-dtype', dest='vggt_dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
        p.add_argument('--artifixer-image-max', dest='artifixer_image_max', type=int, default=960)
        p.add_argument('--reconstruction-steps', dest='reconstruction_steps', type=int, default=10000)
        p.add_argument('--artifixer3d-steps', dest='artifixer3d_steps', type=int, default=30000)
        p.add_argument('--trajectory', default='coverage', choices=['coverage', 'path'],
                       help='coverage: đi qua các điểm + xoay 360° nhìn cả sàn; path: chỉ đi qua các điểm đã quay (cũ)')
        p.add_argument('--coverage-spins', dest='coverage_spins', type=int, default=4)
        p.add_argument('--trajectory-frames', dest='trajectory_frames', type=int, default=321)
        p.add_argument('--trajectory-lateral', dest='trajectory_lateral', type=float, default=.04)
        p.add_argument('--caption', default=DEFAULT_CAPTION)
        p.add_argument('--metric-scale', dest='metric_scale', type=float, default=None)
        p.add_argument('--low-memory', dest='low_memory', action='store_true')
        p.add_argument('--pose-ba', dest='pose_ba', action='store_true', help='tinh chỉnh pose bằng BA của VGGT (demo_colmap)')
        p.add_argument('--fake', action='store_true')
        p.add_argument('--force', action='store_true', help='xoá kết quả cũ của scene này')
    return parser


def main(argv=None):
    sys.path.insert(0, str(REPO))
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()

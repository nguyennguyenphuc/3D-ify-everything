import hashlib
import json
import shutil
import subprocess
import math
import re
from pathlib import Path
from PIL import Image, ImageOps
from .config import DATA, AppError, inside

EXTENSIONS = {'.jpg','.jpeg','.png','.webp','.tif','.tiff'}


def import_images(source, project, limit=500):
    destination = project/'originals'
    destination.mkdir(parents=True,exist_ok=True)
    thumbs = project/'thumbnails'
    thumbs.mkdir(exist_ok=True)
    manifest = []
    for path in sorted(source.rglob('*')):
        if path.suffix.lower() not in EXTENSIONS or not path.is_file(): continue
        if not path.resolve().is_relative_to(source.resolve()): continue
        if len(manifest) >= limit: break
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert('RGB')
        idx = f'{len(manifest):04}'
        name = idx + path.suffix.lower()
        shutil.copy2(path,destination/name)
        image.thumbnail((320,240))
        image.save(thumbs/f'{idx}.jpg',quality=85)
        manifest.append({'id':idx,'name':str(path.relative_to(source)), 'file':f'originals/{name}','thumbnail':f'thumbnails/{idx}.jpg','sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if not manifest: raise AppError('Thư mục không có ảnh hỗ trợ')
    (project/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest


def courtyard(project, emit):
    import py7zr
    downloads = DATA/'downloads'; downloads.mkdir(parents=True,exist_ok=True)
    archive = downloads/'courtyard_dslr_undistorted.7z'
    extracted = DATA/'datasets'/'eth3d'
    marker = extracted/'.complete'
    if not marker.exists():
        emit('Tải ETH3D courtyard undistorted (478 MB)',stage='download',progress=.05)
        if not archive.exists() or archive.stat().st_size != 500990569:
            subprocess.run(['curl','-fL','--retry','5','--retry-all-errors','--connect-timeout','20','--speed-time','30','--speed-limit','1024','-C','-','-o',str(archive),'https://www.eth3d.net/data/courtyard_dslr_undistorted.7z'],check=True)
        with py7zr.SevenZipFile(archive) as z:
            for entry in z.list():
                inside(extracted,entry.filename)
                if getattr(entry,'is_symlink',False): raise AppError('Archive chứa symlink')
            z.extractall(extracted)
        marker.touch()
    # A fixed manifest selected using shared sparse track IDs; never used as VGGT geometry.
    candidates = list(extracted.rglob('images.txt'))
    image_root = next((p for p in extracted.rglob('images') if p.is_dir()),None)
    if not candidates or image_root is None: raise AppError('ETH3D thiếu calibration/images để chọn overlap')
    rows = []
    with candidates[0].open() as f:
        while line := f.readline():
            if line.startswith('#') or not line.strip(): continue
            fields = line.split()
            observations = f.readline().split()
            tracks = {int(p) for p in observations[2::3] if p != '-1'}
            rows.append({'name':fields[9], 'tracks':tracks})
    if len(rows) < 8: raise AppError('Calibration không đủ 8 ảnh')
    anchor = max(range(len(rows)),key=lambda i:sum(len(rows[i]['tracks'] & row['tracks']) for row in rows))
    others = sorted((i for i in range(len(rows)) if i != anchor),key=lambda i:(-len(rows[anchor]['tracks'] & rows[i]['tracks']),rows[i]['name']))[:7]
    chosen = [anchor]+others
    overlaps = [len(rows[anchor]['tracks'] & rows[i]['tracks']) for i in chosen]
    if min(overlaps) <= 0: raise AppError('Không xác minh được overlap của 8 ảnh')
    selected = project/'eth3d-selected'; selected.mkdir(exist_ok=True)
    record = []
    for i,n in enumerate(chosen):
        src = inside(image_root,rows[n]['name'])
        shutil.copy2(src,selected/f'{i:02}_{src.name}')
        record.append({'name':rows[n]['name'],'shared_tracks_with_anchor':overlaps[i],'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
    (project/'courtyard-selection.json').write_text(json.dumps({'source':'https://www.eth3d.net/data/courtyard_dslr_undistorted.7z','method':'max shared sparse tracks; calibration used only for selection','images':record},indent=2))
    emit('Đã chọn 8 ảnh có overlap theo sparse tracks ETH3D',stage='import',progress=.8)
    return import_images(selected,project)


def extract_video(video, project, fps, emit, target_frames=None):
    probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=duration:format=duration','-of','json',str(video)],
        capture_output=True,text=True,check=True,timeout=30)
    info = json.loads(probe.stdout)
    duration = float(info.get('streams',[{}])[0].get('duration',info.get('format',{}).get('duration',0)))
    if not math.isfinite(duration) or duration <= 0: raise AppError('Không đọc được thời lượng video')
    # One frame per second is sufficient for long room walks. Short clips need
    # a denser candidate pool so the selector can still choose 32 distinct,
    # sharp viewpoints rather than training every available frame.
    desired_rate = fps if target_frames is None else max(fps, (target_frames * 4) / duration)
    count = min(1200,max(1,math.ceil(duration*desired_rate)))
    # Sample the middle of equal temporal bins. fps output timestamps map to
    # these bins even for variable-rate inputs. These are approximate seek times.
    rate = count/duration
    import tempfile
    with tempfile.TemporaryDirectory(prefix='frames-',dir=project) as directory:
        frames = Path(directory)
        emit(f'Trích tối đa {count} ảnh trên toàn bộ {duration:.1f} giây',stage='frames',progress=.05)
        log_path = project/'extraction.log'
        with log_path.open('w') as log:
            subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','info','-i',str(video),
                '-vf',f'fps={rate}:round=near,showinfo,scale=1920:1920:force_original_aspect_ratio=decrease',
                '-frames:v',str(count),str(frames/'%06d.jpg')],stderr=log,check=True)
        manifest = import_images(frames,project,limit=1200)
    # `showinfo` reports the PTS of the actual decoded source frame chosen by
    # ffmpeg. It is stronger than the nominal sampling-bin midpoint for VFR
    # phone video and lets the viewer seek back to the source timestamp.
    timestamps = [float(match.group(1)) for match in re.finditer(r'pts_time:([\d.\-]+)', log_path.read_text(errors='replace'))]
    for i,item in enumerate(manifest):
        item['timestamp_seconds'] = timestamps[i] if i < len(timestamps) else min(duration,(i+.5)/rate)
    (project/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (project/'video.json').write_text(json.dumps({'video':video.name,'duration_seconds':duration,
        'requested_fps':fps,'effective_fps':rate,'candidate_count':len(manifest),
        'timestamp_kind':'decoded source PTS when available; sampling-bin midpoint fallback, relative to video start'},indent=2))
    return manifest

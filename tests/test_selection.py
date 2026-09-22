import json
import subprocess
import cv2
import numpy as np
from PIL import Image
from studio.selection import Selector, select_frames
from studio.sources import import_images, extract_video
from studio.config import AppError
import pytest


def fixture_images(tmp_path, scene_cut=False):
    source=tmp_path/'input';source.mkdir()
    rng=np.random.default_rng(4)
    texture=rng.integers(0,256,(360,900),dtype=np.uint8)
    for i in range(12):
        image=texture[:,i*25:i*25+480]
        if scene_cut and i>=6:
            image=rng.integers(0,256,image.shape,dtype=np.uint8)
        Image.fromarray(image).save(source/f'{i:02}.png')
    project=tmp_path/'project';project.mkdir()
    manifest=import_images(source,project)
    for i,m in enumerate(manifest): m['timestamp_seconds']=i*10
    return project,manifest


def test_selection_covers_end_and_detects_shared_features(tmp_path):
    project,manifest=fixture_images(tmp_path)
    result=Selector(project,manifest,lambda *a,**k: None).select(4,64)
    assert result['connected']
    assert result['images'][-1]['timestamp_seconds']>=90
    assert len(result['images'])>=4


def test_blur_and_duplicates_are_rejected(tmp_path):
    project,manifest=fixture_images(tmp_path)
    first=cv2.imread(str(project/manifest[0]['file']))
    cv2.imwrite(str(project/manifest[1]['file']),first)
    cv2.imwrite(str(project/manifest[2]['file']),cv2.GaussianBlur(first,(51,51),20))
    selector=Selector(project,manifest,lambda *a,**k: None)
    reasons={r['id']:r['reason'] for r in selector.rejected}
    assert reasons['0001']=='duplicate'
    assert reasons['0002']=='blur'


def test_scene_cut_saves_diagnostic_manifest(tmp_path):
    project,manifest=fixture_images(tmp_path,scene_cut=True)
    run=tmp_path/'run';run.mkdir()
    with pytest.raises(AppError,match='Thiếu liên kết'):
        select_frames(project,manifest,run,lambda *a,**k:None,target=4)
    result=json.loads((run/'selection.json').read_text())
    assert result['gaps']
    assert not result['connected']
    assert result['attempts'][-1]['limit']==64


def test_long_video_samples_near_end(tmp_path):
    video=tmp_path/'long.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
        'testsrc2=size=32x32:rate=2:duration=611','-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
    project=tmp_path/'project';project.mkdir()
    manifest=extract_video(video,project,1,lambda *a,**k:None)
    assert 600<len(manifest)<=1200
    assert manifest[-1]['timestamp_seconds']>=610
    assert (project/manifest[-1]['file']).exists()

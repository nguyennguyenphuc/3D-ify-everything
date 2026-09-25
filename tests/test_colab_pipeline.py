import json
import struct
import numpy as np
import pytest
from colab import pipeline
from studio.geometry import camera_to_viewer


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setenv('COURTYARD_WORK', str(tmp_path/'work'))
    monkeypatch.setenv('COURTYARD_CACHE', str(tmp_path/'cache'))
    monkeypatch.delenv('COLAB_PUBLISH_DIR', raising=False)
    monkeypatch.delenv('COLAB_RELEASE_DIR', raising=False)
    return tmp_path


def args(**kw):
    ns = pipeline.build_parser().parse_args(['selftest'])
    for k, v in kw.items(): setattr(ns, k, v)
    return ns


def test_selftest_builds_every_output(work, monkeypatch):
    pub, rel = work/'pub', work/'rel'
    monkeypatch.setenv('COLAB_PUBLISH_DIR', str(pub)); monkeypatch.setenv('COLAB_RELEASE_DIR', str(rel))
    pipeline.selftest(args(name='t1'))
    enhance = work/'work'/'t1'/'enhance'
    metrics = json.loads((enhance/'metrics.json').read_text())
    assert metrics['artifixer3d_ply']['dropped'] == 1 and metrics['artifixer3d_ply']['sh_rest'] == 45
    assert (rel/'splat.ply').exists() and (rel/'results.zip').exists()
    assert (pub/'compare.jpg').exists() and (pub/'metrics.json').exists()
    assert len(list((enhance/'fixed_frames').glob('*.png'))) == 81


def test_rerun_skips_finished_stages(work, capsys):
    ns = args(name='t2'); ns.force = False
    pipeline.selftest(ns)
    ns = args(name='t2'); ns.fake, ns.source, ns.force = True, 'synthetic', False
    pipeline.run(ns)
    assert capsys.readouterr().out.count('Bỏ qua') == 3


def synthetic_cameras(work):
    layout = pipeline.layout_for('cams')
    pipeline.synthetic_scene(layout, pipeline.Log())
    return layout, pipeline.load_cameras(layout['run'])


def test_trajectory_passes_through_source_cameras(work):
    layout, cameras = synthetic_cameras(work)
    K = pipeline.shared_intrinsics(cameras, (320, 240))
    traj = pipeline.make_trajectory(cameras, K, (320, 240), frames=81, lateral=0.0)
    assert len(traj['frames']) == 81 and traj['w'] == 320 and traj['camera_model'] == 'OPENCV'
    for f in traj['frames']:
        R = np.asarray(f['transform_matrix'])[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    # 6 cameras over 81 frames: knots every 16 frames equal the source OpenGL c2w.
    for knot, cam in zip(range(0, 81, 16), cameras):
        assert np.allclose(traj['frames'][knot]['transform_matrix'], camera_to_viewer(cam['w2c']), atol=1e-6)
    with pytest.raises(ValueError):
        pipeline.make_trajectory(cameras, K, (320, 240), frames=80)


def test_artifixer_scene_has_one_shared_calibration(work):
    layout, cameras = synthetic_cameras(work)
    out = work/'afscene'
    info = pipeline.build_artifixer_scene(layout['run'], out, 200, pipeline.Log())
    assert info['size'] == [192, 144]
    data = (out/'sparse'/'0'/'cameras.bin').read_bytes()
    count = struct.unpack_from('<Q', data)[0]
    params = {data[8 + i*56 + 24: 8 + (i+1)*56] for i in range(count)}
    assert count == 6 and len(params) == 1
    from PIL import Image
    assert all(Image.open(p).size == (192, 144) for p in (out/'images').glob('*.png'))


def test_convert_truncates_sh_and_drops_bad_rows(work):
    src, dst = work/'in.ply', work/'out.ply'
    pipeline.synthetic_3dgrut_ply(src, n=50, sh_rest=12)
    info = pipeline.convert_3dgrut_ply(src, dst)
    from plyfile import PlyData
    names = PlyData.read(str(dst))['vertex'].data.dtype.names
    assert info['sh_rest'] == 9 and 'nx' not in names and info['splats'] + info['dropped'] == 50
    assert {'f_rest_0', 'f_rest_8'} <= set(names) and 'f_rest_9' not in names


def compact_ply(path, xyz, opacity, log_scale):
    from plyfile import PlyData, PlyElement
    names = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    arr = np.zeros(len(xyz), dtype=[(n, 'f4') for n in names])
    arr['x'], arr['y'], arr['z'] = xyz.T
    arr['opacity'], arr['rot_0'] = opacity, 1
    for i in range(3): arr[f'scale_{i}'] = log_scale
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(str(path))


def test_tidy_crops_to_captured_region(work):
    layout, cameras = synthetic_cameras(work)
    points, _ = pipeline.read_ply_xyzrgb(layout['run']/'geometry'/'points.ply')
    box = pipeline.scene_box(cameras, points)
    assert np.allclose(box['up'], [0, -1, 0], atol=1e-6)
    rng = np.random.default_rng(1)
    inside = points[rng.choice(len(points), 2000, replace=False)]
    xyz = np.vstack([inside, [[40, 0, 4]], [[0, 0, 4]], [[1, 0, 4]], [[-1, 0, 4]]])
    opacity = np.r_[np.full(2000, 2.0), 2.0, 2.0, -6.0, 2.0]
    log_scale = np.r_[np.full(2000, -5.0), -5.0, -5.0, -5.0, np.log(10.0)]
    src, dst = work/'full.ply', work/'tidy.ply'
    compact_ply(src, xyz, opacity, log_scale)
    info = pipeline.tidy_splat(src, dst, box)
    assert (info['outside_box'], info['transparent'], info['oversized']) == (1, 1, 1)
    assert info['after'] >= 1990 and info['after'] == info['before'] - 3 - info['isolated']
    info = pipeline.tidy_splat(dst, dst, box)  # in place is safe
    assert info['before'] == info['after'] + info['isolated']


def test_single_stage_runs_reuse_finished_stages(work):
    for stage in ('inputs', 'vggt', 'artifixer', 'package'):
        ns = args(name='t3', stages=stage); ns.fake, ns.source, ns.force = True, 'synthetic', False
        pipeline.run(ns)
    enhance = work/'work'/'t3'/'enhance'
    assert len(list((enhance/'fixed_frames').glob('*.png'))) == 81 and (enhance/'compare.jpg').exists()
    assert (enhance/'scene.json').exists()
    ns = args(name='t4', stages='vggt'); ns.fake, ns.source, ns.force = False, 'video', False
    with pytest.raises(SystemExit):
        pipeline.run(ns)

import json
import numpy as np
from PIL import Image
from studio import vggt


def test_device_generic_reconstruct_writes_colmap(tmp_path, monkeypatch):
    paths = []
    rng = np.random.default_rng(1)
    for i in range(3):
        p = tmp_path/f'src{i}.png'
        Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    monkeypatch.setattr(vggt, 'load_model', lambda device, dtype, weights=None: object())
    def fake_run(model, arrays, device, dtype):
        n = len(arrays)
        K = np.array([[400., 0, 258.5], [0, 400., 258.5], [0, 0, 1]])
        E = np.column_stack([np.eye(3), np.zeros(3)])
        return np.full((n, 518, 518), 2.0), np.ones((n, 518, 518)), np.stack([E]*n), np.stack([K]*n)
    monkeypatch.setattr(vggt, 'run_model', fake_run)
    run = tmp_path/'run'
    stats = vggt.reconstruct(paths, run, lambda *a, **k: None, max_points=3000, device='cpu', dtype='float32', train_max=256)
    assert stats['camera_count'] == 3 and stats['backend'] == 'cpu'
    assert all((run/'sparse'/'0'/n).exists() for n in ('cameras.bin', 'images.bin', 'points3D.bin'))
    cams = json.loads((run/'geometry'/'cameras.json').read_text())['cameras']
    assert cams[0]['wh'] == [256, 192] and Image.open(run/'images'/'00.png').size == (256, 192)

"""Run VGGT's official demo_colmap.py with the cached VGGT-1B weights (cwd = vendor/vggt, Python 3.11 BA env).

    python colab/vggt_ba.py --weights model.pt -- --scene_dir DIR --use_ba --shared_camera --camera_type PINHOLE

demo_colmap.py downloads model.pt from Hugging Face through torch.hub on every run; point that call at
the copy colab.pipeline already cached on Drive instead.
"""
import argparse
import runpy
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    args, rest = p.parse_known_args()
    rest = rest[1:] if rest[:1] == ['--'] else rest
    import torch
    torch.hub.load_state_dict_from_url = lambda *a, **kw: torch.load(args.weights, map_location='cpu', weights_only=True)
    sys.argv = ['demo_colmap.py', *rest]
    runpy.run_path('demo_colmap.py', run_name='__main__')


if __name__ == '__main__':
    main()

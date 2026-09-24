"""Clone pinned upstream sources into vendor/. The Mac app needs vggt + OpenSplat only.

    python scripts/fetch_upstream.py                 # vggt, OpenSplat (Mac setup)
    python scripts/fetch_upstream.py --only vggt     # e.g. Colab
"""
import argparse
import json
import subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
URLS = {'vggt': 'https://github.com/facebookresearch/vggt.git', 'OpenSplat': 'https://github.com/WebODM/OpenSplat.git',
        'ArtiFixer': 'https://github.com/nv-tlabs/ArtiFixer.git'}
DEFAULT = ('vggt', 'OpenSplat')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+', choices=sorted(URLS), default=list(DEFAULT))
    args = parser.parse_args()
    lock = json.loads((root/'upstream.lock.json').read_text())
    for name in args.only:
        rev = lock[name]
        path = root/'vendor'/name
        if not path.exists(): subprocess.run(['git', 'clone', URLS[name], str(path)], check=True)
        current = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if current != rev:
            subprocess.run(['git', '-C', str(path), 'fetch', 'origin', rev], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', rev], check=True)
        if name == 'ArtiFixer':
            subprocess.run(['git', '-C', str(path), 'submodule', 'update', '--init', '--recursive'], check=True)


if __name__ == '__main__':
    main()

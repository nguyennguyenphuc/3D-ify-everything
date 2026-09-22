"""Use one OpenMP runtime for pip LibTorch and Homebrew OpenCV/OpenBLAS on macOS."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
torch_omp = root / '.venv/lib/python3.11/site-packages/torch/lib/libomp.dylib'
original = torch_omp.with_name('libomp.torch-original.dylib')
homebrew_omp = Path('/opt/homebrew/opt/libomp/lib/libomp.dylib')

if not homebrew_omp.is_file():
    raise SystemExit(f'Missing Homebrew OpenMP runtime: {homebrew_omp}')
if torch_omp.is_symlink() and torch_omp.resolve() == homebrew_omp.resolve():
    print('LibTorch already uses Homebrew libomp')
elif torch_omp.is_file():
    if not original.exists():
        torch_omp.rename(original)
    else:
        torch_omp.unlink()
    torch_omp.symlink_to(homebrew_omp)
    print('Linked LibTorch libomp to Homebrew libomp')
else:
    raise SystemExit(f'Missing LibTorch OpenMP runtime: {torch_omp}')

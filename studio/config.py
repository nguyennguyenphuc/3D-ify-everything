import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("STUDIO_DATA", ROOT / "data")).resolve()
VENDOR = ROOT / "vendor"
MODEL = DATA / "models" / "model.pt"
OPENSPLAT = VENDOR / "OpenSplat" / "build" / "opensplat"
ALLOWED_ROOTS = [Path(p).expanduser().resolve() for p in os.environ.get("STUDIO_IMPORT_ROOTS", str(ROOT)).split(os.pathsep)]


class AppError(Exception):
    def __init__(self, message: str, status: int = 400):
        self.message, self.status = message, status


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise AppError("Đường dẫn nằm ngoài thư mục cho phép", 403)
    return path


def import_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not any(path.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise AppError("Thư mục phải nằm trong STUDIO_IMPORT_ROOTS", 403)
    if not path.is_dir():
        raise AppError("Không tìm thấy thư mục ảnh")
    return path

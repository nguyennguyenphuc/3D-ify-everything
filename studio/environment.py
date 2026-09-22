import json
import os
import platform
import subprocess
from .config import MODEL, OPENSPLAT, VENDOR


def command(args):
    try:
        env = dict(os.environ)
        xcode = "/Applications/Xcode.app/Contents/Developer"
        if os.path.isdir(xcode): env.setdefault("DEVELOPER_DIR", xcode)
        p = subprocess.run(args, capture_output=True, text=True, timeout=20, env=env)
        return {"ok": p.returncode == 0, "detail": (p.stdout + p.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "detail": str(e)}


def inspect(probe=False):
    import torch
    result = {
        "platform": platform.platform(), "architecture": platform.machine(), "torch": torch.__version__,
        "mps": {"ok": torch.backends.mps.is_available(), "detail": "MPS availability"},
        "metal": command(["xcrun", "-sdk", "macosx", "metal", "--version"]),
        "ffmpeg": command(["ffmpeg", "-version"]),
        "model": {"ok": MODEL.is_file(), "detail": str(MODEL)},
        "opensplat": {"ok": False, "detail": "Chưa có bản build Metal đã xác minh"},
        "fallback_disabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1",
    }
    if probe and result["mps"]["ok"]:
        try:
            x = torch.arange(16, device="mps", dtype=torch.float32).reshape(4, 4)
            y = x @ x.T
            torch.mps.synchronize()
            assert y.device.type == "mps" and y[0, 0].item() == 14
            result["mps"] = {"ok": True, "detail": "Phép nhân ma trận MPS đúng; tensor ở mps:0"}
        except Exception as e:
            result["mps"] = {"ok": False, "detail": str(e)}
    receipt = OPENSPLAT.parent / "metal-build.json"
    if OPENSPLAT.is_file() and receipt.exists():
        import hashlib
        meta = json.loads(receipt.read_text())
        valid = meta.get("sha256") == hashlib.sha256(OPENSPLAT.read_bytes()).hexdigest()
        result["opensplat"] = {"ok": valid, "detail": "Metal build fingerprint" if valid else "Binary đã thay đổi"}
    result["ready"] = all(result[k]["ok"] for k in ("mps", "metal", "model", "opensplat", "ffmpeg")) and result["fallback_disabled"]
    return result


if __name__ == "__main__":
    print(json.dumps(inspect(probe=True), ensure_ascii=False, indent=2))

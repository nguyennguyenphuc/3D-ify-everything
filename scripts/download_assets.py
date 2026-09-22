"""Download official inputs outside the timed GPU run. Re-running resumes downloads."""
import hashlib
import json
import subprocess
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio.config import DATA, MODEL, VENDOR


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    from huggingface_hub import HfApi, hf_hub_download
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    revision_file = MODEL.parent / "revision.json"
    if revision_file.exists():
        locked = json.loads(revision_file.read_text())
        revision = locked["revision"]
        expected_hash = locked.get("model_pt_sha256")
        if MODEL.is_file() and expected_hash:
            actual_hash = sha256(MODEL)
            if actual_hash == expected_hash:
                print(f"Model already verified: {MODEL}", flush=True)
                return
            raise RuntimeError("Existing model.pt does not match locked SHA-256")
    else:
        revision = HfApi().model_info("facebook/VGGT-1B").sha
        revision_file.write_text(json.dumps({"repo": "facebook/VGGT-1B", "revision": revision}, indent=2))
    path = hf_hub_download("facebook/VGGT-1B", "model.pt", revision=revision, local_dir=MODEL.parent)
    if Path(path) != MODEL: Path(path).replace(MODEL)
    print(f"Model: {MODEL}", flush=True)


if __name__ == "__main__":
    main()

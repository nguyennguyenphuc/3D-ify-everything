"""Run an end-to-end local room job without an HTTP server.

This is useful for acceptance testing a video already on disk. The normal user
path remains the upload endpoint.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studio import db
from studio.config import DATA
from studio.worker import main


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--name", default="Room acceptance smoke")
    parser.add_argument("--target-frames", type=int, default=32, choices=range(2, 65))
    args = parser.parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Missing video: {args.video}")
    db.migrate()
    project = db.create_project(args.name)
    destination = DATA / "projects" / project["id"] / f"video{args.video.suffix.lower()}"
    shutil.copy2(args.video, destination)
    job = db.new_job(project["id"], "room", {
        "video": destination.name,
        "fps": 1,
        "selection_mode": "auto",
        "target_frames": args.target_frames,
        "confidence": 0.5,
        "iterations": 10000,
        "budget_minutes": 30,
        "uploaded_at": time.time(),
    })
    print(job["id"], flush=True)
    main(job["id"])
    print(db.job(job["id"]), flush=True)


if __name__ == "__main__":
    run()

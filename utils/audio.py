from pathlib import Path
import shutil
import time


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def cleanup_dir(p: Path, ttl_seconds: int = 3600):
    if not p.exists():
        return
    now = time.time()
    for f in p.glob("*"):
        try:
            if f.is_file() and now - f.stat().st_mtime > ttl_seconds:
                f.unlink()
        except Exception:
            pass

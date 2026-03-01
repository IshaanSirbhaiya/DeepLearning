"""
download_model.py — SafeEdge
Downloads the best available pre-trained YOLOv8n fire/smoke weights.

Run once before first use:
    python detection/models/download_model.py

Priority order:
  1. keremberke/yolov8n-fire-smoke-detection  (HuggingFace) — BEST
  2. Direct GitHub release fallback
  3. Base YOLOv8n notice (user must supply own weights)
"""

import sys
import hashlib
import os
from pathlib import Path

MODELS_DIR = Path(__file__).parent
MODEL_PATH = MODELS_DIR / "fire_smoke.pt"

# Known good HuggingFace model
HF_REPO     = "keremberke/yolov8n-fire-smoke-detection"
HF_FILENAME = "best.pt"

# Fallback: another community fire model
ALT_REPOS = [
    ("StanislawStankiewicz/fire-detection-yolov8", "fire-detection.pt"),
    ("arnabdhar/YOLOv8-Fire-and-Smoke",            "best.pt"),
]


def download_from_huggingface(repo_id: str, filename: str, dest: Path) -> bool:
    try:
        print(f"  Trying HuggingFace: {repo_id}/{filename} ...")
        from huggingface_hub import hf_hub_download
        tmp = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
        )
        Path(tmp).rename(dest)
        print(f"  ✓ Downloaded to {dest}  ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except ImportError:
        print("  ✗ huggingface_hub not installed. Run: pip install huggingface-hub")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    return False


def verify_model(path: Path) -> bool:
    """Basic sanity check — file exists and is >1MB (valid .pt)."""
    return path.exists() and path.stat().st_size > 1_000_000


def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          MANUAL DOWNLOAD INSTRUCTIONS                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Option A — HuggingFace (recommended):                       ║
║    1. Go to huggingface.co/keremberke/yolov8n-fire-smoke-... ║
║    2. Download best.pt                                       ║
║    3. Save as:  detection/models/fire_smoke.pt               ║
║                                                              ║
║  Option B — Kaggle:                                          ║
║    Search: "yolov8 fire smoke detection weights"             ║
║    Pick highest-voted dataset with best.pt                   ║
║    Save as:  detection/models/fire_smoke.pt                  ║
║                                                              ║
║  Option C — Fine-tune (1-2 hours, highest accuracy):         ║
║    Dataset: kaggle.com/datasets/phylake1337/fire-dataset     ║
║    Run:  yolo train model=yolov8n.pt data=fire.yaml          ║
║           epochs=50 imgsz=640                                ║
║    Copy runs/detect/train/weights/best.pt                    ║
║         → detection/models/fire_smoke.pt                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    print("═" * 60)
    print("SafeEdge — Model Downloader")
    print("═" * 60)

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        print(f"\n✓ Model already exists: {MODEL_PATH}  ({size_mb:.1f} MB)")
        print("  Delete it to re-download.\n")
        return

    print(f"\nTarget path: {MODEL_PATH}")
    print("Attempting automatic download...\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Try primary source
    if download_from_huggingface(HF_REPO, HF_FILENAME, MODEL_PATH):
        if verify_model(MODEL_PATH):
            print(f"\n✓ Model ready at {MODEL_PATH}")
            _print_usage()
            return
        else:
            print("  Downloaded file seems invalid (too small). Trying alternatives...")
            MODEL_PATH.unlink(missing_ok=True)

    # Try alternatives
    for repo, fname in ALT_REPOS:
        if download_from_huggingface(repo, fname, MODEL_PATH):
            if verify_model(MODEL_PATH):
                print(f"\n✓ Model ready at {MODEL_PATH}")
                _print_usage()
                return
            MODEL_PATH.unlink(missing_ok=True)

    # All failed
    print("\n✗ Automatic download failed.")
    print_manual_instructions()
    sys.exit(1)


def _print_usage():
    print("""
Usage after download:
  python detection/detector.py --input testbench/sample_fire.mp4
  python detection/detector.py --input 0  (webcam)
  python detection/detector.py --input rtsp://... --serve
""")


if __name__ == "__main__":
    main()

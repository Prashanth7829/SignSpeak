from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
PILOT = Path(os.environ.get("SIGNSPEAK_DATASET", ROOT.parent / "dataset")) / "include50_pilot"
CHECKPOINT = ROOT / "artifacts" / "recognizer.pt"
IMAGE_TYPES = ("jpg", "jpeg", "png", "webp")
VIDEO_TYPES = ("mp4", "mov", "avi", "mkv", "webm")


DATA = PILOT.parent
CACHE = ROOT / "artifacts" / "corpus"
CATALOG = CACHE / "catalog.json"
FULL_MODEL = CACHE / "include.pt"

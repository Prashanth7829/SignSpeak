import argparse
import platform
import struct
import sys
from importlib.metadata import version

import torch

from signspeak.settings import DATA as DATASET_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Verify the SignSpeak environment before training."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run the calculation on CPU for fallback testing.",
    )
    args = parser.parse_args()

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"Architecture: {struct.calcsize('P') * 8}-bit")
    print(f"Dataset directory: {DATASET_DIR}")

    if not DATASET_DIR.is_dir():
        raise SystemExit(f"Dataset directory missing: {DATASET_DIR}")

    # Check one known INCLUDE archive without ingesting unrelated datasets.
    archive = DATASET_DIR / "Seasons_1of1.zip"
    if not archive.is_file():
        raise SystemExit(f"Expected INCLUDE archive missing: {archive}")
    print(f"Located INCLUDE archive: {archive.name}")

    for package in ("torch", "torchvision", "mediapipe", "numpy", "streamlit"):
        print(f"{package}: {version(package)}")

    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = torch.cuda.get_device_properties(device)
        print(f"GPU: {gpu.name}")
        print(f"GPU memory: {gpu.total_memory / 1024**3:.2f} GiB")
    else:
        raise SystemExit(
            "CUDA is unavailable in this interpreter. "
            "Use --cpu to verify the CPU fallback."
        )

    # Exercise an actual compute kernel and copy the result back for checking.
    with torch.inference_mode():
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        result = (values @ values).cpu()
        expected = torch.tensor([[7.0, 10.0], [15.0, 22.0]])
        torch.testing.assert_close(result, expected)

    print(f"Verified matrix multiplication on {device.type}.")
    print("Phase 1 environment check passed.")


if __name__ == "__main__":
    main()

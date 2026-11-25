import json
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import pandas as pd


METADATA_API = (
    "https://datasets-server.huggingface.co/parquet?dataset=ai4bharat%2FINCLUDE"
)
METADATA_DIR = Path(__file__).resolve().parent / "include_metadata"


def main():
    METADATA_DIR.mkdir(exist_ok=True)

    with urlopen(METADATA_API, timeout=30) as response:
        files = json.load(response)["parquet_files"]

    frames = []
    for entry in files:
        if entry["config"] != "default":
            continue

        split = entry["split"]
        destination = METADATA_DIR / f"{split}-{entry['filename']}"
        if not destination.exists():
            temporary_file = destination.with_suffix(".part")
            urlretrieve(entry["url"], temporary_file)
            temporary_file.replace(destination)

        frame = pd.read_parquet(destination)
        subset = frame.loc[
            frame["include_50"].eq(True), ["parent_label", "label", "video_path"]
        ].copy()
        subset["split"] = split
        frames.append(subset)

    if not frames:
        raise RuntimeError("No Parquet files found for the default INCLUDE dataset.")

    include50 = pd.concat(frames, ignore_index=True)
    labels = sorted(include50["label"].dropna().unique(), key=str.casefold)

    print(f"Metadata rows in INCLUDE-50: {len(include50)}")
    print(f"Unique video paths: {include50['video_path'].nunique()}")
    print(f"Classes: {len(labels)}")

    print("\nRows by published split:")
    print(include50.groupby("split").size().to_string())

    print("\nRows by category:")
    print(include50.groupby("parent_label").size().to_string())

    print("\nClass labels:")
    print(", ".join(labels))

    print("\nExample video paths:")
    print("\n".join(include50["video_path"].dropna().head(5)))


if __name__ == "__main__":
    main()
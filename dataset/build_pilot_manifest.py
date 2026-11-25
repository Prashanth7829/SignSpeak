from pathlib import Path

import pandas as pd


PILOT_DIR = Path(__file__).resolve().parent / "include50_pilot"


def main():
    manifest = pd.concat(
        [
            pd.read_csv(PILOT_DIR / "manifest.csv"),
            pd.read_csv(PILOT_DIR / "greetings_manifest.csv"),
        ],
        ignore_index=True,
    )

    if manifest["video_path"].duplicated().any():
        raise ValueError("A video path appears more than once in the pilot manifests.")

    missing_paths = [
        path for path in manifest["video_path"]
        if not (PILOT_DIR / path).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError("Missing videos:\n" + "\n".join(missing_paths))

    output_path = PILOT_DIR / "pilot_manifest.csv"
    manifest.sort_values(["split", "label", "video_path"]).to_csv(
        output_path, index=False
    )

    print(f"Clips: {len(manifest)}")
    print(f"Classes: {manifest['label'].nunique()}")
    print(manifest.groupby("split")["video_path"].count().to_string())
    print(f"Manifest saved to: {output_path}")


if __name__ == "__main__":
    main()
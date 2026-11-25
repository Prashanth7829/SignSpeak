from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from zipfile import ZipFile

import cv2
import pandas as pd


DATASET_DIR = Path(__file__).resolve().parent
ARCHIVE = DATASET_DIR / "Seasons_1of1.zip"
METADATA_DIR = DATASET_DIR / "include_metadata"
PILOT_DIR = DATASET_DIR / "include50_pilot"


def main():
    frames = []
    for split in ("train", "val", "test"):
        metadata_files = sorted(METADATA_DIR.glob(f"{split}-*.parquet"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata files found for {split}.")

        for path in metadata_files:
            metadata = pd.read_parquet(path)
            selected = metadata["include_50"].eq(True) & metadata[
                "video_path"
            ].str.startswith("Seasons/", na=False)
            frame = metadata.loc[selected, ["video_path"]].copy()
            frame["split"] = split
            frames.append(frame)

    clips = pd.concat(frames, ignore_index=True).drop_duplicates()
    if clips.empty:
        raise ValueError("No INCLUDE-50 clips found under Seasons/.")

    if clips["video_path"].duplicated().any():
        raise ValueError("At least one Seasons clip is assigned to multiple splits.")

    manifest_rows = []
    with ZipFile(ARCHIVE) as archive:
        available_paths = set(archive.namelist())
        missing = sorted(set(clips["video_path"]) - available_paths)
        if missing:
            raise FileNotFoundError("Missing ZIP entries:\n" + "\n".join(missing))

        for clip in clips.itertuples(index=False):
            relative_path = PurePosixPath(clip.video_path)
            if len(relative_path.parts) != 3 or any(
                part in (".", "..") or "\\" in part or ":" in part
                for part in relative_path.parts
            ):
                raise ValueError(f"Unexpected archive path: {clip.video_path}")

            destination = PILOT_DIR.joinpath(*relative_path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(clip.video_path) as source, destination.open("wb") as target:
                copyfileobj(source, target)

            video = cv2.VideoCapture(str(destination))
            try:
                success, frame = video.read()
            finally:
                video.release()

            if not success or frame is None:
                raise RuntimeError(f"OpenCV could not decode {destination}")

            manifest_rows.append(
                {
                    "video_path": relative_path.as_posix(),
                    "label": relative_path.parent.name,
                    "split": clip.split,
                }
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["split", "label", "video_path"]
    )
    manifest_path = PILOT_DIR / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"Clips extracted and decoded: {len(manifest)}")
    print(f"Manifest: {manifest_path}")
    print("\nClip counts by label and split:")
    print(manifest.groupby(["label", "split"]).size().to_string())


if __name__ == "__main__":
    main()
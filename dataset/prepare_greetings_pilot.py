from contextlib import ExitStack
from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile

import cv2
import pandas as pd


DATASET_DIR = Path(__file__).resolve().parent
METADATA_DIR = DATASET_DIR / "include_metadata"
PILOT_DIR = DATASET_DIR / "include50_pilot"


def main():
    frames = []
    for split in ("train", "val", "test"):
        files = sorted(METADATA_DIR.glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No metadata files for the {split} split.")

        for file in files:
            metadata = pd.read_parquet(file)
            selected = metadata["include_50"].eq(True) & metadata[
                "video_path"
            ].str.startswith("Greetings/", na=False)
            frames.append(metadata.loc[selected, ["video_path"]].assign(split=split))

    clips = pd.concat(frames, ignore_index=True).drop_duplicates()
    if clips.empty or clips["video_path"].duplicated().any():
        raise ValueError("Expected Greetings clips with one split per video path.")

    manifest_rows = []
    with ExitStack() as stack:
        archives = [
            stack.enter_context(ZipFile(DATASET_DIR / f"Greetings_{part}of2.zip"))
            for part in (1, 2)
        ]
        entries = [set(archive.namelist()) for archive in archives]
        clip_archives = {}

        for path in clips["video_path"]:
            parts = path.split("/")
            if len(parts) != 3 or any(
                part in ("", ".", "..") or "\\" in part or ":" in part
                for part in parts
            ):
                raise ValueError(f"Unexpected video path: {path}")

            matches = [
                archive
                for archive, names in zip(archives, entries)
                if path in names
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one ZIP match for {path}, found {len(matches)}.")
            clip_archives[path] = matches[0]

        for clip in clips.itertuples(index=False):
            parts = clip.video_path.split("/")
            destination = PILOT_DIR.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)

            with clip_archives[clip.video_path].open(clip.video_path) as source:
                with destination.open("wb") as target:
                    copyfileobj(source, target)

            video = cv2.VideoCapture(str(destination))
            try:
                decoded, frame = video.read()
            finally:
                video.release()

            if not decoded or frame is None:
                raise RuntimeError(f"OpenCV could not decode {destination}")

            manifest_rows.append(
                {"video_path": clip.video_path, "label": parts[1], "split": clip.split}
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(["split", "label", "video_path"])
    manifest_path = PILOT_DIR / "greetings_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Greetings clips extracted and decoded: {len(manifest)}")
    print(f"Manifest: {manifest_path}")
    print(manifest.groupby(["label", "split"]).size().to_string())


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


METADATA_DIR = Path(__file__).resolve().parent / "include_metadata"
SPLITS = ("train", "val", "test")


def main():
    frames = []
    for split in SPLITS:
        files = sorted(METADATA_DIR.glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No metadata files found for the {split} split.")

        for path in files:
            frame = pd.read_parquet(path)
            subset = frame.loc[
                frame["include_50"].eq(True),
                ["video_path", "parent_label", "label"],
            ].copy()
            subset["split"] = split
            frames.append(subset)

    metadata = pd.concat(frames, ignore_index=True)
    missing_paths = metadata["video_path"].isna() | metadata["video_path"].eq("")
    valid_rows = metadata.loc[~missing_paths]

    path_summary = valid_rows.groupby("video_path").agg(
        rows=("split", "size"),
        splits=("split", lambda values: ", ".join(sorted(set(values)))),
        split_count=("split", "nunique"),
        label_count=("label", "nunique"),
        category_count=("parent_label", "nunique"),
    )
    repeated = path_summary.loc[path_summary["rows"].gt(1)]
    cross_split = repeated["split_count"].gt(1)
    conflicts = repeated["label_count"].gt(1) | repeated["category_count"].gt(1)

    print(f"Rows with missing video paths: {missing_paths.sum()}")
    print(f"Video paths listed more than once: {len(repeated)}")
    print(f"Video paths listed across splits: {cross_split.sum()}")
    print(f"Repeated paths with conflicting labels or categories: {conflicts.sum()}")

    if not repeated.empty:
        print("\nRepeated paths:")
        print(repeated.to_string())


if __name__ == "__main__":
    main()
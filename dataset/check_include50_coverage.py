from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd


DATASET_DIR = Path(__file__).resolve().parent
METADATA_DIR = DATASET_DIR / "include_metadata"


def main():
    metadata_files = sorted(METADATA_DIR.glob("*.parquet"))
    if not metadata_files:
        raise FileNotFoundError(f"No Parquet metadata found in {METADATA_DIR}.")

    metadata = pd.concat(
        [pd.read_parquet(path) for path in metadata_files], ignore_index=True
    )
    paths = (
        metadata.loc[metadata["include_50"].eq(True), "video_path"]
        .dropna()
        .drop_duplicates()
    )
    if paths.empty:
        raise ValueError("The metadata contains no INCLUDE-50 video paths.")

    target_paths = set(paths)
    found_paths = set()

    for archive_path in sorted(DATASET_DIR.glob("*.zip")):
        try:
            with ZipFile(archive_path) as archive:
                matches = target_paths.intersection(archive.namelist())
        except (OSError, BadZipFile) as error:
            print(f"Could not read {archive_path.name}: {error}")
            continue

        print(f"{archive_path.name}: {len(matches)} target clips")
        found_paths.update(matches)

    coverage = pd.DataFrame({"video_path": sorted(target_paths)})
    coverage["category"] = coverage["video_path"].str.split("/", n=1).str[0]
    coverage["available"] = coverage["video_path"].isin(found_paths)
    summary = coverage.groupby("category").agg(
        expected=("video_path", "size"),
        found=("available", "sum"),
    )
    summary["missing"] = summary["expected"] - summary["found"]

    print("\nCoverage by category")
    print(summary.to_string())
    print(f"\nTotal target clips found: {len(found_paths)}/{len(target_paths)}")


if __name__ == "__main__":
    main()
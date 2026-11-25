import hashlib
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


DATASET = Path(__file__).resolve().parent
ARCHIVE = DATASET / "raw" / "ISL-CSLTR" / "ISL_CSLRT_Corpus.zip"
ANNOTATIONS = DATASET / "annotations"
MANIFEST = ANNOTATIONS / "csltr_sentence_splits.csv"
OUTPUT = ANNOTATIONS / "csltr_video_hashes.csv"


def main():
    data = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    hashes = []

    with ZipFile(ARCHIVE) as archive:
        for index, path in enumerate(data["video_path_in_zip"], start=1):
            with archive.open(path) as video:
                digest = hashlib.file_digest(video, "sha256").hexdigest()
            hashes.append(digest)

            if index % 50 == 0 or index == len(data):
                print(f"Checked {index}/{len(data)} videos", flush=True)

    data["sha256"] = hashes
    duplicates = data[data["sha256"].duplicated(keep=False)]
    groups = duplicates.groupby("sha256")
    cross_split = groups["split"].nunique()
    conflicting_gloss = groups["gloss_group"].nunique()

    print(f"\nVideos checked: {len(data)}")
    print(f"Distinct video hashes: {data['sha256'].nunique()}")
    print(f"Duplicate groups: {len(cross_split)}")
    print(f"Duplicate groups spanning splits: {(cross_split > 1).sum()}")
    print(f"Duplicate groups with conflicting glosses: {(conflicting_gloss > 1).sum()}")

    if not duplicates.empty:
        print("\nDuplicate entries:")
        for digest, group in groups:
            print(f"\n{digest[:16]}")
            for row in group.itertuples(index=False):
                print(f"  {row.split} | {row.english_sentence} | {row.video_path_in_zip}")

    temporary = OUTPUT.with_suffix(".tmp")
    data.to_csv(temporary, index=False)
    temporary.replace(OUTPUT)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


ANNOTATIONS = Path(__file__).resolve().parent / "annotations"
SOURCE = ANNOTATIONS / "csltr_video_hashes.csv"
CLEAN = ANNOTATIONS / "csltr_clean_manifest.csv"
EXCLUSIONS = ANNOTATIONS / "csltr_duplicate_exclusions.csv"


def save_csv(data, path):
    temporary = path.with_suffix(".tmp")
    data.to_csv(temporary, index=False)
    temporary.replace(path)


def main():
    data = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    if not data["sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Missing or invalid SHA-256 value in the audit file.")

    # Stable ordering makes the retained copy reproducible.
    data = data.sort_values("video_path_in_zip").reset_index(drop=True)
    data["exclusion_reason"] = ""

    repeats = data[data["sha256"].duplicated(keep=False)]
    for _, group in repeats.groupby("sha256"):
        english = group["english_sentence"].str.split().str.join(" ").str.casefold()

        if group["gloss_group"].nunique() > 1 or english.nunique() > 1:
            data.loc[group.index, "exclusion_reason"] = "annotation conflict"
        elif group["split"].nunique() > 1:
            data.loc[group.index, "exclusion_reason"] = "identical footage across splits"
        else:
            data.loc[group.index[1:], "exclusion_reason"] = "redundant exact copy"

    excluded = data.loc[data["exclusion_reason"].ne("")].copy()
    clean = data.loc[data["exclusion_reason"].eq("")].drop(columns="exclusion_reason")

    if clean["sha256"].duplicated().any():
        raise ValueError("Duplicate footage remains in the clean manifest.")
    if clean.groupby("gloss_group")["split"].nunique().gt(1).any():
        raise ValueError("A retained gloss group spans multiple splits.")

    save_csv(clean, CLEAN)
    save_csv(excluded, EXCLUSIONS)

    print(f"Source rows: {len(data)}")
    print(f"Retained unique videos: {len(clean)}")
    print(f"Excluded rows: {len(excluded)}")
    print(excluded["exclusion_reason"].value_counts().to_string())

    print("\nRetained split sizes:")
    summary = clean.groupby("split").agg(
        videos=("sha256", "size"),
        gloss_groups=("gloss_group", "nunique"),
    )
    print(summary.reindex(["train", "val", "test"]).to_string())

    lost_groups = sorted(set(data["gloss_group"]) - set(clean["gloss_group"]))
    print("\nGloss groups with no retained videos:")
    for gloss in lost_groups:
        print(f"  {gloss}")

    print(f"\nClean manifest: {CLEAN}")
    print(f"Exclusion log: {EXCLUSIONS}")


if __name__ == "__main__":
    main()
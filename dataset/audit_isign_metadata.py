from pathlib import Path

import pandas as pd


CSV_PATH = Path(__file__).resolve().parent / "iSign_v1.1.csv"


def main():
    data = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    data["uid"] = data["uid"].str.strip()
    data["text"] = data["text"].str.strip()

    ids = data.loc[data["uid"].ne(""), "uid"]
    duplicate_ids = ids[ids.duplicated(keep=False)].unique()

    parts = data["uid"].str.extract(r"^(?P<prefix>.+)-(?P<sequence>\d+)$")
    matched = parts["prefix"].notna()
    group_sizes = parts.loc[matched, "prefix"].value_counts()

    page_references = data["text"].str.fullmatch(
        r"page\s+\d+[.]?", case=False, na=False
    )
    numeric_only = data["text"].str.fullmatch(r"\d+", na=False)

    print(f"Total rows: {len(data)}")
    print(f"Blank IDs: {data['uid'].eq('').sum()}")
    print(f"Distinct nonblank IDs: {ids.nunique()}")
    print(f"Distinct IDs appearing more than once: {len(duplicate_ids)}")
    print(f"Blank translations: {data['text'].eq('').sum()}")
    print(f"Page-reference translations: {page_references.sum()}")
    print(f"Number-only translations: {numeric_only.sum()}")
    print(f"IDs matching prefix-number format: {matched.sum()}")
    print(f"Distinct matching prefixes: {len(group_sizes)}")
    print(f"Prefixes containing multiple rows: {group_sizes.gt(1).sum()}")

    if matched.any():
        first_prefix = parts.loc[matched, "prefix"].iloc[0]
        group = parts["prefix"].eq(first_prefix)
        preview = data.loc[group, ["uid", "text"]].copy()
        preview["sequence"] = pd.to_numeric(parts.loc[group, "sequence"])
        preview = preview.sort_values("sequence").head(10)

        print(f"\nFirst group in numeric order: {first_prefix}")
        print(preview[["uid", "text"]].to_string(index=False))


if __name__ == "__main__":
    main()
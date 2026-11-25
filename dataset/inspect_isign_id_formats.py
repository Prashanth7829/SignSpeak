from pathlib import Path

import pandas as pd


CSV_PATH = Path(__file__).resolve().parent / "iSign_v1.1.csv"


def main():
    rows = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    ids = rows["uid"].str.strip()

    standard = ids.str.fullmatch(r".+-\d+")
    unusual = rows.loc[~standard, ["uid", "text"]].copy()

    # Split at the final underscore because the prefix can contain underscores.
    suffix = unusual["uid"].str.strip().str.extract(
        r"^.+_([A-Za-z]+\d*)$", expand=False
    )
    unusual["suffix"] = suffix.fillna("(unmatched)")

    print(f"Standard video-sequence IDs: {standard.sum()}")
    print(f"Other IDs: {len(unusual)}")
    print("\nSuffix counts:")
    print(unusual["suffix"].value_counts().to_string())

    print("\nOne example per suffix:")
    for suffix_name, group in unusual.groupby("suffix", sort=True):
        row = group.iloc[0]
        print(f"\nSuffix: {suffix_name}")
        print(f"UID:    {row['uid']}")
        print(f"Text:   {row['text']}")


if __name__ == "__main__":
    main()
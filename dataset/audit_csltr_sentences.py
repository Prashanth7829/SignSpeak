from pathlib import Path

import pandas as pd


MANIFEST = Path(__file__).resolve().parent / "annotations" / "csltr_sentences.csv"


def normalize(series):
    return series.str.split().str.join(" ").str.casefold()


def main():
    data = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    data["sentence"] = normalize(data["english_sentence"])
    data["gloss"] = normalize(data["isl_gloss"])

    counts = data.groupby("sentence")["video_path_in_zip"].nunique()
    print(f"Distinct English sentences: {len(counts)}")
    print(f"Distinct gloss sequences: {data['gloss'].nunique()}")

    print("\nRecordings per sentence:")
    for recordings, sentences in counts.value_counts().sort_index().items():
        print(f"  {recordings} recording(s): {sentences} sentence(s)")

    print("\nSentences with the fewest recordings:")
    for sentence in counts[counts == counts.min()].index:
        print(f"  {counts[sentence]} | {sentence}")

    pairs = data[["sentence", "gloss"]].drop_duplicates()
    shared = pairs[pairs.duplicated("gloss", keep=False)]
    print("\nGloss sequences shared by different English sentences:")
    if shared.empty:
        print("  None")
    else:
        for gloss, group in shared.groupby("gloss"):
            print(f"  {gloss.upper()} -> {' / '.join(sorted(group['sentence']))}")


if __name__ == "__main__":
    main()
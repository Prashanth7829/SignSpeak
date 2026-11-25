from pathlib import Path
from random import Random

import pandas as pd


ANNOTATIONS = Path(__file__).resolve().parent / "annotations"
SOURCE = ANNOTATIONS / "csltr_sentences.csv"
OUTPUT = ANNOTATIONS / "csltr_sentence_splits.csv"
SEED = 42


def normalize(series):
    return series.str.split().str.join(" ").str.casefold()


def gloss_tokens(series):
    return {token for sequence in series for token in sequence.split()}


def main():
    data = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    data["gloss_group"] = normalize(data["isl_gloss"])
    sentences = normalize(data["english_sentence"])
    paths = normalize(data["video_path_in_zip"])

    if paths.duplicated().any():
        raise ValueError("A video path occurs more than once in the manifest.")
    if data["gloss_group"].eq("").any() or sentences.eq("").any():
        raise ValueError("The manifest contains an empty gloss or English sentence.")

    groups = sorted(data["gloss_group"].unique())
    if len(groups) != 100:
        raise ValueError(f"Expected 100 audited gloss groups, found {len(groups)}.")

    Random(SEED).shuffle(groups)
    assignment = dict.fromkeys(groups[:80], "train")
    assignment.update(dict.fromkeys(groups[80:90], "val"))
    assignment.update(dict.fromkeys(groups[90:], "test"))
    data["split"] = data["gloss_group"].map(assignment)

    if data.groupby(sentences)["split"].nunique().gt(1).any():
        raise ValueError("An English sentence appears in multiple splits.")

    train_tokens = gloss_tokens(data.loc[data["split"].eq("train"), "gloss_group"])
    for split in ("train", "val", "test"):
        subset = data.loc[data["split"].eq(split)]
        glosses = subset["gloss_group"]
        unseen = sorted(gloss_tokens(glosses) - train_tokens)
        print(
            f"{split}: {len(subset)} videos | "
            f"{glosses.nunique()} gloss groups | "
            f"{sentences.loc[subset.index].nunique()} English sentences"
        )
        if split != "train":
            print(f"  Gloss tokens absent from training: {', '.join(unseen) or 'none'}")

    temporary = OUTPUT.with_suffix(".tmp")
    data.sort_values(["split", "gloss_group", "video_path_in_zip"]).to_csv(
        temporary, index=False
    )
    temporary.replace(OUTPUT)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
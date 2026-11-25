import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PILOT_DIR = PROJECT_DIR / "dataset" / "include50_pilot"
MANIFEST_PATH = PILOT_DIR / "pilot_manifest.csv"
DICTIONARY_PATH = Path(__file__).resolve().parent / "pilot_dictionary.csv"


def main():
    entries = {}
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as manifest:
        for row in csv.DictReader(manifest):
            if row["split"] != "train":
                continue

            english = row["label"].split(". ", 1)[-1]
            key = " ".join(english.casefold().split())
            if (PILOT_DIR / row["video_path"]).is_file():
                entries.setdefault(
                    key,
                    {
                        "english": english,
                        "dataset_label": row["label"],
                        "video_path": row["video_path"],
                    },
                )

    if len(entries) != 5:
        raise ValueError(f"Expected 5 pilot dictionary entries, found {len(entries)}.")

    with DICTIONARY_PATH.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=["english", "dataset_label", "video_path"]
        )
        writer.writeheader()
        writer.writerows(entries.values())

    for entry in entries.values():
        print(f"{entry['english']} -> {entry['video_path']}")
    print(f"Saved: {DICTIONARY_PATH}")


if __name__ == "__main__":
    main()
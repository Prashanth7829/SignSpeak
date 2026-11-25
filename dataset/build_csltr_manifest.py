import csv
from collections import Counter, defaultdict
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from openpyxl import load_workbook


DATASET = Path(__file__).resolve().parent
ARCHIVE = DATASET / "raw" / "ISL-CSLTR" / "ISL_CSLRT_Corpus.zip"
ANNOTATIONS = "ISL_CSLRT_Corpus/corpus_csv_files/"
VIDEO_ROOT = "isl_cslrt_corpus/videos_sentence_level/"
OUTPUT_DIR = DATASET / "annotations"

SENTENCE_ALIASES = {
    "which collegeschool are you from": "which college school are you from",
}


def normalize(text):
    return " ".join(text.split()).casefold()


def write_csv(path, headers, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)
    temporary.replace(path)


def main():
    records = {}
    corrected_paths = set()
    issues = []

    with ZipFile(ARCHIVE) as archive:
        raw = archive.read(ANNOTATIONS + "ISL Corpus sign glosses.csv")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")

        glosses = defaultdict(set)
        for row in csv.DictReader(StringIO(text)):
            sentence = (row["Sentence"] or "").strip()
            gloss = " ".join((row["SIGN GLOSSES"] or "").split())
            if sentence and gloss:
                glosses[normalize(sentence)].add((sentence, gloss))

        videos = {}
        for entry in archive.infolist():
            path = entry.filename
            key = path.casefold()
            if key.startswith(VIDEO_ROOT) and key.endswith(".mp4"):
                if key in videos:
                    raise ValueError(f"Ambiguous archive path: {path}")
                videos[key] = path

        workbook = load_workbook(
            BytesIO(archive.read(ANNOTATIONS + "ISL_CSLRT_Corpus details.xlsx")),
            read_only=True,
            data_only=True,
        )
        try:
            rows = workbook["Sheet1"].iter_rows(values_only=True)
            headers = [str(value).strip() for value in next(rows)]
            sentence_col = headers.index("Sentences")
            path_col = headers.index("File location")

            for line, row in enumerate(rows, start=2):
                original = str(row[sentence_col] or "").strip()
                path = str(row[path_col] or "").strip().replace("\\", "/")

                # The audited workbook ends with a blank sentence and ">>>".
                if not original and path in ("", ">>>"):
                    continue

                key = normalize(original)
                lookup = SENTENCE_ALIASES.get(key, key)
                labels = glosses.get(lookup, set())
                video_path = videos.get(path.casefold())

                if video_path is None or len(labels) != 1:
                    issues.append(
                        f"Row {line}: {original!r}, {path!r}, "
                        f"video found={video_path is not None}, "
                        f"sentence/gloss matches={len(labels)}"
                    )
                    continue

                english, gloss = next(iter(labels))
                record = (video_path, english, gloss, original)
                if video_path in records and records[video_path] != record:
                    issues.append(f"Row {line}: conflicting labels for {video_path}")
                    continue

                records[video_path] = record
                if key != lookup:
                    corrected_paths.add(video_path)
        finally:
            workbook.close()

    if issues:
        raise ValueError("Unresolved annotations:\n" + "\n".join(issues[:15]))
    if not records:
        raise ValueError("No annotated video records matched.")

    unlisted = sorted(set(videos.values()) - set(records))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = OUTPUT_DIR / "csltr_sentences.csv"
    write_csv(
        manifest,
        ["video_path_in_zip", "english_sentence", "isl_gloss", "workbook_sentence"],
        (records[path] for path in sorted(records)),
    )

    review = OUTPUT_DIR / "csltr_unlisted_review.csv"
    write_csv(
        review,
        ["video_path_in_zip", "folder_name_for_review"],
        ((path, PurePosixPath(path).parent.name) for path in unlisted),
    )

    sentence_counts = Counter(normalize(row[1]) for row in records.values())
    print(f"MP4s in archive: {len(videos)}")
    print(f"Workbook-mapped videos: {len(records)}")
    print(f"Videos recovered by spelling correction: {len(corrected_paths)}")
    print(f"Distinct mapped sentences: {len(sentence_counts)}")
    print(f"Unlisted videos saved for review: {len(unlisted)}")
    print(f"Manifest: {manifest}")
    print(f"Review list: {review}")


if __name__ == "__main__":
    main()
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
COUNTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def main():
    with ZipFile(ARCHIVE) as archive:
        raw = archive.read(ANNOTATIONS + "ISL Corpus sign glosses.csv")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")

        print("Gloss CSV rows mentioning college or school:")
        for row in csv.DictReader(StringIO(text)):
            sentence = (row["Sentence"] or "").strip()
            gloss = (row["SIGN GLOSSES"] or "").strip()
            combined = f"{sentence} {gloss}".casefold()
            if "college" in combined or "school" in combined:
                print(f"  {sentence!r} -> {gloss!r}")

        workbook = load_workbook(
            BytesIO(archive.read(ANNOTATIONS + "ISL_CSLRT_Corpus details.xlsx")),
            read_only=True,
            data_only=True,
        )
        workbook_paths = set()
        try:
            rows = workbook["Sheet1"].iter_rows(values_only=True)
            headers = [str(value).strip() for value in next(rows)]
            path_col = headers.index("File location")

            for row in rows:
                path = str(row[path_col] or "").strip().replace("\\", "/")
                if path.casefold().startswith(VIDEO_ROOT):
                    workbook_paths.add(path.casefold())
        finally:
            workbook.close()

        extensions = Counter()
        other_examples = {}
        unlisted = defaultdict(list)

        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if entry.is_dir() or not entry.filename.casefold().startswith(VIDEO_ROOT):
                continue

            extension = path.suffix.casefold() or "(none)"
            extensions[extension] += 1

            if extension not in COUNTED_EXTENSIONS:
                other_examples.setdefault(extension, entry.filename)
            elif entry.filename.casefold() not in workbook_paths:
                unlisted[path.parent.name].append(path.name)

    print("\nAll extensions under Videos_Sentence_Level:")
    for extension, count in extensions.most_common():
        print(f"  {extension}: {count}")

    for extension, example in other_examples.items():
        print(f"  Example outside the original extension filter: {example}")

    print(f"\nUnique workbook paths under the video folder: {len(workbook_paths)}")
    print(f"Counted video entries absent from workbook: {sum(map(len, unlisted.values()))}")
    print(f"Sentence folders affected: {len(unlisted)}")
    print("\nExamples from up to 10 affected folders:")
    for folder, names in sorted(unlisted.items())[:10]:
        print(f"  {folder}: {len(names)} unlisted; example: {names[0]}")


if __name__ == "__main__":
    main()
import csv
from io import BytesIO, StringIO
from itertools import islice
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook


ARCHIVE = (
    Path(__file__).resolve().parent
    / "raw" / "ISL-CSLTR" / "ISL_CSLRT_Corpus.zip"
)
ANNOTATIONS = "ISL_CSLRT_Corpus/corpus_csv_files/"


def preview(rows):
    nonempty = (
        row for row in rows
        if any(value is not None and str(value).strip() for value in row)
    )
    for row in islice(nonempty, 5):
        values = ["" if value is None else str(value).strip() for value in row]
        while values and not values[-1]:
            values.pop()
        print(" | ".join(values))


def main():
    with ZipFile(ARCHIVE) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith(ANNOTATIONS):
                continue

            extension = Path(name).suffix.lower()
            if extension not in {".csv", ".xlsx"}:
                continue

            print(f"\nFILE: {Path(name).name}")
            data = archive.read(name)

            if extension == ".csv":
                try:
                    text = data.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = data.decode("cp1252")
                preview(csv.reader(StringIO(text)))
            else:
                workbook = load_workbook(BytesIO(data), read_only=True, data_only=True)
                try:
                    for sheet in workbook.worksheets:
                        print(f"Sheet: {sheet.title}")
                        preview(sheet.iter_rows(values_only=True))
                finally:
                    workbook.close()


if __name__ == "__main__":
    main()
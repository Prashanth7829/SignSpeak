"""Prepare compact RGB clips and a labeled playback catalog from local archives."""
import hashlib
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd

from signspeak.settings import CACHE, CATALOG, DATA

SAMPLES = CACHE / "samples"


def sample(path):
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise ValueError("Empty video")
        frames = []
        previous = -1
        for target in np.linspace(.10, .90, 8) * (count - 1):
            position = int(target)
            if position != previous:
                for _ in range(previous + 1, position + 1):
                    if not capture.grab():
                        raise ValueError("Incomplete video")
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError("Unreadable video")
                previous = position
            frames.append(cv2.cvtColor(cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        return np.stack(frames)
    finally:
        capture.release()


def prepare():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    members = {}
    signatures = {}
    for archive in sorted(DATA.glob("*.zip")):
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                if info.filename not in members:
                    members[info.filename] = archive
                    signatures[info.filename] = (info.CRC, info.file_size)
    rows = []
    absent = 0
    for table in sorted((DATA / "include_metadata").glob("*.parquet")):
        for item in pd.read_parquet(table).itertuples():
            name = item.video_path
            if item.include_50 or Path(name).parent.name.strip().casefold() != item.label.strip().casefold():
                continue
            if name in members:
                rows.append((members[name], name, re.sub(r"^\d+\.\s*", "", item.label).strip(), table.name.split("-")[0], "INCLUDE"))
            else:
                absent += 1
    # The metadata includes overlapping subset splits. Reserve test and validation
    # copies first and discard byte-identical recordings with conflicting labels.
    priority = {"test": 0, "val": 1, "train": 2}
    grouped = {}
    for row in sorted(rows, key=lambda row: priority[row[3]]):
        grouped.setdefault(signatures[row[1]], []).append(row)
    rows = [group[0] for group in grouped.values() if len({r[2].casefold() for r in group}) == 1]
    archive = DATA / "raw/ISL-CSLTR/ISL_CSLRT_Corpus.zip"
    for item in pd.read_csv(DATA / "annotations/csltr_clean_manifest.csv").itertuples():
        rows.append((archive, item.video_path_in_zip, item.english_sentence, item.split, "CSLTR"))
    catalog = []
    for archive, member, label, split, corpus in rows:
        key = hashlib.sha256((str(archive.relative_to(DATA)) + "/" + member).encode()).hexdigest()[:24]
        catalog.append(dict(archive=str(archive.relative_to(DATA)), member=member, label=label,
                            split=split, corpus=corpus, key=key))
    def convert(row):
        dest = SAMPLES / (row["key"] + ".npy")
        if dest.exists():
            return True
        try:
            with zipfile.ZipFile(DATA / row["archive"]) as source, TemporaryDirectory() as directory:
                path = Path(directory) / ("clip" + Path(row["member"]).suffix)
                path.write_bytes(source.read(row["member"]))
                rgb = sample(path)
                temp = dest.with_suffix(".tmp")
                with temp.open("wb") as file:
                    np.save(file, rgb)
                temp.replace(dest)
            return True
        except (OSError, ValueError, KeyError, cv2.error, zipfile.BadZipFile) as exc:
            print(row["member"], exc, flush=True)
            return False
    print(f"Preparing {len(catalog)} clips; {absent} metadata paths absent", flush=True)
    good = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        for index, (row, ok) in enumerate(zip(catalog, pool.map(convert, catalog)), 1):
            if ok:
                good.append(row)
            if index % 200 == 0:
                print(index, flush=True)
    tmp = CATALOG.with_suffix(".tmp")
    tmp.write_text(json.dumps(good, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CATALOG)
    print("Prepared", len(good), flush=True)


if __name__ == "__main__":
    prepare()

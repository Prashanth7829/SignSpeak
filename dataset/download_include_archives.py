import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


DATASET_DIR = Path(__file__).resolve().parent
RECORD_API = "https://zenodo.org/api/records/4010759"
FILES_URL = "https://zenodo.org/records/4010759/files"


def md5_digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "md5").hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Download one INCLUDE category.")
    parser.add_argument("category", help="Zenodo category prefix, such as Greetings")
    category = parser.parse_args().category

    with urlopen(RECORD_API, timeout=60) as response:
        record = json.load(response)

    archives = sorted(
        (
            entry
            for entry in record["files"]
            if entry["key"].startswith(f"{category}_")
            and entry["key"].endswith(".zip")
        ),
        key=lambda entry: entry["key"],
    )
    if not archives:
        raise SystemExit(f"No ZIP archives found for category {category!r}.")

    for entry in archives:
        name = entry["key"]
        algorithm, expected_hash = entry["checksum"].split(":", 1)
        if algorithm != "md5":
            raise SystemExit(f"Unexpected checksum type for {name}: {algorithm}")

        destination = DATASET_DIR / name
        if destination.exists():
            if md5_digest(destination) != expected_hash.lower():
                raise SystemExit(f"Checksum mismatch for existing file: {destination}")
            print(f"Already verified: {name}", flush=True)
            continue

        partial = destination.with_suffix(".zip.part")
        url = f"{FILES_URL}/{quote(name)}?download=1"
        print(f"Downloading {name} ({entry['size'] / 1e9:.2f} GB)", flush=True)
        subprocess.run(
            [
                "curl.exe", "--fail", "--location", "--retry", "3",
                "--output", str(partial), url,
            ],
            check=True,
        )

        if md5_digest(partial) != expected_hash.lower():
            raise SystemExit(f"Checksum mismatch for downloaded file: {partial}")

        partial.replace(destination)
        print(f"Verified: {name}", flush=True)


if __name__ == "__main__":
    main()
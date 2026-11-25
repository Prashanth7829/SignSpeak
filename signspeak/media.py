import csv
import json
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe

from signspeak.settings import PILOT, ROOT, CATALOG, DATA


@dataclass(frozen=True)
class ArchivedClip:
    archive: Path
    member: str

    def is_file(self):
        return self.archive.is_file()


def words(text):
    return re.findall(r"[^\W_]+", text.casefold())


def dictionary():
    entries = {}
    with (ROOT / "dictionary.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            label = row["english"].strip()
            key = tuple(words(label))
            if not key or key in entries:
                raise ValueError(f"Empty or duplicate English entry in dictionary.csv: {label!r}")
            entries[key] = label, PILOT / row["video_path"]
    if CATALOG.is_file():
        rows = json.loads(CATALOG.read_text(encoding="utf8"))
        for row in sorted(rows, key=lambda r: r["split"] != "train"):
            key = tuple(words(row["label"]))
            if key and (key not in entries or not entries[key][1].is_file()):
                entries[key] = row["label"], ArchivedClip(DATA / row["archive"], row["member"])
    return entries


def map_signs(text, entries):
    tokens = words(text)
    length = max(map(len, entries), default=0)
    position = 0
    while position < len(tokens):
        for size in range(min(length, len(tokens) - position), 0, -1):
            key = tuple(tokens[position:position + size])
            if key in entries:
                yield entries[key]
                position += size
                break
        else:
            yield tokens[position], None
            position += 1


def render_clips(paths, output):
    command = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    filters = []
    for index, path in enumerate(paths):
        if isinstance(path, ArchivedClip):
            source = output.parent / f"source-{index}{Path(path.member).suffix}"
            with zipfile.ZipFile(path.archive) as archive:
                source.write_bytes(archive.read(path.member))
            path = source
        command.extend(("-i", str(path)))
        filters.append(
            f"[{index}:v]scale=640:480:force_original_aspect_ratio=decrease:force_divisible_by=2,"
            f"pad=640:480:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p,"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
    streams = "".join(f"[v{index}]" for index in range(len(paths)))
    filters.append(f"{streams}concat=n={len(paths)}:v=1:a=0[out]")
    command.extend(("-filter_complex", ";".join(filters), "-map", "[out]",
                    "-c:v", "libx264", "-preset", "veryfast", "-movflags", "+faststart", str(output)))
    run_media(command, timeout=90)
    return output.read_bytes()


def run_media(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", **kwargs)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Media processing failed.")

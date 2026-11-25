import csv
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import urlopen


PAGE_URL = "https://exploration-lab.github.io/iSign/"
CSV_PATH = Path(__file__).resolve().parent / "iSign_v1.1.csv"


class VideoSources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = set()

    def handle_starttag(self, tag, attrs):
        if tag in ("video", "source"):
            src = dict(attrs).get("src")
            if src and urlparse(src).path.lower().endswith(".mp4"):
                self.urls.add(urljoin(PAGE_URL, src))


def main():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        translations = {
            row["uid"].strip(): row["text"].strip()
            for row in csv.DictReader(file)
        }

    with urlopen(PAGE_URL, timeout=30) as response:
        html = response.read().decode("utf-8")

    parser = VideoSources()
    parser.feed(html)

    matched = 0
    for url in sorted(parser.urls):
        path = PurePosixPath(unquote(urlparse(url).path))
        uid = path.stem
        text = translations.get(uid)
        if text:
            matched += 1
            print(f"{path.parent.name} | {uid} | {text}")
        else:
            print(f"{path.parent.name} | {uid} | no matching nonblank CSV text")

    print(f"\nEmbedded MP4 links: {len(parser.urls)}")
    print(f"Links with an exact UID and nonblank text: {matched}")


if __name__ == "__main__":
    main()
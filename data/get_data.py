"""Download a few public-domain long documents for the perplexity evaluation.

    python data/get_data.py

Writes data/*.txt (Project Gutenberg plain-text). Any other .txt works too:
pass --text your_file.txt to the eval scripts.
"""
import os
import urllib.request

BOOKS = {
    "pride_and_prejudice.txt": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "moby_dick.txt": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "frankenstein.txt": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
}


def strip_gutenberg_header(text: str) -> str:
    start = text.find("*** START OF")
    if start != -1:
        text = text[text.find("\n", start) + 1:]
    end = text.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text.strip()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for name, url in BOOKS.items():
        path = os.path.join(here, name)
        if os.path.exists(path):
            print("exists", path)
            continue
        print("downloading", url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", errors="ignore")
        with open(path, "w", encoding="utf-8") as f:
            f.write(strip_gutenberg_header(raw))
        print("wrote", path)


if __name__ == "__main__":
    main()

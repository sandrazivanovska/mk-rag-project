#!/usr/bin/env python3
"""
Download all datasets needed for the MK-RAG thesis project.

Usage:
    python scripts/download_datasets.py                  # download everything
    python scripts/download_datasets.py --only wiki      # only MK Wikipedia
    python scripts/download_datasets.py --only lvstck    # only HuggingFace corpus
    python scripts/download_datasets.py --only setimes   # only SETimes parallel
    python scripts/download_datasets.py --only en-wiki   # only EN Wikipedia mirror
    python scripts/download_datasets.py --skip lvstck    # everything except LVSTCK

Datasets downloaded:
    1. MK Wikipedia dump (dumps.wikimedia.org)          ~300 MB compressed
    2. LVSTCK/macedonian-corpus-cleaned (HuggingFace)   streams up to --max-lvstck docs
    3. SETimes MK-EN parallel corpus (OPUS)             ~20 MB
    4. EN Wikipedia mirror of MK articles (via Wikidata) per-article API calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# ── Progress bar (no extra deps) ──────────────────────────────────────────────

def _progress_hook(count: int, block_size: int, total_size: int) -> None:
    downloaded = count * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB")
        sys.stdout.flush()
    if count * block_size >= total_size > 0:
        print()


def download_file(url: str, dest: Path, *, retries: int = 3, desc: str = "") -> bool:
    """Download a file with retry logic and progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name} — skipping")
        return True

    label = desc or dest.name
    for attempt in range(1, retries + 1):
        try:
            print(f"  Downloading {label} (attempt {attempt}/{retries})...")
            urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  ✓ Saved → {dest}  ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            print(f"  ✗ Attempt {attempt} failed: {e}")
            if dest.exists():
                dest.unlink()
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    return False


# ── 1. MK Wikipedia dump ──────────────────────────────────────────────────────

def download_mk_wikipedia(data_dir: Path) -> bool:
    """Download the latest MK Wikipedia dump."""
    print("\n[1/4] MK Wikipedia dump")
    url = "https://dumps.wikimedia.org/mkwiki/latest/mkwiki-latest-pages-articles.xml.bz2"
    dest = data_dir / "raw" / "mk" / "mkwiki-latest-pages-articles.xml.bz2"
    return download_file(url, dest, desc="mkwiki dump (~300 MB)")


# ── 2. LVSTCK Macedonian corpus (HuggingFace) ─────────────────────────────────

def download_lvstck(data_dir: Path, max_docs: int = 200_000) -> bool:
    """Stream the LVSTCK Macedonian corpus from HuggingFace."""
    print(f"\n[2/4] LVSTCK/macedonian-corpus-cleaned (first {max_docs:,} docs)")
    out_file = data_dir / "raw" / "mk" / "lvstck_mk.jsonl"

    if out_file.exists():
        line_count = sum(1 for _ in open(out_file, encoding="utf-8"))
        print(f"  ✓ Already exists: {out_file.name} ({line_count:,} docs) — skipping")
        return True

    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("  Installing dependencies: datasets tqdm")
        os.system(f"{sys.executable} -m pip install datasets tqdm -q")
        from datasets import load_dataset
        from tqdm import tqdm

    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("  Streaming from HuggingFace (no full download needed)...")
    try:
        dataset = load_dataset(
            "LVSTCK/macedonian-corpus-cleaned",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            pbar = tqdm(total=max_docs, desc="  LVSTCK", unit=" docs")
            for item in dataset:
                if count >= max_docs:
                    break
                text = item.get("text", "").strip()
                if not text:
                    continue
                doc = {
                    "id": f"lvstck_{count:07d}",
                    "title": "",
                    "url": item.get("url", ""),
                    "text": text,
                    "lang": "mk",
                    "source": "lvstck",
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1
                pbar.update(1)
            pbar.close()

        size_mb = out_file.stat().st_size / 1_048_576
        print(f"  ✓ Saved {count:,} docs → {out_file}  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ LVSTCK download failed: {e}")
        return False


# ── 3. SETimes parallel corpus (OPUS) ─────────────────────────────────────────

def download_setimes(data_dir: Path) -> bool:
    """Download the SETimes MK-EN parallel corpus from OPUS."""
    print("\n[3/4] SETimes MK-EN parallel corpus")
    raw_dir = data_dir / "raw" / "setimes"

    url = "https://object.pouta.csc.fi/OPUS-SETimes/v2/moses/en-mk.txt.zip"
    dest = raw_dir / "en-mk.txt.zip"

    if not download_file(url, dest, desc="SETimes corpus (~20 MB)"):
        # Fallback: try NLTK COMTRANS if available
        print("  Trying fallback source...")
        url2 = "https://opus.nlpl.eu/download.php?f=SETimes/v2/moses/en-mk.txt.zip"
        if not download_file(url2, dest, desc="SETimes corpus (fallback)"):
            return False

    # Unzip
    import zipfile
    print(f"  Extracting {dest.name}...")
    with zipfile.ZipFile(dest, "r") as z:
        z.extractall(raw_dir)
    print(f"  ✓ Extracted → {raw_dir}")

    # Convert to JSONL
    mk_txt = raw_dir / "SETimes.en-mk.mk"
    en_txt = raw_dir / "SETimes.en-mk.en"

    if mk_txt.exists() and en_txt.exists():
        mk_jsonl = data_dir / "raw" / "mk" / "setimes_mk.jsonl"
        en_jsonl = data_dir / "raw" / "en" / "setimes_en.jsonl"
        mk_jsonl.parent.mkdir(parents=True, exist_ok=True)
        en_jsonl.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with (
            open(mk_txt, encoding="utf-8") as mk_f,
            open(en_txt, encoding="utf-8") as en_f,
            open(mk_jsonl, "w", encoding="utf-8") as mk_out,
            open(en_jsonl, "w", encoding="utf-8") as en_out,
        ):
            for i, (mk_line, en_line) in enumerate(zip(mk_f, en_f)):
                mk_line = mk_line.strip()
                en_line = en_line.strip()
                if not mk_line or not en_line:
                    continue
                mk_out.write(json.dumps({
                    "id": f"setimes_{i:06d}", "text": mk_line,
                    "lang": "mk", "source": "setimes", "pair_id": i,
                }, ensure_ascii=False) + "\n")
                en_out.write(json.dumps({
                    "id": f"setimes_{i:06d}", "text": en_line,
                    "lang": "en", "source": "setimes", "pair_id": i,
                }, ensure_ascii=False) + "\n")
                count += 1
        print(f"  ✓ Converted {count:,} sentence pairs → JSONL")
    return True


# ── 4. English Wikipedia mirror (Wikidata-linked to MK articles) ──────────────

def download_en_wikipedia_mirror(data_dir: Path, max_articles: int = 5000) -> bool:
    """
    Fetch English Wikipedia articles that are linked to Macedonian articles
    via Wikidata interlanguage links.

    Uses the Wikipedia API — no authentication required.
    """
    print(f"\n[4/4] EN Wikipedia mirror (linked to MK articles, max {max_articles:,})")

    out_file = data_dir / "raw" / "en" / "en_wiki_mirror.jsonl"
    if out_file.exists():
        line_count = sum(1 for _ in open(out_file, encoding="utf-8"))
        print(f"  ✓ Already exists ({line_count:,} articles) — skipping")
        return True

    out_file.parent.mkdir(parents=True, exist_ok=True)

    import urllib.parse

    MK_API = "https://mk.wikipedia.org/w/api.php"
    EN_API = "https://en.wikipedia.org/w/api.php"

    def api_get(base_url: str, params: dict) -> dict:
        params["format"] = "json"
        url = base_url + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_en_title_for_mk(mk_title: str) -> str | None:
        """Find the English Wikipedia title for a given MK article via Wikidata."""
        try:
            data = api_get(MK_API, {
                "action": "query",
                "titles": mk_title,
                "prop": "langlinks",
                "lllang": "en",
                "lllimit": 1,
            })
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                for ll in page.get("langlinks", []):
                    if ll.get("lang") == "en":
                        return ll.get("*")
        except Exception:
            pass
        return None

    def fetch_en_article(en_title: str) -> str | None:
        """Fetch the plain-text extract of an EN Wikipedia article."""
        try:
            data = api_get(EN_API, {
                "action": "query",
                "titles": en_title,
                "prop": "extracts",
                "explaintext": True,
                "exsectionformat": "plain",
                "exlimit": 1,
            })
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                return page.get("extract", "")
        except Exception:
            pass
        return None

    # Step 1: Get list of MK Wikipedia articles
    print("  Fetching MK article list...")
    mk_titles = []
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": 0,
        "aplimit": 500,
    }
    collected = 0
    while collected < max_articles:
        try:
            data = api_get(MK_API, params)
            pages = data.get("query", {}).get("allpages", [])
            mk_titles.extend(p["title"] for p in pages)
            collected += len(pages)
            cont = data.get("continue", {}).get("apcontinue")
            if not cont or collected >= max_articles:
                break
            params["apcontinue"] = cont
            time.sleep(0.1)  # polite delay
        except Exception as e:
            print(f"  ✗ Failed to list MK articles: {e}")
            break

    print(f"  Found {len(mk_titles):,} MK article titles")

    # Step 2: For each MK article, find EN equivalent and download
    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for i, mk_title in enumerate(mk_titles[:max_articles]):
            en_title = get_en_title_for_mk(mk_title)
            if not en_title:
                continue
            en_text = fetch_en_article(en_title)
            if not en_text or len(en_text) < 100:
                continue

            doc = {
                "id": f"en_wiki_{i:06d}",
                "title": en_title,
                "mk_title": mk_title,
                "text": en_text,
                "lang": "en",
                "source": "en_wikipedia",
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

            if count % 100 == 0:
                print(f"  ... {count} EN articles downloaded")
            time.sleep(0.05)  # rate limit

    size_mb = out_file.stat().st_size / 1_048_576
    print(f"  ✓ Saved {count:,} EN articles → {out_file}  ({size_mb:.1f} MB)")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download MK-RAG datasets")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--only", choices=["wiki", "lvstck", "setimes", "en-wiki"],
        help="Download only this dataset",
    )
    parser.add_argument(
        "--skip", choices=["wiki", "lvstck", "setimes", "en-wiki"],
        action="append", default=[],
        help="Skip this dataset (can specify multiple times)",
    )
    parser.add_argument(
        "--max-lvstck", type=int, default=200_000,
        help="Max LVSTCK documents to download (default: 200,000)",
    )
    parser.add_argument(
        "--max-en-wiki", type=int, default=5000,
        help="Max EN Wikipedia mirror articles (default: 5,000)",
    )
    args = parser.parse_args()

    def should_run(name: str) -> bool:
        if args.only:
            return args.only == name
        return name not in args.skip

    print("=" * 60)
    print("  MK-RAG Dataset Downloader")
    print("=" * 60)
    print(f"  Data root: {args.data_dir.resolve()}")

    results = {}

    if should_run("wiki"):
        results["MK Wikipedia"] = download_mk_wikipedia(args.data_dir)

    if should_run("lvstck"):
        results["LVSTCK corpus"] = download_lvstck(args.data_dir, args.max_lvstck)

    if should_run("setimes"):
        results["SETimes parallel"] = download_setimes(args.data_dir)

    if should_run("en-wiki"):
        results["EN Wiki mirror"] = download_en_wikipedia_mirror(args.data_dir, args.max_en_wiki)

    print("\n" + "=" * 60)
    print("  Download Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status}  {name}")

    print()
    if all(results.values()):
        print("  All datasets ready. Next step:")
        print("    python main.py setup-data")
        print("    python main.py build-indices")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  Some downloads failed: {', '.join(failed)}")
        print("  Check your internet connection and try again.")
    print()


if __name__ == "__main__":
    main()

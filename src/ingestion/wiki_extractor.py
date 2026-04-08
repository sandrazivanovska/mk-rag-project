"""
Wikipedia dump extraction for Macedonian (mk) and English (en).

Uses WikiExtractor under the hood. The extracted plain-text articles
are saved as JSONL files with fields: id, title, url, text, lang.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Generator

from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger("wiki_extractor")


def extract_mk_wikipedia(
    dump_path: str | Path,
    output_dir: str | Path,
    *,
    min_length: int = 200,
) -> Path:
    """
    Extract articles from a Macedonian Wikipedia XML dump.

    Args:
        dump_path: Path to the .xml.bz2 dump file.
        output_dir: Directory to write the extracted JSONL.
        min_length: Minimum article character length to keep.

    Returns:
        Path to the output JSONL file.
    """
    dump_path = Path(dump_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mk_wiki.jsonl"

    logger.info(f"Extracting MK Wikipedia from {dump_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Run WikiExtractor
        subprocess.run(
            [
                "python", "-m", "wikiextractor.WikiExtractor",
                str(dump_path),
                "--output", tmp_dir,
                "--json",
                "--processes", "4",
                "--quiet",
            ],
            check=True,
        )

        # Collect and write to single JSONL
        count = 0
        with open(output_file, "w", encoding="utf-8") as out_f:
            for article in _iter_wikiextractor_output(Path(tmp_dir)):
                if len(article["text"]) < min_length:
                    continue
                article["lang"] = "mk"
                out_f.write(json.dumps(article, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"Extracted {count:,} MK Wikipedia articles → {output_file}")
    return output_file


def extract_en_wikipedia(
    wikidata_linked_ids: list[str],
    output_dir: str | Path,
    *,
    lang: str = "en",
) -> Path:
    """
    Extract English Wikipedia articles that are linked to Macedonian
    counterparts via Wikidata interlanguage links.

    For each Macedonian article we retrieve the corresponding EN article
    using the Wikipedia REST API.

    Args:
        wikidata_linked_ids: List of EN Wikipedia page titles.
        output_dir: Directory to write the extracted JSONL.
        lang: Source language code (default: 'en').

    Returns:
        Path to the output JSONL file.
    """
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "en_wiki_linked.jsonl"

    API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

    logger.info(f"Fetching {len(wikidata_linked_ids):,} EN Wikipedia articles via REST API")

    count = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for title in tqdm(wikidata_linked_ids, desc="EN Wikipedia"):
            try:
                resp = requests.get(API_URL.format(title=title), timeout=10)
                resp.raise_for_status()
                data = resp.json()
                article = {
                    "id": str(data.get("pageid", "")),
                    "title": data.get("title", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "text": data.get("extract", ""),
                    "lang": lang,
                }
                if article["text"]:
                    out_f.write(json.dumps(article, ensure_ascii=False) + "\n")
                    count += 1
            except Exception as exc:
                logger.warning(f"Skipped '{title}': {exc}")

    logger.info(f"Fetched {count:,} EN Wikipedia articles → {output_file}")
    return output_file


def get_mk_en_wikidata_links(mk_wiki_jsonl: str | Path) -> list[str]:
    """
    Given a JSONL of MK Wikipedia articles, query the Wikidata API
    to find linked EN Wikipedia page titles.

    Returns a list of EN page titles.
    """
    import requests

    mk_wiki_jsonl = Path(mk_wiki_jsonl)
    mk_titles = []
    with open(mk_wiki_jsonl, encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            mk_titles.append(article["title"])

    logger.info(f"Querying Wikidata for {len(mk_titles):,} MK titles → EN links")

    # Batch queries (50 titles per request)
    en_titles = []
    batch_size = 50
    for i in tqdm(range(0, len(mk_titles), batch_size), desc="Wikidata links"):
        batch = mk_titles[i : i + batch_size]
        titles_param = "|".join(batch)
        params = {
            "action": "query",
            "prop": "langlinks",
            "titles": titles_param,
            "lllang": "en",
            "format": "json",
            "formatversion": "2",
        }
        try:
            resp = requests.get(
                "https://mk.wikipedia.org/w/api.php", params=params, timeout=15
            )
            pages = resp.json().get("query", {}).get("pages", [])
            for page in pages:
                for ll in page.get("langlinks", []):
                    if ll.get("lang") == "en":
                        en_titles.append(ll["title"])
        except Exception as exc:
            logger.warning(f"Wikidata batch {i} failed: {exc}")

    return en_titles


# ── Helpers ────────────────────────────────────────────────────────────────────

def _iter_wikiextractor_output(base_dir: Path) -> Generator[dict, None, None]:
    """Iterate over all JSONL files produced by WikiExtractor."""
    for json_file in sorted(base_dir.rglob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

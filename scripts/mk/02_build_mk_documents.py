from __future__ import annotations

import json
import logging
import re
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


INPUT_DIR = Path("data/raw/mk_wikipedia/extracted")
MANUAL_TEXTS_DIR = Path("data/raw/manual_texts")
OUTPUT_PATH = Path("data/processed/mk_documents.jsonl")

MIN_TEXT_LENGTH = 300
MIN_CYRILLIC_RATIO = 0.70

WHITESPACE_PATTERN = re.compile(r"\s+")
LETTER_PATTERN = re.compile(r"[A-Za-zА-Яа-яЃѓЌќЅѕЉљЊњЈјЏџШшЧчЖж]")
CYRILLIC_PATTERN = re.compile(r"[А-Яа-яЃѓЌќЅѕЉљЊњЈјЏџШшЧчЖж]")


logger = logging.getLogger(__name__)


Document = Dict[str, Any]


def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove non-breaking spaces.
    """

    text = text.replace("\xa0", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def calculate_cyrillic_ratio(text: str) -> float:
    """
    Return the ratio of Cyrillic letters to all Latin/Cyrillic letters.

    This helps filter out texts that are not primarily Macedonian/Cyrillic.
    """

    if not text:
        return 0.0

    letters = LETTER_PATTERN.findall(text)

    if not letters:
        return 0.0

    cyrillic_letters = CYRILLIC_PATTERN.findall(text)

    return len(cyrillic_letters) / len(letters)


def is_good_macedonian_text(text: str) -> bool:
    """
    Check whether a text is long enough and mostly written in Cyrillic.
    """

    cleaned_text = clean_text(text)

    if len(cleaned_text) < MIN_TEXT_LENGTH:
        return False

    if calculate_cyrillic_ratio(cleaned_text) < MIN_CYRILLIC_RATIO:
        return False

    return True


def iter_wikiextractor_articles(input_dir: Path) -> Iterator[Document]:
    """
    Iterate over WikiExtractor JSONL files and yield clean Macedonian documents.
    """

    if not input_dir.exists():
        logger.warning("Wikipedia input directory does not exist: %s", input_dir)
        return

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue

        try:
            with path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        article = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping invalid JSON in %s at line %d",
                            path,
                            line_number,
                        )
                        continue

                    raw_text = str(article.get("text", ""))
                    text = clean_text(raw_text)

                    if not is_good_macedonian_text(text):
                        continue

                    wiki_id = str(article.get("id", "")).strip()
                    title = str(article.get("title", "")).strip()
                    url = str(article.get("url", "")).strip()

                    if not wiki_id:
                        logger.warning(
                            "Skipping Wikipedia article without id in %s at line %d",
                            path,
                            line_number,
                        )
                        continue

                    yield {
                        "doc_id": f"mk_wiki_{wiki_id}",
                        "source": "wikipedia",
                        "title": title,
                        "language": "mk",
                        "text": text,
                        "url": url or None,
                        "metadata": {
                            "original_id": wiki_id,
                            "domain": "general",
                            "source_file": str(path),
                        },
                    }

        except OSError as error:
            logger.error("Could not read file %s: %s", path, error)


def iter_manual_text_files(manual_dir: Path) -> Iterator[Document]:
    """
    Iterate over optional manual Macedonian .txt files.

    Example:
        data/raw/manual_texts/finki_info.txt
        data/raw/manual_texts/news_article_001.txt
    """

    if not manual_dir.exists():
        logger.info("Manual texts directory does not exist: %s", manual_dir)
        return

    for index, path in enumerate(sorted(manual_dir.glob("*.txt")), start=1):
        try:
            text = clean_text(path.read_text(encoding="utf-8"))
        except OSError as error:
            logger.error("Could not read manual text file %s: %s", path, error)
            continue

        if not is_good_macedonian_text(text):
            logger.warning(
                "Skipping manual file because it is too short or not mostly Cyrillic: %s",
                path,
            )
            continue

        yield {
            "doc_id": f"mk_manual_{index:06d}",
            "source": "manual",
            "title": path.stem,
            "language": "mk",
            "text": text,
            "url": None,
            "metadata": {
                "filename": path.name,
                "domain": "manual",
            },
        }


def write_jsonl(records: Iterable[Document], output_path: Path) -> int:
    """
    Write documents to a JSONL file while removing duplicate document IDs.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    seen_doc_ids: set[str] = set()

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            doc_id = str(record.get("doc_id", "")).strip()

            if not doc_id:
                logger.warning("Skipping record without doc_id")
                continue

            if doc_id in seen_doc_ids:
                logger.warning("Skipping duplicate document id: %s", doc_id)
                continue

            seen_doc_ids.add(doc_id)

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()

    logger.info("Reading Wikipedia articles from %s", INPUT_DIR)
    logger.info("Reading manual text files from %s", MANUAL_TEXTS_DIR)

    records = chain(
        iter_wikiextractor_articles(INPUT_DIR),
        iter_manual_text_files(MANUAL_TEXTS_DIR),
    )

    logger.info("Writing documents to %s", OUTPUT_PATH)
    count = write_jsonl(records, OUTPUT_PATH)

    logger.info("Done. Total documents written: %d", count)


if __name__ == "__main__":
    main()
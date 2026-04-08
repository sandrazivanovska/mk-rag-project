"""
Loaders for additional Macedonian corpora:
  - LVSTCK/macedonian-corpus-cleaned (HuggingFace)
  - SETimes.MK parallel corpus
  - MaCoCu-mk 2.0 (CLARIN)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger("corpus_loader")

# ── HuggingFace: LVSTCK/macedonian-corpus-cleaned ─────────────────────────────


def load_hf_corpus(
    output_dir: str | Path,
    *,
    max_docs: int = 200_000,
    streaming: bool = True,
) -> Path:
    """
    Stream the LVSTCK/macedonian-corpus-cleaned dataset from HuggingFace
    and save a subset as JSONL.

    Args:
        output_dir: Directory to write JSONL.
        max_docs: Maximum number of documents to load.
        streaming: Use streaming mode to avoid downloading the full 1.47B dataset.

    Returns:
        Path to the output JSONL file.
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "lvstck_mk.jsonl"

    logger.info(f"Loading LVSTCK Macedonian corpus (max {max_docs:,} docs, streaming={streaming})")

    dataset = load_dataset(
        "LVSTCK/macedonian-corpus-cleaned",
        split="train",
        streaming=streaming,
    )

    count = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for item in tqdm(dataset, desc="LVSTCK corpus", total=max_docs):
            if count >= max_docs:
                break
            doc = {
                "id": str(count),
                "title": "",
                "url": item.get("url", ""),
                "text": item.get("text", ""),
                "lang": "mk",
                "source": "lvstck",
            }
            if doc["text"].strip():
                out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"Saved {count:,} LVSTCK documents → {output_file}")
    return output_file


# ── SETimes parallel corpus ────────────────────────────────────────────────────


def load_setimes(
    mk_file: str | Path,
    en_file: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Load the SETimes MK-EN parallel corpus from tab-separated files.

    Each line in mk_file / en_file corresponds to a sentence.
    Returns paths to JSONL files for MK and EN respectively.
    """
    mk_file = Path(mk_file)
    en_file = Path(en_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mk_out = output_dir / "setimes_mk.jsonl"
    en_out = output_dir / "setimes_en.jsonl"

    logger.info("Loading SETimes MK-EN parallel corpus")

    with (
        open(mk_file, encoding="utf-8") as mk_f,
        open(en_file, encoding="utf-8") as en_f,
        open(mk_out, "w", encoding="utf-8") as mk_out_f,
        open(en_out, "w", encoding="utf-8") as en_out_f,
    ):
        for idx, (mk_line, en_line) in enumerate(zip(mk_f, en_f)):
            mk_line = mk_line.strip()
            en_line = en_line.strip()
            if not mk_line or not en_line:
                continue

            mk_doc = {
                "id": f"setimes-{idx}",
                "text": mk_line,
                "lang": "mk",
                "source": "setimes",
                "pair_id": idx,
            }
            en_doc = {
                "id": f"setimes-{idx}",
                "text": en_line,
                "lang": "en",
                "source": "setimes",
                "pair_id": idx,
            }
            mk_out_f.write(json.dumps(mk_doc, ensure_ascii=False) + "\n")
            en_out_f.write(json.dumps(en_doc, ensure_ascii=False) + "\n")

    logger.info(f"SETimes → {mk_out} / {en_out}")
    return mk_out, en_out


# ── Generic JSONL helpers ──────────────────────────────────────────────────────


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Iterate over records in a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def merge_jsonl(sources: list[str | Path], output: str | Path) -> Path:
    """Merge multiple JSONL files into one."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output, "w", encoding="utf-8") as out_f:
        for src in sources:
            for record in iter_jsonl(src):
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    logger.info(f"Merged {len(sources)} sources → {output} ({count:,} records)")
    return output

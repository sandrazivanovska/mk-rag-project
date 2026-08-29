"""
Step 14 — Add English counterpart chunk IDs to the gold dataset.

Why this is needed
------------------
The gold set marks relevance with Macedonian chunk IDs (``mk_wiki_*``). Three of
the six pipelines (translate_retrieve, cross_lingual_embed, bilingual_fusion)
retrieve from the ENGLISH index, whose IDs live in a different namespace
(``en_wiki_*`` / ``en_mt_mk_wiki_*``). Scoring those pipelines against the MK
IDs would report 0.0 retrieval for all of them — a measurement artefact that
looks like total failure.

Every EN chunk carries an ``mk_doc_id`` field pointing back at the Macedonian
article it corresponds to, so the mapping is a direct join.

Granularity note
----------------
The MK side is relevant at CHUNK level (one specific chunk answers the
question). The EN side can only be resolved at DOCUMENT level: we know which
English article corresponds to the Macedonian one, but not which English chunk
matches the specific Macedonian chunk. So ``relevant_doc_ids_en`` holds every EN
chunk of the counterpart article. This makes EN retrieval look *easier* than MK
at chunk level, and any comparison must account for it — see
``--strict-doc-level``, which also emits a document-level MK key so both sides
can be scored on equal terms.

Usage
-----
    python scripts/mk/14_build_crosslingual_gold.py
    python scripts/mk/14_build_crosslingual_gold.py --strict-doc-level
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

DEFAULT_GOLD = Path("data/evaluation/gold_dataset.jsonl")
DEFAULT_EN_CHUNKS = Path("data/processed/en/chunks.jsonl")
DEFAULT_OUT = Path("data/evaluation/gold_dataset_crosslingual.jsonl")


def build_mk_to_en(en_chunks: Path) -> dict[str, list[str]]:
    """Map each Macedonian doc id to the EN chunk ids derived from it."""
    mapping: dict[str, list[str]] = collections.defaultdict(list)
    with open(en_chunks, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            mk = d.get("mk_doc_id")
            if mk:
                mapping[mk].append(d["chunk_id"])
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--en-chunks", type=Path, default=DEFAULT_EN_CHUNKS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--strict-doc-level",
        action="store_true",
        help="Also emit relevant_doc_ids_mk_doclevel (every MK chunk of the "
             "source article), so MK and EN can be scored at the same "
             "document granularity.",
    )
    args = ap.parse_args()

    mk2en = build_mk_to_en(args.en_chunks)
    print(f"EN chunks grouped under {len(mk2en):,} Macedonian doc ids")

    mk_doc_chunks: dict[str, list[str]] = collections.defaultdict(list)
    if args.strict_doc_level:
        with open("data/processed/mk/chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                mk_doc_chunks[d["doc_id"]].append(d["chunk_id"])

    rows, matched = [], 0
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            en_ids = mk2en.get(g["source_doc_id"], [])
            g["relevant_doc_ids_en"] = en_ids
            g["has_en_counterpart"] = bool(en_ids)
            if en_ids:
                matched += 1
            if args.strict_doc_level:
                g["relevant_doc_ids_mk_doclevel"] = mk_doc_chunks.get(
                    g["source_doc_id"], list(g["relevant_doc_ids"])
                )
            rows.append(g)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for g in rows:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"gold rows: {n}")
    print(f"with EN counterpart: {matched} ({matched / n * 100:.1f}%)")
    print(f"without (EN retrieval unscoreable for these): {n - matched}")
    print(f"→ {args.out}")
    if matched < n:
        print("\nNote: rows with has_en_counterpart=false must be EXCLUDED from "
              "EN retrieval scoring, not counted as misses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

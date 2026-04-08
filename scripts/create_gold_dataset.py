"""
Script to create a small gold evaluation dataset for MK RAG.

The dataset contains Macedonian factual questions with reference answers
and (optionally) relevant document IDs.

Usage:
    python scripts/create_gold_dataset.py --output data/gold_dataset.jsonl
"""

import json
import argparse
from pathlib import Path

# Small hand-crafted gold dataset for initial experiments
GOLD_SAMPLES = [
    {
        "query": "Кој е главниот град на Македонија?",
        "answer": "Скопје е главниот град на Македонија.",
        "relevant_doc_ids": [],
    },
    {
        "query": "Кога е прогласена независноста на Македонија?",
        "answer": "Македонија ја прогласи независноста на 8 септември 1991 година.",
        "relevant_doc_ids": [],
    },
    {
        "query": "Кој го напишал делото Македонска крвава свадба?",
        "answer": "Делото Македонска крвава свадба го напишал Војдан Чернодрински.",
        "relevant_doc_ids": [],
    },
    {
        "query": "Колку жители има Скопје?",
        "answer": "Скопје има околу 600.000 жители.",
        "relevant_doc_ids": [],
    },
    {
        "query": "Која е официјалната валута во Македонија?",
        "answer": "Официјалната валута на Македонија е македонскиот денар.",
        "relevant_doc_ids": [],
    },
    {
        "query": "На кој јазик зборуваат во Македонија?",
        "answer": "Во Македонија службен јазик е македонскиот јазик.",
        "relevant_doc_ids": [],
    },
    {
        "query": "Кој е Гоце Делчев?",
        "answer": "Гоце Делчев е македонски револуционер и еден од основачите на ВМОРО.",
        "relevant_doc_ids": [],
    },
    {
        "query": "Кога е основан Универзитетот Св. Кирил и Методиј?",
        "answer": "Универзитетот Св. Кирил и Методиј е основан во 1949 година.",
        "relevant_doc_ids": [],
    },
    {
        "query": "Каква клима има Македонија?",
        "answer": "Македонија има континентална клима со влијание на медитеранската клима.",
        "relevant_doc_ids": [],
    },
    {
        "query": "Кој е Крсте Петков Мисирков?",
        "answer": "Крсте Петков Мисирков е македонски јазичар и просветител, познат по книгата За македонцките работи.",
        "relevant_doc_ids": [],
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/gold_dataset.jsonl")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for sample in GOLD_SAMPLES:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Gold dataset written to {out_path} ({len(GOLD_SAMPLES)} samples)")


if __name__ == "__main__":
    main()

"""
Step 10b — Checkpointed BGE-M3 embedding for CPU-only machines.

Drop-in alternative to step 10 for slow/low-RAM boxes. Embeds chunks in shards
and saves each shard as a ``.npy`` checkpoint, so a crash or reboot resumes from
the last finished shard instead of restarting a 20+ hour run. faiss is only
imported in the final assembly step (``--assemble``), keeping it out of the
embedding process entirely (torch+faiss in one process has caused segfaults on
Windows).

The assembled index is identical to step 10's output: an ``IndexFlatIP`` over
L2-normalized fp32 vectors in chunk-file order, loadable by
``src.retrieval.dense_retriever.DenseRetriever``.

Usage:
    python scripts/mk/10b_embed_chunks_checkpointed.py                 # embed (resumable)
    python scripts/mk/10b_embed_chunks_checkpointed.py --assemble      # shards -> FAISS index
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = Path("data/processed/en_chunks.jsonl")
DEFAULT_SHARDS_DIR = Path("data/indices/faiss/en_shards")
DEFAULT_INDEX_PATH = Path("data/indices/faiss/en_bge_m3.index")
DEFAULT_MODEL = "BAAI/bge-m3"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed BGE-M3 embedding (CPU-friendly).")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu; default picks cuda when available.")
    parser.add_argument("--assemble", action="store_true",
                        help="Skip embedding; build the FAISS index from finished shards.")
    return parser.parse_args()


def load_texts(chunks_path: Path) -> list[str]:
    texts: list[str] = []
    with chunks_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def shard_path(shards_dir: Path, shard_index: int) -> Path:
    return shards_dir / f"shard_{shard_index:05d}.npy"


def embed(args: argparse.Namespace) -> None:
    import numpy as np

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch

    torch.set_num_threads(args.threads)

    texts = load_texts(args.chunks)
    total_shards = (len(texts) + args.shard_size - 1) // args.shard_size
    args.shards_dir.mkdir(parents=True, exist_ok=True)

    done = {p.name for p in args.shards_dir.glob("shard_*.npy")}
    remaining = [i for i in range(total_shards) if shard_path(args.shards_dir, i).name not in done]
    logger.info("%d chunks in %d shards; %d shards already done, %d to go",
                len(texts), total_shards, total_shards - len(remaining), len(remaining))
    if not remaining:
        logger.info("All shards finished. Run with --assemble to build the index.")
        return

    from FlagEmbedding import BGEM3FlagModel

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Embedding on device: %s", device)
    model = BGEM3FlagModel(args.model, use_fp16=False, device=device)

    started = time.time()
    for n, shard_index in enumerate(remaining, start=1):
        lo = shard_index * args.shard_size
        hi = min(lo + args.shard_size, len(texts))
        output = model.encode(texts[lo:hi], batch_size=args.batch_size, max_length=args.max_length)
        vectors = np.asarray(output["dense_vecs"], dtype=np.float32)

        # Write-then-rename so an interrupted write never counts as a done shard.
        # The temp name must end in .npy or np.save silently appends it.
        final = shard_path(args.shards_dir, shard_index)
        tmp = final.with_name(f"tmp_{final.name}")
        np.save(tmp, vectors)
        tmp.replace(final)

        rate = (time.time() - started) / n
        eta_h = rate * (len(remaining) - n) / 3600
        logger.info("shard %d/%d done (%d vecs) — %.1f min/shard, ETA %.1f h",
                    n, len(remaining), vectors.shape[0], rate / 60, eta_h)

    logger.info("Embedding complete. Run with --assemble to build the index.")


def assemble(args: argparse.Namespace) -> None:
    import faiss
    import numpy as np

    texts_count = len(load_texts(args.chunks))
    total_shards = (texts_count + args.shard_size - 1) // args.shard_size

    parts = []
    for shard_index in range(total_shards):
        path = shard_path(args.shards_dir, shard_index)
        if not path.exists():
            raise FileNotFoundError(f"Missing shard {path} — embedding is not finished.")
        parts.append(np.load(path))

    embeddings = np.vstack(parts).astype(np.float32)
    if embeddings.shape[0] != texts_count:
        raise ValueError(f"Shards hold {embeddings.shape[0]} vectors but chunks file has {texts_count}.")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    args.index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.index))
    logger.info("FAISS index (%d x %d) → %s", index.ntotal, embeddings.shape[1], args.index)


def main() -> None:
    setup_logging()
    args = parse_args()
    if args.assemble:
        assemble(args)
    else:
        embed(args)


if __name__ == "__main__":
    main()

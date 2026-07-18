from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List


DEFAULT_CHUNKS_PATH = Path("data/processed/mk_lvstck_chunks.jsonl")
DEFAULT_FAISS_INDEX_PATH = Path("data/indices/faiss/mk_lvstck_bge_m3.index")
DEFAULT_BM25_INDEX_PATH = Path("data/indices/bm25/mk_lvstck/mk_lvstck_bm25.pkl")
DEFAULT_EMBEDDING_CACHE_DIR = Path("data/indices/embeddings/mk_lvstck_bge_m3_batches")
DEFAULT_MODEL_NAME = "BAAI/bge-m3"


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} on line {line_number}, "
                    f"got {type(record).__name__}"
                )

            yield record


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def validate_chunks(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        raise ValueError("No chunks found")

    required_fields = ["chunk_id", "doc_id", "text", "language", "source"]
    seen_chunk_ids = set()

    for index, chunk in enumerate(chunks, start=1):
        for field in required_fields:
            if field not in chunk or chunk[field] is None or str(chunk[field]).strip() == "":
                raise ValueError(f"Chunk {index} is missing required field: {field}")

        chunk_id = str(chunk["chunk_id"])

        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")

        seen_chunk_ids.add(chunk_id)

        if chunk["language"] != "mk":
            raise ValueError(f"Chunk {chunk_id} has language={chunk['language']!r}")

        if chunk["source"] != "lvstck":
            raise ValueError(f"Chunk {chunk_id} has source={chunk['source']!r}")


def require_module(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise SystemExit(
            f"Missing dependency '{module_name}'. Install it first: {install_hint}"
        )


def build_faiss_index(
    *,
    chunks_path: Path,
    index_path: Path,
    embedding_cache_dir: Path,
    model_name: str,
    batch_size: int,
    use_fp16: bool,
    force: bool,
) -> None:
    require_module("faiss", "python -m pip install faiss-cpu")
    require_module("FlagEmbedding", "python -m pip install FlagEmbedding")
    require_module("numpy", "python -m pip install numpy")

    import faiss
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel
    from tqdm import tqdm

    if index_path.exists() and not force:
        index = faiss.read_index(str(index_path))
        logger.info(
            "FAISS index already exists: %s (vectors=%s, dim=%s)",
            index_path,
            index.ntotal,
            index.d,
        )
        return

    chunks = load_chunks(chunks_path)

    logger.info("Building FAISS dense index")
    logger.info("Chunks: %s", chunks_path)
    logger.info("Output: %s", index_path)
    logger.info("Embedding cache: %s", embedding_cache_dir)
    logger.info("Model: %s", model_name)
    logger.info("Batch size: %s", batch_size)

    texts = [chunk["text"] for chunk in chunks]
    embed_model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    all_batch_paths = []

    for batch_number, start in enumerate(
        tqdm(range(0, len(texts), batch_size), desc="Embedding")
    ):
        batch = texts[start : start + batch_size]
        batch_path = embedding_cache_dir / f"batch_{batch_number:05d}.npy"
        all_batch_paths.append(batch_path)

        if batch_path.exists() and not force:
            try:
                existing = np.load(batch_path, mmap_mode="r")

                if existing.shape[0] == len(batch):
                    continue

                logger.warning(
                    "Recomputing batch %s because cached row count is %s, expected %s",
                    batch_number,
                    existing.shape[0],
                    len(batch),
                )
            except Exception as error:
                logger.warning("Recomputing unreadable cached batch %s: %s", batch_path, error)

        output = embed_model.encode(batch, batch_size=batch_size, max_length=512)
        batch_embeddings = output["dense_vecs"].astype(np.float32)

        tmp_path = batch_path.with_suffix(".tmp.npy")
        np.save(tmp_path, batch_embeddings)
        tmp_path.replace(batch_path)

    missing_batches = [path for path in all_batch_paths if not path.exists()]

    if missing_batches:
        raise RuntimeError(
            f"Missing {len(missing_batches)} embedding batch file(s); "
            "rerun the command to resume."
        )

    logger.info("Loading cached embedding batches")
    embeddings = np.vstack([np.load(path) for path in all_batch_paths]).astype(np.float32)

    if embeddings.shape[0] != len(chunks):
        raise RuntimeError(
            f"Embedding row count mismatch: got {embeddings.shape[0]}, "
            f"expected {len(chunks)}"
        )

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    metadata_path = index_path.with_suffix(".metadata.json")

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "chunks_path": str(chunks_path),
                "index_path": str(index_path),
                "embedding_cache_dir": str(embedding_cache_dir),
                "model_name": model_name,
                "vector_count": int(index.ntotal),
                "dimension": int(index.d),
                "batch_size": batch_size,
                "use_fp16": use_fp16,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    index = faiss.read_index(str(index_path))
    logger.info(
        "FAISS index written: %s (vectors=%s, dim=%s)",
        index_path,
        index.ntotal,
        index.d,
    )
    logger.info("FAISS metadata written: %s", metadata_path)


def build_bm25_index(
    *,
    chunks_path: Path,
    index_path: Path,
    force: bool,
) -> None:
    require_module("rank_bm25", "python -m pip install rank-bm25")

    if index_path.exists() and not force:
        with index_path.open("rb") as file:
            data = pickle.load(file)

        corpus = data.get("corpus", [])
        logger.info("BM25 index already exists: %s (chunks=%s)", index_path, len(corpus))
        return

    from rank_bm25 import BM25Okapi

    logger.info("Building BM25 index")
    logger.info("Chunks: %s", chunks_path)
    logger.info("Output: %s", index_path)

    chunks = load_chunks(chunks_path)
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    index_path.parent.mkdir(parents=True, exist_ok=True)

    with index_path.open("wb") as file:
        pickle.dump({"bm25": bm25, "corpus": chunks}, file)

    with index_path.open("rb") as file:
        data = pickle.load(file)

    logger.info(
        "BM25 index written: %s (chunks=%s)",
        index_path,
        len(data.get("corpus", [])),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build retrieval indices for the LVSTCK Macedonian chunks: "
            "BGE-M3 dense embeddings in FAISS and BM25 for sparse/hybrid retrieval."
        )
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help=f"Input LVSTCK chunks JSONL. Default: {DEFAULT_CHUNKS_PATH}",
    )
    parser.add_argument(
        "--faiss-index",
        type=Path,
        default=DEFAULT_FAISS_INDEX_PATH,
        help=f"Output FAISS index. Default: {DEFAULT_FAISS_INDEX_PATH}",
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=DEFAULT_BM25_INDEX_PATH,
        help=f"Output BM25 pickle. Default: {DEFAULT_BM25_INDEX_PATH}",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_CACHE_DIR,
        help=(
            "Directory for cached dense embedding batches. "
            f"Default: {DEFAULT_EMBEDDING_CACHE_DIR}"
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Embedding model name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size. Lower this if memory is limited.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing indices.",
    )
    parser.add_argument(
        "--skip-faiss",
        action="store_true",
        help="Do not build the dense FAISS index.",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Do not build the BM25 index.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 model inference.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.chunks.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {args.chunks}")

    chunks = load_chunks(args.chunks)
    validate_chunks(chunks)
    logger.info("Validated LVSTCK chunks: %s", len(chunks))

    if not args.skip_faiss:
        build_faiss_index(
            chunks_path=args.chunks,
            index_path=args.faiss_index,
            embedding_cache_dir=args.embedding_cache_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            use_fp16=not args.no_fp16,
            force=args.force,
        )

    if not args.skip_bm25:
        build_bm25_index(
            chunks_path=args.chunks,
            index_path=args.bm25_index,
            force=args.force,
        )

    logger.info("LVSTCK index build finished successfully")


if __name__ == "__main__":
    main()

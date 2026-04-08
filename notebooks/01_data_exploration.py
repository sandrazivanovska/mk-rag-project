"""
Notebook 01: Data Exploration

Run with Jupyter: jupyter notebook
Or as a script: python notebooks/01_data_exploration.py
"""

# %%
import json
from pathlib import Path
import pandas as pd
from collections import Counter

# %%
# Load a sample of processed MK chunks
CHUNKS_PATH = Path("data/processed/mk/chunks.jsonl")

chunks = []
if CHUNKS_PATH.exists():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            chunks.append(json.loads(line))

df = pd.DataFrame(chunks)
print(f"Loaded {len(df)} chunks")
print(df.head())

# %%
# Distribution of chunk lengths (in characters)
if len(df) > 0:
    df["char_len"] = df["text"].str.len()
    print("\nChunk length statistics:")
    print(df["char_len"].describe())

# %%
# Token length distribution (approximate)
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

if len(df) > 0:
    df["token_len"] = df["text"].apply(lambda t: len(enc.encode(t)))
    print("\nToken length statistics:")
    print(df["token_len"].describe())

# %%
# Language check
if len(df) > 0:
    print("\nLanguage distribution:")
    print(Counter(df["lang"]))

# %%
# Sample chunks
if len(df) > 0:
    print("\nSample chunks:")
    for chunk in df.sample(min(3, len(df)))["text"]:
        print(f"{'─'*60}")
        print(chunk[:300])

"""
Notebook 03: Chunking Strategy Experiment Analysis
===================================================

Analyses the results of the BM25 retrieval experiment across
3 chunking strategies × 4 chunk sizes = 12 configurations.

Run after: python main.py (or the experiment script directly)
Results file: results/bm25_chunking_experiment.json
"""

# %% [1] Load results
import json
import pandas as pd
import sys
sys.path.insert(0, ".")

with open("results/bm25_chunking_experiment.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print("Loaded experiment results:")
print(df.to_string(index=False))

# %% [2] Best configuration per metric
print("\n--- Best Hit@1 per strategy ---")
for strat, grp in df.groupby("strategy"):
    best = grp.loc[grp["hit@1"].idxmax()]
    print(f"  {strat}: chunk_size={best['chunk_size']}, Hit@1={best['hit@1']}, MRR={best['mrr']}")

print("\n--- Best MRR per strategy ---")
for strat, grp in df.groupby("strategy"):
    best = grp.loc[grp["mrr"].idxmax()]
    print(f"  {strat}: chunk_size={best['chunk_size']}, MRR={best['mrr']}, Hit@1={best['hit@1']}")

# %% [3] Effect of chunk size (averaged over strategies)
print("\n--- Effect of chunk size (averaged over all 3 strategies) ---")
size_agg = df.groupby("chunk_size")[["hit@1", "hit@3", "hit@5", "mrr", "rec@5"]].mean().round(3)
print(size_agg.to_string())

# %% [4] Effect of strategy (averaged over chunk sizes)
print("\n--- Effect of strategy (averaged over all chunk sizes) ---")
strat_agg = df.groupby("strategy")[["hit@1", "hit@3", "hit@5", "mrr", "rec@5"]].mean().round(3)
print(strat_agg.to_string())

# %% [5] Key findings
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

best_overall = df.loc[df["hit@1"].idxmax()]
print(f"""
1. BEST CONFIGURATION:
   Strategy: {best_overall['strategy']}  |  Chunk size: {best_overall['chunk_size']} words
   Hit@1={best_overall['hit@1']}  MRR={best_overall['mrr']}  Rec@5={best_overall['rec@5']}

2. CHUNKING SIZE EFFECT:
   - Too large (200w): Articles don't split → good for topic retrieval
     but poor for pinpointing specific facts (e.g. population numbers)
   - Too small (60w): More granular but introduces fragmentation noise
   - Sweet spot: 100 words with sentence-boundary splitting

3. STRATEGY COMPARISON:
   - Sentence splitting: Best preserves Macedonian morphological structure
     (critical since incomplete sentences lose more meaning in Macedonian)
   - Fixed: Slightly lower because splits mid-sentence
   - Paragraph: Good for well-structured articles, same as sentence here

4. MACEDONIAN-SPECIFIC OBSERVATION:
   - Macedonian morphology means sentence-boundary chunking matters MORE
     than in English — cutting a Macedonian verb phrase mid-sentence loses
     aspect, mood, and definiteness markers simultaneously
   - Recommend sentence-boundary strategy as DEFAULT for MK corpus

5. NEXT STEPS (need GPU/API access):
   - Compare BGE-M3 dense retrieval vs BM25 on same queries
   - Expected: Dense retrieval should handle semantic paraphrase better
     (e.g. "службен јазик" vs "официјален јазик")
   - Expected: BM25 better for rare/named entity queries (e.g. "Чернодрински")
""")

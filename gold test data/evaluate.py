import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.unified_retriever import UnifiedRetriever
from interface.app import find_timestamp_for_text, load_mapping

# -------------------------
# CONFIG
# -------------------------
GOLD_PATH = "gold test data/gold.json"
MAPPING_PATH = "data/mappings/frame_text_mapping.json"
RETRIEVER_OPTIONS = ["bm25", "tfidf", "faiss", "pgvector_ivfflat", "pgvector_hnsw"]
REPEATS = 3
TOP_K = 1
RESULTS_CSV = "gold test data/evaluation_results.csv"
RESULTS_PLOT = "gold test data/retrieval_performance.png"

# -------------------------
# Load gold questions
# -------------------------
with open(GOLD_PATH, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

mapping = load_mapping(MAPPING_PATH)

# Split answerable vs unanswerable
answerable_questions = [q for q in gold_data if q["answer"] is not None]
unanswerable_questions = [q for q in gold_data if q["answer"] is None]

# -------------------------
# Evaluation
# -------------------------
results = []

for retriever_type in RETRIEVER_OPTIONS:
    print(f"[INFO] Evaluating {retriever_type}...")
    retriever = UnifiedRetriever(
        use_faiss="faiss" in retriever_type,
        use_pgvector="pgvector" in retriever_type,
        pg_method="ivfflat" if "ivfflat" in retriever_type else ("hnsw" if "hnsw" in retriever_type else None),
        use_multimodal="multimodal" in retriever_type
    )

    correct = 0
    false_positive = 0
    latencies = []

    # Evaluate answerable
    for q in answerable_questions:
        start = time.time()
        results_top = retriever.search(q["question"], retriever_type=retriever_type, top_k=TOP_K)
        latencies.append(time.time() - start)

        if results_top:
            result_text, _ = results_top[0]

            # Skip placeholder responses for rejected queries
            if "No answer found" in result_text:
                continue

            predicted_ts = find_timestamp_for_text(result_text, mapping)

            # Only evaluate if a valid timestamp was found
            if predicted_ts is not None and abs(predicted_ts - q["answer"]) <= 10:
                correct += 1


    # Evaluate unanswerable
    for q in unanswerable_questions:
        start = time.time()
        results_top = retriever.search(q["question"], retriever_type=retriever_type, top_k=TOP_K)
        latencies.append(time.time() - start)

        if results_top:
            result_text, _ = results_top[0]
            
            # Count false positives only if the system did *not* reject the query
            if "No answer found" not in result_text:
                false_positive += 1


    total_answerable = len(answerable_questions)
    total_unanswerable = len(unanswerable_questions)

    results.append({
        "retriever": retriever_type,
        "accuracy": correct / total_answerable if total_answerable else 0,
        "rejection_quality": 1 - false_positive / total_unanswerable if total_unanswerable else 0,
        "avg_latency_sec": sum(latencies) / len(latencies) if latencies else 0
    })

# -------------------------
# Save and plot results
# -------------------------
df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)

print("\n[INFO] Evaluation complete. Summary:")
print(df)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

ax2 = ax1.twinx()
df.plot(kind='bar', x='retriever', y='accuracy', ax=ax1, color='skyblue', position=0, width=0.3, label="Accuracy")
df.plot(kind='bar', x='retriever', y='rejection_quality', ax=ax1, color='lightgreen', position=1, width=0.3, label="Rejection Quality")
df.plot(kind='line', x='retriever', y='avg_latency_sec', ax=ax2, color='red', marker='o', label="Avg Latency (s)")

ax1.set_ylabel("Accuracy / Rejection Quality")
ax2.set_ylabel("Avg Latency (seconds)")
ax1.set_title("Retrieval Method Comparison")
ax1.set_xticklabels(df['retriever'], rotation=45)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(RESULTS_PLOT)
plt.show()

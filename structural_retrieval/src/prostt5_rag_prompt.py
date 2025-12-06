"""
RAG prompt construction using Hybrid RRF (ProstT5 + ESM-2) embeddings.
This script creates prompts for RAPM by retrieving similar proteins
using Reciprocal Rank Fusion combining:
- ProstT5 structure-aware embeddings (1024-dim)
- ESM-2 sequence-based embeddings (1280-dim)
"""
import os
import json
import time
import tempfile
import numpy as np
import faiss
import sys
from tqdm import tqdm
import torch
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(__file__))
from prostt5_features import extract_features, reciprocal_rank_fusion, adaptive_weighted_rrf


def score_to_confidence(score):
    """Convert score to confidence level."""
    if score >= 90:
        return "<High Confidence>"
    elif score >= 60:
        return "<Medium Confidence>"
    else:
        return "<Low Confidence>"


def RAG_prompt_construction(db_seqs, db_labels, db_prostt5_features, db_esm2_features, train_labels, test_insts, test_seqs, test_labels, test_metas, task_name, topk, prostt5_index, esm2_index, output_dir):
    """
    Construct RAG prompts using Hybrid RRF (ProstT5 + ESM-2) retrieval.

    Args:
        db_seqs: All training sequences across tasks
        db_labels: All training labels (metadata) across tasks
        db_prostt5_features: All training ProstT5 features (1024-dim)
        db_esm2_features: All training ESM-2 features (1280-dim)
        train_labels: Current task's training labels (descriptions)
        test_insts: Test instructions
        test_seqs: Test sequences
        test_labels: Test labels (descriptions)
        test_metas: Test metadata
        task_name: Name of current task
        topk: Number of top retrievals
        prostt5_index: Pre-built ProstT5 Faiss index (or None to build new)
        esm2_index: Pre-built ESM-2 Faiss index (or None to build new)
        output_dir: Directory to save output files
    """
    # --- Build ProstT5 Faiss Index ---
    if prostt5_index is None:
        d = db_prostt5_features.shape[1]
        prostt5_index = faiss.IndexHNSWFlat(d, 32)
        db_prostt5_norm = db_prostt5_features / np.linalg.norm(db_prostt5_features, axis=1, keepdims=True)
        prostt5_index.add(db_prostt5_norm.astype(np.float32))
        prostt5_index.hnsw.efSearch = max(50, topk * 2)

    # --- Build ESM-2 Faiss Index ---
    if esm2_index is None:
        d = db_esm2_features.shape[1]
        esm2_index = faiss.IndexHNSWFlat(d, 32)
        db_esm2_norm = db_esm2_features / np.linalg.norm(db_esm2_features, axis=1, keepdims=True)
        esm2_index.add(db_esm2_norm.astype(np.float32))
        esm2_index.hnsw.efSearch = max(50, topk * 2)

    # Load test features
    test_prostt5 = np.load(os.path.join(output_dir, f"hybrid_{task_name}_test_prostt5.npy"))
    test_esm2 = np.load(os.path.join(output_dir, f"hybrid_{task_name}_test_esm2.npy"))

    # Normalize features
    test_prostt5_norm = test_prostt5 / np.linalg.norm(test_prostt5, axis=1, keepdims=True)
    test_esm2_norm = test_esm2 / np.linalg.norm(test_esm2, axis=1, keepdims=True)

    # Search with ProstT5
    st_time = time.time()
    D_prostt5, I_prostt5 = prostt5_index.search(test_prostt5_norm.astype(np.float32), topk)
    print(f"ProstT5 Faiss HNSW search time: {time.time() - st_time:.4f} seconds")

    # Search with ESM-2
    st_time = time.time()
    D_esm2, I_esm2 = esm2_index.search(test_esm2_norm.astype(np.float32), topk)
    print(f"ESM-2 Faiss HNSW search time: {time.time() - st_time:.4f} seconds")

    # Perform standard RRF fusion for each test sample
    print(f"Performing standard RRF fusion for {len(test_prostt5)} test samples...")
    print(f"  Using balanced weights: k_prostt5=60, k_esm2=60")
    rrf_results = []

    for i in tqdm(range(len(test_prostt5)), desc="RRF fusion for prompts"):
        # Get rankings from both methods
        prostt5_ranking = [(I_prostt5[i][j], D_prostt5[i][j]) for j in range(topk)]
        esm2_ranking = [(I_esm2[i][j], D_esm2[i][j]) for j in range(topk)]

        # Apply standard balanced RRF
        fused_ranking, weights_used = adaptive_weighted_rrf(
            [prostt5_ranking, esm2_ranking],
            instruction=test_insts[i],
            k_base=60,
            weight_adaptation='balanced'  # Balanced weights
        )

        # Get top-k from fused results
        rrf_topk = []
        for idx, rrf_score in fused_ranking[:topk]:
            rrf_topk.append({
                "db_label": db_labels[idx],
                "rrf_score": rrf_score
            })
        rrf_results.append(rrf_topk)

    # Build Faiss index on current task's training set for examples
    train_prostt5 = np.load(os.path.join(output_dir, f"hybrid_{task_name}_train_prostt5.npy"))
    train_prostt5_norm = train_prostt5 / np.linalg.norm(train_prostt5, axis=1, keepdims=True)
    train_faiss_index = faiss.IndexHNSWFlat(train_prostt5_norm.shape[1], 32)
    train_faiss_index.hnsw.efSearch = max(50, topk * 2)
    train_faiss_index.add(train_prostt5_norm.astype(np.float32))
    train_D, train_I = train_faiss_index.search(train_prostt5_norm.astype(np.float32), topk)
    train_faiss_results = []
    for idxs, scores in zip(train_I, train_D):
        topk_list = []
        for idx, score in zip(idxs, scores):
            topk_list.append({
                "train_seqs_label": train_labels[idx],
                "confidence_level": score_to_confidence(score * 100),
            })
        train_faiss_results.append(topk_list)

    output_json = []
    for i in range(len(test_insts)):
        # Use RRF results (already top-k)
        retrieved_info = []
        for item in rrf_results[i]:
            retrieved_info.append({
                "db_label": item["db_label"],
                "confidence level": score_to_confidence(item["rrf_score"] * 100),
                "rrf_score": item["rrf_score"]
            })

        train_examples = []
        for item in train_faiss_results[i]:
            train_examples.append({
                "example answer": item["train_seqs_label"],
                "confidence level": item["confidence_level"]
            })

        rag_prompt = {
            "instructions": test_insts[i],
            "sequence": test_seqs[i],
            "labels": test_labels[i],
            "meta_label": test_metas[i],
            "RAG_prompt": (
                f"You are given a protein sequence and a list of related proteins retrieved from a database.\n"
                f"Instruction: {test_insts[i]}\n"
                f"Protein sequence: {test_seqs[i]}\n"
                f"Retrieved proteins annotations by RRF (ProstT5 + ESM-2): {retrieved_info}\n"
                f"Here are some example input-output pairs for this task:\n"
                f"{train_examples}\n"
                "Based on the instruction, the protein sequence, the retrieved information, and the examples, "
                "output ONLY the functional description of this protein in the following JSON format:\n"
                '{"description": "..."}'
                "\nDo not output any other text or explanation. Only output the JSON answer."
            )
        }
        output_json.append(rag_prompt)

    output_path = os.path.join(output_dir, f"HYBRID_RRF_{task_name}_RAP_Top_{topk}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    print(f"Saved RAG prompts to: {output_path}")


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python prostt5_rag_prompt.py <dataset_path> <top_k>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    top_k = int(sys.argv[2])

    # Get script directory for output
    script_dir = os.path.dirname(os.path.abspath(__file__))

    all_train_seqs = []
    all_train_labels = []
    all_train_prostt5_features = []
    all_train_esm2_features = []

    print("="*80)
    print("Hybrid RRF RAG Prompt Construction")
    print(f"Output directory: {script_dir}")
    print("="*80)

    # Collect all training data across tasks
    for p in sorted(os.listdir(dataset_path)):
        now_task = p[:-5]
        if not p.endswith(".json"):
            continue

        print(f"\n=== Loading task: {now_task} ===")
        JSON_PATH = os.path.join(dataset_path, p)
        dic = json.load(open(JSON_PATH, "r"))

        # Load ProstT5 and ESM-2 features
        try:
            now_train_prostt5 = np.load(os.path.join(script_dir, f"hybrid_{now_task}_train_prostt5.npy"))
            now_train_esm2 = np.load(os.path.join(script_dir, f"hybrid_{now_task}_train_esm2.npy"))
        except FileNotFoundError:
            print(f"Hybrid feature files for {now_task} not found!")
            print(f"Please run prostt5_retrieval.py first to generate hybrid features.")
            sys.exit(1)

        now_train_seqs = [d["sequence"] for d in dic if d['split'] == 'train']
        now_train_labels = [d["metadata"] for d in dic if d['split'] == 'train']
        all_train_seqs.extend(now_train_seqs)
        all_train_labels.extend(now_train_labels)
        all_train_prostt5_features.extend(now_train_prostt5)
        all_train_esm2_features.extend(now_train_esm2)

    print(f"\nTotal training samples before aggregation: {len(all_train_labels)}")

    # Feature Aggregation: Average features for duplicate labels
    label_to_prostt5 = defaultdict(list)
    label_to_esm2 = defaultdict(list)
    for prostt5_feat, esm2_feat, label in zip(all_train_prostt5_features, all_train_esm2_features, all_train_labels):
        label_to_prostt5[label].append(prostt5_feat)
        label_to_esm2[label].append(esm2_feat)

    new_all_train_prostt5 = []
    new_all_train_esm2 = []
    new_all_train_labels = []
    for label in label_to_prostt5.keys():
        if len(label_to_prostt5[label]) == 1:
            continue
        prostt5_feats = np.stack(label_to_prostt5[label])
        esm2_feats = np.stack(label_to_esm2[label])
        mean_prostt5 = prostt5_feats.mean(axis=0)
        mean_esm2 = esm2_feats.mean(axis=0)
        new_all_train_prostt5.append(mean_prostt5)
        new_all_train_esm2.append(mean_esm2)
        new_all_train_labels.append(label)

    all_train_prostt5_features = np.vstack([np.array(all_train_prostt5_features), np.array(new_all_train_prostt5)])
    all_train_esm2_features = np.vstack([np.array(all_train_esm2_features), np.array(new_all_train_esm2)])
    all_train_labels = all_train_labels + new_all_train_labels

    print(f"Total training samples after aggregation: {len(all_train_labels)}")

    # Generate RAG prompts for each task
    for p in sorted(os.listdir(dataset_path)):
        now_task = p[:-5]
        if not p.endswith(".json"):
            continue

        print(f"\n{'='*60}")
        print(f"Generating RAG prompts for: {now_task}")
        print(f"{'='*60}")

        JSON_PATH = os.path.join(dataset_path, p)
        dic = json.load(open(JSON_PATH, "r"))
        now_test_instructions = [d["instruction"] for d in dic if d['split'] == 'test']
        now_test_seqs = [d["sequence"] for d in dic if d['split'] == 'test']
        now_test_labels = [d["description"] for d in dic if d['split'] == 'test']
        now_test_meta = [d["metadata"] for d in dic if d['split'] == 'test']

        now_train_seqs = [d["sequence"] for d in dic if d['split'] == 'train']
        now_train_labels = [d["description"] for d in dic if d['split'] == 'train']

        RAG_prompt_construction(db_seqs=all_train_seqs,
                               db_labels=all_train_labels,
                               db_prostt5_features=all_train_prostt5_features,
                               db_esm2_features=all_train_esm2_features,
                               train_labels=now_train_labels,
                               test_insts=now_test_instructions,
                               test_seqs=now_test_seqs,
                               test_labels=now_test_labels,
                               test_metas=now_test_meta,
                               task_name=now_task,
                               topk=top_k,
                               prostt5_index=None,
                               esm2_index=None,
                               output_dir=script_dir)

    print(f"\n{'='*60}")
    print("RAG prompt construction complete!")
    print(f"{'='*60}")

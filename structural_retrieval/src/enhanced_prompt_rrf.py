"""
Enhanced RAG Prompt Construction for Hybrid RRF RAPM
=====================================================

This script applies enhanced prompt engineering to the Hybrid RRF (ProstT5 + ESM-2) method:
1. Same RRF retrieval fusion strategy (proven to work well)
2. Enhanced prompt structure with task-specific guidance
3. Confidence-based presentation of retrieved proteins
4. Explicit terminology-focused instructions
5. Better formatting and JSON output enforcement

Key Improvements over prostt5_rag_prompt.py:
- Task-specific instructions (from get_task_specific_instructions())
- Confidence-level grouping (High/Medium/Low based on RRF scores)
- Structured prompt with emoji indicators for clarity
- Terminology emphasis to improve Meta-BLEU scores
- Better few-shot example presentation

Usage:
    python enhanced_prompt_rrf.py <dataset_path> <top_k>

Example:
    python enhanced_prompt_rrf.py ../dataset 10
"""

import os
import json
import time
import numpy as np
import faiss
import sys
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(__file__))
from prostt5_features import adaptive_weighted_rrf


# ===== Task-Specific Prompt Templates =====

def get_task_specific_instructions(task_name):
    """
    Get task-specific instructions to guide LLM toward better terminology usage.

    Args:
        task_name: Task name (without _OOD suffix)

    Returns:
        String with task-specific instructions
    """
    task_instructions = {
        "catalytic_activity": """
**Task-Specific Guidance for Catalytic Activity Prediction**:
- Focus on SPECIFIC catalytic mechanisms (e.g., "acid-base catalysis", "nucleophilic substitution")
- Identify the ENZYME CLASS (e.g., "serine protease", "zinc metalloenzyme", "ATP-dependent kinase")
- Specify SUBSTRATE and COFACTOR requirements (e.g., "ATP-binding", "requires Mg2+", "acts on peptide bonds")
- Mention ACTIVE SITE residues if relevant (e.g., "catalytic triad Ser-His-Asp")
- Use EC number terminology when applicable (e.g., "hydrolase", "transferase", "oxidoreductase")
""",
        "domain_motif": """
**Task-Specific Guidance for Domain/Motif Prediction**:
- Identify SPECIFIC DOMAINS by name (e.g., "SH3 domain", "zinc finger", "WD40 repeat")
- Describe STRUCTURAL MOTIFS (e.g., "helix-turn-helix", "beta-barrel", "leucine zipper")
- Mention BINDING SITES (e.g., "DNA-binding domain", "ATP-binding motif", "protein-protein interaction domain")
- Use InterPro/Pfam terminology when relevant
- Specify domain ORGANIZATION and ARCHITECTURE
""",
        "protein_function": """
**Task-Specific Guidance for Protein Function Prediction**:
- Use MOLECULAR FUNCTION terms from Gene Ontology (GO:MF)
- Specify BIOLOGICAL PROCESSES involved (GO:BP)
- Identify PROTEIN FAMILY or superfamily (e.g., "GPCR family", "immunoglobulin superfamily")
- Describe functional KEYWORDS (e.g., "transcription factor", "signal transduction", "cell cycle regulation")
- Be PRECISE with technical terminology (avoid vague terms like "important" or "involved")
""",
        "general_function": """
**Task-Specific Guidance for General Function Prediction**:
- Provide BROAD FUNCTIONAL CATEGORIES (e.g., "metabolic enzyme", "structural protein", "regulatory protein")
- Mention CELLULAR LOCATION if relevant (e.g., "membrane-bound", "cytoplasmic", "secreted")
- Describe BIOLOGICAL ROLE (e.g., "immune response", "development", "homeostasis")
- Include PATHWAY involvement (e.g., "glycolysis", "signal transduction", "apoptosis")
- Use standard biological terminology, avoid colloquialisms
"""
    }

    return task_instructions.get(task_name, "Use precise biological terminology in your prediction.")


def score_to_confidence_level(score):
    """
    Convert RRF score to confidence level with thresholds.

    RRF scores are typically in [0, 1] range after normalization.
    """
    if score >= 0.85:
        return "High"
    elif score >= 0.65:
        return "Medium"
    else:
        return "Low"


def enhanced_RAG_prompt_construction(db_seqs, db_labels, db_prostt5_features, db_esm2_features,
                                      train_labels, test_insts, test_seqs, test_labels, test_metas,
                                      task_name, topk, prostt5_index, esm2_index, output_dir):
    """
    Construct enhanced RAG prompts using Hybrid RRF (ProstT5 + ESM-2) retrieval.

    This function applies enhanced prompt engineering to the RRF method:
    - Task-specific instructions
    - Confidence-based grouping
    - Better formatting and structure
    - Terminology-focused guidance

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

    # Load test features from parent directory
    parent_dir = os.path.dirname(output_dir)  # Get parent if output_dir is src/
    feature_dir = parent_dir if os.path.basename(output_dir) == 'src' else output_dir
    test_prostt5 = np.load(os.path.join(feature_dir, f"hybrid_{task_name}_test_prostt5.npy"))
    test_esm2 = np.load(os.path.join(feature_dir, f"hybrid_{task_name}_test_esm2.npy"))

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

    # Perform RRF fusion for each test sample
    print(f"Performing RRF fusion for {len(test_prostt5)} test samples...")
    print(f"  Using balanced weights: k_prostt5=60, k_esm2=60")
    rrf_results = []

    for i in tqdm(range(len(test_prostt5)), desc="RRF fusion"):
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
                "rrf_score": rrf_score,
                "confidence": score_to_confidence_level(rrf_score)
            })
        rrf_results.append(rrf_topk)

    # Build Faiss index on current task's training set for examples
    train_prostt5 = np.load(os.path.join(feature_dir, f"hybrid_{task_name}_train_prostt5.npy"))
    train_prostt5_norm = train_prostt5 / np.linalg.norm(train_prostt5, axis=1, keepdims=True)
    train_faiss_index = faiss.IndexHNSWFlat(train_prostt5_norm.shape[1], 32)
    train_faiss_index.hnsw.efSearch = max(50, topk * 2)
    train_faiss_index.add(train_prostt5_norm.astype(np.float32))
    train_D, train_I = train_faiss_index.search(train_prostt5_norm.astype(np.float32), topk)

    # Get task-specific instructions
    task_base = task_name.replace('_OOD', '')
    task_guidance = get_task_specific_instructions(task_base)

    # Generate enhanced prompts
    print(f"\nGenerating enhanced prompts for {len(test_insts)} test samples...")
    output_json = []

    for i in tqdm(range(len(test_insts)), desc="Generating prompts"):
        # Group retrieved proteins by confidence level
        high_conf = []
        medium_conf = []
        low_conf = []

        for item in rrf_results[i]:
            if item["confidence"] == "High":
                high_conf.append(item["db_label"])
            elif item["confidence"] == "Medium":
                medium_conf.append(item["db_label"])
            else:
                low_conf.append(item["db_label"])

        # Get training examples (top 5 similar from in-task training set)
        train_examples = []
        for idx in train_I[i][:5]:
            train_examples.append(train_labels[idx])

        # Build enhanced prompt
        rag_prompt = f"""You are a protein function prediction expert with deep knowledge of biological terminology.

**Task**: {test_insts[i]}

**Query Protein Sequence**:
{test_seqs[i]}

**Retrieved Similar Proteins (Hybrid RRF: ProstT5 + ESM-2)**:

🟢 **High Confidence Matches (RRF score ≥ 0.85)**:
{chr(10).join([f"  • {ann}" for ann in high_conf]) if high_conf else "  None"}

🟡 **Medium Confidence Matches (0.65 ≤ RRF score < 0.85)**:
{chr(10).join([f"  • {ann}" for ann in medium_conf]) if medium_conf else "  None"}

🔴 **Lower Confidence Matches (RRF score < 0.65)**:
{chr(10).join([f"  • {ann}" for ann in low_conf]) if low_conf else "  None"}

**In-Task Training Examples** (for format reference):
{chr(10).join([f"  • {ex}" for ex in train_examples[:3]])}

{task_guidance}

**IMPORTANT INSTRUCTIONS**:
1. **Use PRECISE biological terminology** from the retrieved annotations
2. **Prioritize high-confidence matches** - they are most similar (based on RRF fusion)
3. **Extract domain-specific terms** (enzyme names, GO terms, motifs, etc.)
4. **Avoid generic descriptions** - be specific
5. **Match the terminology style** of the training examples
6. **Be concise but technically accurate**

Output ONLY the functional description in JSON format:
{{"description": "..."}}

Do not include explanations, justifications, or any other text. Only the JSON answer."""

        output_json.append({
            "instructions": test_insts[i],
            "sequence": test_seqs[i],
            "labels": test_labels[i],
            "meta_label": test_metas[i],
            "RAG_prompt": rag_prompt
        })

    # Save prompts to parent directory (where other results are stored)
    output_path = os.path.join(feature_dir, f"ENHANCED_RRF_{task_name}_RAP_Top_{topk}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved enhanced RRF prompts to: {output_path}")
    return output_path


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python enhanced_prompt_rrf.py <dataset_path> <top_k>")
        print("Example: python enhanced_prompt_rrf.py ../dataset 10")
        sys.exit(1)

    dataset_path = sys.argv[1]
    top_k = int(sys.argv[2])

    # Get script directory and parent directory (where features are stored)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # structural_retrieval/

    all_train_seqs = []
    all_train_labels = []
    all_train_prostt5_features = []
    all_train_esm2_features = []

    print("="*80)
    print("Enhanced Hybrid RRF RAG Prompt Construction")
    print(f"Output directory: {parent_dir}")
    print(f"Top-K: {top_k}")
    print("="*80)

    # Collect all training data across tasks
    for p in sorted(os.listdir(dataset_path)):
        now_task = p[:-5]
        if not p.endswith(".json"):
            continue

        print(f"\n=== Loading task: {now_task} ===")
        JSON_PATH = os.path.join(dataset_path, p)
        dic = json.load(open(JSON_PATH, "r"))

        # Load ProstT5 and ESM-2 features from parent directory
        try:
            now_train_prostt5 = np.load(os.path.join(parent_dir, f"hybrid_{now_task}_train_prostt5.npy"))
            now_train_esm2 = np.load(os.path.join(parent_dir, f"hybrid_{now_task}_train_esm2.npy"))
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

    # Generate enhanced RAG prompts for each task
    for p in sorted(os.listdir(dataset_path)):
        now_task = p[:-5]
        if not p.endswith(".json"):
            continue

        print(f"\n{'='*60}")
        print(f"Generating enhanced prompts for: {now_task}")
        print(f"{'='*60}")

        JSON_PATH = os.path.join(dataset_path, p)
        dic = json.load(open(JSON_PATH, "r"))
        now_test_instructions = [d["instruction"] for d in dic if d['split'] == 'test']
        now_test_seqs = [d["sequence"] for d in dic if d['split'] == 'test']
        now_test_labels = [d["description"] for d in dic if d['split'] == 'test']
        now_test_meta = [d["metadata"] for d in dic if d['split'] == 'test']

        now_train_seqs = [d["sequence"] for d in dic if d['split'] == 'train']
        now_train_labels = [d["description"] for d in dic if d['split'] == 'train']

        enhanced_RAG_prompt_construction(
            db_seqs=all_train_seqs,
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
            output_dir=parent_dir
        )

    print(f"\n{'='*80}")
    print("Enhanced RRF Prompt Construction Complete!")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("1. Run inference with: python run_prostt5_rapm.py <task_name> <top_k>")
    print("   (Make sure to update run_prostt5_rapm.py to load ENHANCED_RRF_* files)")
    print("2. Compare results with baseline RRF method")
    print(f"{'='*80}\n")

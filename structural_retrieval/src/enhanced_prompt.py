"""
Enhanced RAPM with Terminology-Focused Prompting
=================================================

This module implements improved prompt engineering for RAPM:
1. Same weighted similarity retrieval (no complex multi-hop)
2. Better prompt structure emphasizing biological terminology
3. Explicit instructions to use precise technical terms
4. Task-adaptive prompt templates

Key Improvements:
- Terminology emphasis: Instructs LLM to use precise biological terms
- Structured retrieval presentation: Groups by confidence levels
- Domain-specific instructions: Task-specific guidance
- Simpler than multi-hop: Just better prompting, no concept extraction

Usage:
    python con_retrieval.py <task_name> <top_k> [alpha] [model]
"""

import json
from tqdm import tqdm
import os
import sys
import numpy as np
import faiss
import time
from collections import defaultdict, Counter
import re

import google.generativeai as genai
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

os.environ['OPENBLAS_NUM_THREADS'] = '1'


# ===== Weighted Similarity Functions (from run_prostt5_rapm_sim.py) =====

def distance_to_similarity(distances, method='inverse', sigma=None):
    """Convert L2 distances to similarity scores."""
    if method == 'inverse':
        return 1.0 / (1.0 + distances)
    elif method == 'gaussian':
        if sigma is None:
            sigma = np.median(distances) if len(distances) > 0 else 1.0
        return np.exp(-distances**2 / (2 * sigma**2))
    elif method == 'exp':
        return np.exp(-distances)
    elif method == 'negative':
        return -distances
    else:
        raise ValueError(f"Unknown distance conversion method: {method}")


def normalize_scores(scores, method='minmax'):
    """Normalize similarity scores to [0,1] range."""
    if method == 'none':
        return scores

    if method == 'minmax':
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-10:
            return np.ones_like(scores) * 0.5
        return (scores - min_score) / (max_score - min_score)

    elif method == 'zscore':
        mean = np.mean(scores)
        std = np.std(scores)
        if std < 1e-10:
            return np.ones_like(scores) * 0.5
        z_scores = (scores - mean) / std
        return 1.0 / (1.0 + np.exp(-z_scores))

    elif method == 'softmax':
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def weighted_similarity_retrieval(test_prostt5, test_esm2,
                                  prostt5_index, esm2_index,
                                  top_k=100, alpha=0.7,
                                  distance_conversion='inverse',
                                  score_normalization='minmax'):
    """Retrieve top-k candidates using weighted similarity fusion."""
    D_prostt5, I_prostt5 = prostt5_index.search(test_prostt5.astype(np.float32), top_k)
    D_esm2, I_esm2 = esm2_index.search(test_esm2.astype(np.float32), top_k)

    results = []
    for i in range(len(test_prostt5)):
        sim_prostt5 = distance_to_similarity(D_prostt5[i], method=distance_conversion)
        sim_esm2 = distance_to_similarity(D_esm2[i], method=distance_conversion)

        sim_prostt5_norm = normalize_scores(sim_prostt5, method=score_normalization)
        sim_esm2_norm = normalize_scores(sim_esm2, method=score_normalization)

        combined_scores = {}

        for j, idx in enumerate(I_prostt5[i]):
            if idx not in combined_scores:
                combined_scores[idx] = 0.0
            combined_scores[idx] += alpha * sim_prostt5_norm[j]

        for j, idx in enumerate(I_esm2[i]):
            if idx not in combined_scores:
                combined_scores[idx] = 0.0
            combined_scores[idx] += (1 - alpha) * sim_esm2_norm[j]

        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        results.append(sorted_results)

    return results


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


genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


# ===== Evaluation Functions =====

def extract_words(text):
    words = re.findall(r'\w+', text)
    words = [w.lower() for w in words if w]
    return words


def evaluation(lines, labels, meta_labels, result_file):
    """Evaluate LLM predictions using BLEU, Meta-BLEU, METEOR, ROUGE, and Exact Match."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    labels = [l.strip() for l in labels]
    lines = [line.strip() for line in lines]

    assert len(labels) == len(lines) == len(meta_labels)

    total_exact_match = 0
    meteor_scores = []
    references = []
    hypotheses = []
    meta_references = []
    meta_hypotheses = []

    for pred, label, meta in tqdm(zip(lines, labels, meta_labels), desc="Evaluating"):

        if pred.strip() == label.strip():
            total_exact_match += 1

        gt_tokens = tokenizer.tokenize(label, truncation=True, max_length=1024, padding='longest')
        gt_tokens = list(filter(('<pad>').__ne__, gt_tokens))
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(pred, truncation=True, max_length=1024, padding='longest')
        out_tokens = list(filter(('<pad>').__ne__, out_tokens))
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        meta_words = extract_words(meta)
        pred_words = [word for word in extract_words(pred) if word in meta_words]
        meta_pred = " ".join(pred_words)

        label_words = [word for word in extract_words(label) if word in meta_words]
        meta_label = " ".join(label_words)

        gt_tokens = tokenizer.tokenize(meta_label, truncation=True, max_length=1024, padding='longest')
        gt_tokens = list(filter(('<pad>').__ne__, gt_tokens))
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(meta_pred, truncation=True, max_length=1024, padding='longest')
        out_tokens = list(filter(('<pad>').__ne__, out_tokens))
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        meta_references.append([gt_tokens])
        meta_hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5, .5)) * 100
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25, .25, .25, .25)) * 100

    print('BLEU-2 score:', bleu2, file=result_file)
    print('BLEU-4 score:', bleu4, file=result_file)

    meta_bleu2 = corpus_bleu(meta_references, meta_hypotheses, weights=(.5, .5)) * 100
    meta_bleu4 = corpus_bleu(meta_references, meta_hypotheses, weights=(.25, .25, .25, .25)) * 100
    print('Meta-BLEU-2 score:', meta_bleu2, file=result_file)
    print('Meta-BLEU-4 score:', meta_bleu4, file=result_file)
    print('Meta-BLEU-2 score:', meta_bleu2)
    print('Meta-BLEU-4 score:', meta_bleu4)

    _meteor_score = np.mean(meteor_scores) * 100
    print('Average Meteor score:', _meteor_score, file=result_file)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge_scores = []

    for gt, out in tqdm(zip(labels, lines), desc="Computing ROUGE"):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:', file=result_file)
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    print('rouge1:', rouge_1, file=result_file)
    print('rouge2:', rouge_2, file=result_file)
    print('rougeL:', rouge_l, file=result_file)
    print('rougeL:', rouge_l)
    print("Exact Match:", total_exact_match / len(lines), file=result_file)


# ===== Gemini API Inference =====

def api_inference(RAG_prompt, model):
    """Inference using Google Gemini API"""

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    gemini_model = genai.GenerativeModel(model, safety_settings=safety_settings)

    generation_config = genai.GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=8192,
    )

    try:
        response = gemini_model.generate_content(RAG_prompt, generation_config=generation_config)

        if not response.candidates:
            return ""

        candidate = response.candidates[0]

        if candidate.content and candidate.content.parts:
            output_results = "".join(part.text for part in candidate.content.parts)
        else:
            return ""

        output_results = output_results.replace("```json", "").replace("```", "")

        try:
            result_json = json.loads(output_results)
            if "description" in result_json:
                output_results = result_json["description"]
        except json.JSONDecodeError:
            pass

    except Exception as e:
        print(f"Gemini API Error: {e}")
        output_results = ""

    return output_results


# ===== Main Pipeline =====

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python con_retrieval.py <task_name> <top_k> [alpha] [model]")
        print("Example: python con_retrieval.py protein_function_OOD 10 0.7 gemini-2.5-flash")
        sys.exit(1)

    now_task = sys.argv[1]
    now_k = int(sys.argv[2])
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    model = sys.argv[4] if len(sys.argv) > 4 else "gemini-2.5-flash"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(os.path.dirname(script_dir), "dataset")

    # Configuration
    distance_conversion = 'inverse'
    score_normalization = 'minmax'

    result_file = open(os.path.join(script_dir, f"enhanced_prompt_alpha{alpha:.2f}_rapm_results.txt"), "a+")

    print("="*80)
    print(f"Enhanced RAPM with Terminology-Focused Prompting")
    print(f"Task: {now_task}, Top-K: {now_k}, Alpha: {alpha}, Model: {model}")
    print(f"Output directory: {script_dir}")
    print("="*80)
    print(f"Task: {now_task}, Top-K: {now_k}, Alpha: {alpha}, Model: {model}", file=result_file)

    # ===== Load All Training Data =====

    print("\n=== Loading all training data ===")
    all_train_seqs = []
    all_train_labels = []
    all_train_prostt5_features = []
    all_train_esm2_features = []

    for p in sorted(os.listdir(dataset_path)):
        if not p.endswith(".json"):
            continue

        task = p[:-5]
        JSON_PATH = os.path.join(dataset_path, p)
        dic = json.load(open(JSON_PATH, "r"))

        try:
            task_train_prostt5 = np.load(os.path.join(script_dir, f"hybrid_{task}_train_prostt5.npy"))
            task_train_esm2 = np.load(os.path.join(script_dir, f"hybrid_{task}_train_esm2.npy"))
        except FileNotFoundError:
            print(f"Warning: Features for {task} not found, skipping...")
            continue

        task_train_seqs = [d["sequence"] for d in dic if d['split'] == 'train']
        task_train_labels = [d["metadata"] for d in dic if d['split'] == 'train']

        all_train_seqs.extend(task_train_seqs)
        all_train_labels.extend(task_train_labels)
        all_train_prostt5_features.extend(task_train_prostt5)
        all_train_esm2_features.extend(task_train_esm2)

    print(f"Total training samples before aggregation: {len(all_train_labels)}")

    # Feature Aggregation
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

    # ===== Build FAISS Indices =====

    print("\n=== Building FAISS indices ===")

    db_prostt5_norm = all_train_prostt5_features / np.linalg.norm(all_train_prostt5_features, axis=1, keepdims=True)
    db_esm2_norm = all_train_esm2_features / np.linalg.norm(all_train_esm2_features, axis=1, keepdims=True)

    d = db_prostt5_norm.shape[1]
    prostt5_index = faiss.IndexHNSWFlat(d, 32)
    prostt5_index.add(db_prostt5_norm.astype(np.float32))
    prostt5_index.hnsw.efSearch = max(50, now_k * 2)

    d = db_esm2_norm.shape[1]
    esm2_index = faiss.IndexHNSWFlat(d, 32)
    esm2_index.add(db_esm2_norm.astype(np.float32))
    esm2_index.hnsw.efSearch = max(50, now_k * 2)

    print("FAISS indices built successfully")

    # ===== Load Test Data =====

    JSON_PATH = os.path.join(dataset_path, f"{now_task}.json")
    dic = json.load(open(JSON_PATH, "r"))

    test_instructions = [d["instruction"] for d in dic if d['split'] == 'test']
    test_seqs = [d["sequence"] for d in dic if d['split'] == 'test']
    test_labels = [d["description"] for d in dic if d['split'] == 'test']
    test_metas = [d["metadata"] for d in dic if d['split'] == 'test']

    train_labels = [d["description"] for d in dic if d['split'] == 'train']

    test_prostt5 = np.load(os.path.join(script_dir, f"hybrid_{now_task}_test_prostt5.npy"))
    test_esm2 = np.load(os.path.join(script_dir, f"hybrid_{now_task}_test_esm2.npy"))

    test_prostt5_norm = test_prostt5 / np.linalg.norm(test_prostt5, axis=1, keepdims=True)
    test_esm2_norm = test_esm2 / np.linalg.norm(test_esm2, axis=1, keepdims=True)

    print(f"\nLoaded {len(test_instructions)} test samples")

    # Set to evaluate all test samples
    infer_numbers = 256
    # ===== Build In-Task FAISS Index =====

    train_prostt5 = np.load(os.path.join(script_dir, f"hybrid_{now_task}_train_prostt5.npy"))
    train_prostt5_norm = train_prostt5 / np.linalg.norm(train_prostt5, axis=1, keepdims=True)
    train_faiss_index = faiss.IndexHNSWFlat(train_prostt5_norm.shape[1], 32)
    train_faiss_index.hnsw.efSearch = max(50, now_k * 2)
    train_faiss_index.add(train_prostt5_norm.astype(np.float32))
    train_D, train_I = train_faiss_index.search(train_prostt5_norm.astype(np.float32), now_k)

    # ===== Contrastive Multi-Hop RAPM =====

    all_answers = []
    all_labels = []
    all_meta_labels = []

    # Get task-specific instructions
    task_type = now_task.replace("_OOD", "")
    task_guidance = get_task_specific_instructions(task_type)

    print(f"\n=== Running Enhanced RAPM with Terminology-Focused Prompting on {infer_numbers} samples ===")

    for i in tqdm(range(infer_numbers)):

        # === Weighted Similarity Retrieval (Simple, No Multi-Hop) ===
        retrieval_results = weighted_similarity_retrieval(
            test_prostt5_norm[i:i+1], test_esm2_norm[i:i+1],
            prostt5_index, esm2_index,
            top_k=now_k, alpha=alpha,
            distance_conversion=distance_conversion,
            score_normalization=score_normalization
        )[0]

        # Group retrieved proteins by confidence levels
        high_conf = []
        medium_conf = []
        low_conf = []

        for idx, score in retrieval_results[:now_k]:
            annotation = all_train_labels[idx]
            if score >= 0.9:
                high_conf.append(annotation)
            elif score >= 0.7:
                medium_conf.append(annotation)
            else:
                low_conf.append(annotation)

        # === Get In-Task Examples ===
        train_examples = []
        for idx, score in zip(train_I[i][:5], train_D[i][:5]):
            train_examples.append(train_labels[idx])

        # === Construct Enhanced Prompt (Simple, Focused on Terminology) ===

        RAG_prompt = f"""You are a protein function prediction expert with deep knowledge of biological terminology.

**Task**: {test_instructions[i]}

**Query Protein Sequence**:
{test_seqs[i]}

**Retrieved Similar Proteins (Weighted Similarity: α={alpha})**:

🟢 **High Confidence Matches (score ≥ 0.9)**:
{chr(10).join([f"  • {ann}" for ann in high_conf]) if high_conf else "  None"}

🟡 **Medium Confidence Matches (0.7 ≤ score < 0.9)**:
{chr(10).join([f"  • {ann}" for ann in medium_conf]) if medium_conf else "  None"}

🔴 **Lower Confidence Matches (score < 0.7)**:
{chr(10).join([f"  • {ann}" for ann in low_conf]) if low_conf else "  None"}

**In-Task Training Examples** (for format reference):
{chr(10).join([f"  • {ex}" for ex in train_examples[:3]])}

{task_guidance}

**IMPORTANT INSTRUCTIONS**:
1. **Use PRECISE biological terminology** from the retrieved annotations
2. **Prioritize high-confidence matches** - they are most similar
3. **Extract domain-specific terms** (enzyme names, GO terms, motifs, etc.)
4. **Avoid generic descriptions** - be specific
5. **Match the terminology style** of the training examples

Output ONLY the functional description in JSON format:
{{"description": "..."}}

Do not include explanations, justifications, or any other text. Only the JSON answer.
"""

        # === API Inference ===
        LLM_answer = api_inference(RAG_prompt, model=model)

        all_answers.append(LLM_answer)
        all_labels.append(test_labels[i])
        all_meta_labels.append(test_metas[i])

    # ===== Evaluate Results =====

    print("\n=== Evaluating predictions ===")
    evaluation(all_answers, all_labels, all_meta_labels, result_file)

    # Save detailed results
    print(f"\n=== Saving detailed results ===")
    results_path = os.path.join(script_dir, f"ENHANCED_PROMPT_alpha{alpha:.2f}_{now_task}_{now_k}_results.json")

    with open(results_path, 'w') as f:
        for i in range(len(all_answers)):
            info_dict = {
                "instruction": test_instructions[i],
                "sequence": test_seqs[i],
                "answer": all_answers[i],
                "label": all_labels[i],
                "meta_label": all_meta_labels[i]
            }
            json.dump(info_dict, f)
            f.write('\n')

    print("="*80)
    print("Evaluation complete!")
    print(f"Results saved to: {results_path}")
    print(f"Metrics saved to: {os.path.join(script_dir, f'enhanced_prompt_alpha{alpha:.2f}_rapm_results.txt')}")
    print("="*80)

    result_file.close()

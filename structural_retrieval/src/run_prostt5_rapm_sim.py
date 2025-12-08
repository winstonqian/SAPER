"""
ProstT5-based RAPM inference using Weighted Similarity Fusion.
This script uses weighted similarity (α*sim_prostt5 + (1-α)*sim_esm2) instead of RRF
for retrieval-augmented protein modeling.

Usage:
    python run_prostt5_rapm_sim.py <task_name> <top_k> [alpha] [model]

Example:
    python run_prostt5_rapm_sim.py protein_function_OOD 10 0.7 gemini-2.5-flash
"""
import json
from tqdm import tqdm
import random
import os
import sys
import numpy as np
import faiss
import time
from collections import defaultdict

import google.generativeai as genai
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import AutoTokenizer
import re

os.environ['OPENBLAS_NUM_THREADS'] = '1'


# ===== Weighted Similarity Functions =====

def distance_to_similarity(distances, method='inverse', sigma=None):
    """
    Convert L2 distances to similarity scores.

    Args:
        distances: Array of L2 distances from FAISS
        method: Conversion method ('inverse', 'gaussian', 'exp', 'negative')
        sigma: Bandwidth for gaussian kernel (auto if None)

    Returns:
        similarity scores (higher = more similar)
    """
    if method == 'inverse':
        # 1/(1+d): Bounded [0,1], distance=0 -> sim=1
        return 1.0 / (1.0 + distances)

    elif method == 'gaussian':
        # Gaussian/RBF kernel: exp(-d²/(2σ²))
        if sigma is None:
            sigma = np.median(distances) if len(distances) > 0 else 1.0
        return np.exp(-distances**2 / (2 * sigma**2))

    elif method == 'exp':
        # Exponential: exp(-d)
        return np.exp(-distances)

    elif method == 'negative':
        # Simple negative: -d
        return -distances

    else:
        raise ValueError(f"Unknown distance conversion method: {method}")


def normalize_scores(scores, method='minmax'):
    """
    Normalize similarity scores to [0,1] range.

    Args:
        scores: Array of similarity scores
        method: Normalization method ('minmax', 'zscore', 'softmax', 'none')

    Returns:
        Normalized scores
    """
    if method == 'none':
        return scores

    if method == 'minmax':
        # Min-Max normalization to [0,1]
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-10:
            return np.ones_like(scores) * 0.5
        return (scores - min_score) / (max_score - min_score)

    elif method == 'zscore':
        # Z-score normalization + sigmoid to map to [0,1]
        mean = np.mean(scores)
        std = np.std(scores)
        if std < 1e-10:
            return np.ones_like(scores) * 0.5
        z_scores = (scores - mean) / std
        return 1.0 / (1.0 + np.exp(-z_scores))

    elif method == 'softmax':
        # Softmax normalization
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        return exp_scores / np.sum(exp_scores)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def weighted_similarity_retrieval(test_prostt5, test_esm2,
                                  prostt5_index, esm2_index,
                                  top_k=100, alpha=0.7,
                                  distance_conversion='inverse',
                                  score_normalization='minmax'):
    """
    Retrieve top-k candidates using weighted similarity fusion.

    Args:
        test_prostt5: Test ProstT5 features (normalized)
        test_esm2: Test ESM-2 features (normalized)
        prostt5_index: FAISS index for ProstT5
        esm2_index: FAISS index for ESM-2
        top_k: Number of candidates to retrieve
        alpha: Weight for ProstT5 (1-alpha for ESM-2)
        distance_conversion: Method to convert distances to similarities
        score_normalization: Method to normalize scores

    Returns:
        List of (indices, scores) for each test sample
    """
    # Search with both indices
    D_prostt5, I_prostt5 = prostt5_index.search(test_prostt5.astype(np.float32), top_k)
    D_esm2, I_esm2 = esm2_index.search(test_esm2.astype(np.float32), top_k)

    results = []
    for i in range(len(test_prostt5)):
        # Convert distances to similarities
        sim_prostt5 = distance_to_similarity(D_prostt5[i], method=distance_conversion)
        sim_esm2 = distance_to_similarity(D_esm2[i], method=distance_conversion)

        # Normalize similarities
        sim_prostt5_norm = normalize_scores(sim_prostt5, method=score_normalization)
        sim_esm2_norm = normalize_scores(sim_esm2, method=score_normalization)

        # Combine scores from both modalities
        combined_scores = {}

        # Add ProstT5 scores
        for j, idx in enumerate(I_prostt5[i]):
            if idx not in combined_scores:
                combined_scores[idx] = 0.0
            combined_scores[idx] += alpha * sim_prostt5_norm[j]

        # Add ESM-2 scores
        for j, idx in enumerate(I_esm2[i]):
            if idx not in combined_scores:
                combined_scores[idx] = 0.0
            combined_scores[idx] += (1 - alpha) * sim_esm2_norm[j]

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        results.append(sorted_results)

    return results


def score_to_confidence(score):
    """Convert score to confidence level."""
    if score >= 0.9:
        return "<High Confidence>"
    elif score >= 0.6:
        return "<Medium Confidence>"
    else:
        return "<Low Confidence>"


# ===== Evaluation Functions =====

def extract_words(text):
    words = re.findall(r'\w+', text)
    words = [w.lower() for w in words if w]
    return words


def evaluation(lines, labels, meta_labels, result_file):
    """
    Evaluate LLM predictions using BLEU, Meta-BLEU, METEOR, ROUGE, and Exact Match.
    """
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

        gt_tokens = tokenizer.tokenize(label, truncation=True, max_length=1024,
                                       padding='longest')

        gt_tokens = list(filter(('<pad>').__ne__, gt_tokens))
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(pred, truncation=True, max_length=1024,
                                        padding='longest')
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

        gt_tokens = tokenizer.tokenize(meta_label, truncation=True, max_length=1024,
                                       padding='longest')

        gt_tokens = list(filter(('<pad>').__ne__, gt_tokens))
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(meta_pred, truncation=True, max_length=1024,
                                        padding='longest')
        out_tokens = list(filter(('<pad>').__ne__, out_tokens))
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        meta_references.append([gt_tokens])
        meta_hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5, .5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25, .25, .25, .25))
    bleu2 *= 100
    bleu4 *= 100

    print('BLEU-2 score:', bleu2, file=result_file)
    print('BLEU-4 score:', bleu4, file=result_file)

    bleu2 = corpus_bleu(meta_references, meta_hypotheses, weights=(.5, .5))
    bleu4 = corpus_bleu(meta_references, meta_hypotheses, weights=(.25, .25, .25, .25))
    bleu2 *= 100
    bleu4 *= 100
    print('Meta-BLEU-2 score:', bleu2, file=result_file)
    print('Meta-BLEU-4 score:', bleu4, file=result_file)
    print('Meta-BLEU-2 score:', bleu2)
    print('Meta-BLEU-4 score:', bleu4)

    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
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


# ===== Gemini API Functions =====

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def api_inference(RAG_prompt, model):
    """Inference using Google Gemini API"""

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    gemini_model = genai.GenerativeModel(
        model,
        safety_settings=safety_settings
    )

    generation_config = genai.GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=8192,  # Increased from 8192 to reduce truncation
    )

    try:
        response = gemini_model.generate_content(
            RAG_prompt,
            generation_config=generation_config,
        )

        if not response.candidates:
            print("Warning: No candidates returned (likely blocked)")
            return ""

        candidate = response.candidates[0]

        if candidate.content and candidate.content.parts:
            output_results = "".join(part.text for part in candidate.content.parts)
        else:
            print(f"Warning: No content parts in response (finish_reason={candidate.finish_reason})")
            return ""

        if candidate.finish_reason == 3:
            print(f"Warning: Response blocked by safety filters")
        elif candidate.finish_reason == 2:
            print("Warning: Response truncated (max tokens) - output may be incomplete")

        output_results = output_results.replace("```json", "")
        output_results = output_results.replace("```", "")

        try:
            result_json = json.loads(output_results)
            if "description" in result_json:
                output_results = result_json["description"]
            else:
                print("Error: 'description' key not found in JSON. Returning original output.")
        except json.JSONDecodeError:
            pass

    except Exception as e:
        print(f"Gemini API Error: {e}")
        output_results = ""

    return output_results


# ===== Main RAPM Pipeline =====

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python run_prostt5_rapm_sim.py <task_name> <top_k> [alpha] [model]")
        print("Example: python run_prostt5_rapm_sim.py protein_function_OOD 10 0.7 gemini-2.5-flash")
        sys.exit(1)

    now_task = sys.argv[1]
    now_k = int(sys.argv[2])
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    model = sys.argv[4] if len(sys.argv) > 4 else "gemini-2.5-flash"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    # Dataset path relative to script directory
    dataset_path = os.path.join(os.path.dirname(script_dir), "dataset")

    # Configuration for weighted similarity
    distance_conversion = 'inverse'
    score_normalization = 'minmax'

    result_file = open(os.path.join(results_dir, f"hybrid_weighted_sim_alpha{alpha:.2f}_256_rapm_results.txt"), "a+")
    all_input_prompt_len = 0

    print("="*80)
    print(f"Weighted Similarity RAPM Evaluation")
    print(f"Task: {now_task}, Top-K: {now_k}, Alpha: {alpha}, Model: {model}")
    print(f"Distance conversion: {distance_conversion}")
    print(f"Score normalization: {score_normalization}")
    print(f"Output directory: {parent_dir}")
    print("="*80)
    print(f"Task: {now_task}, Top-K: {now_k}, Alpha: {alpha}, Model: {model}", file=result_file)

    # ===== Load All Training Data =====

    all_train_seqs = []
    all_train_labels = []
    all_train_prostt5_features = []
    all_train_esm2_features = []

    print("\n=== Loading all training data for cross-task retrieval ===")
    for p in sorted(os.listdir(dataset_path)):
        if not p.endswith(".json"):
            continue

        task = p[:-5]
        JSON_PATH = os.path.join(dataset_path, p)
        dic = json.load(open(JSON_PATH, "r"))

        try:
            task_train_prostt5 = np.load(os.path.join(parent_dir, f"hybrid_{task}_train_prostt5.npy"))
            task_train_esm2 = np.load(os.path.join(parent_dir, f"hybrid_{task}_train_esm2.npy"))
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

    # ===== Build FAISS Indices =====

    print("\n=== Building FAISS indices ===")

    # Normalize features
    db_prostt5_norm = all_train_prostt5_features / np.linalg.norm(all_train_prostt5_features, axis=1, keepdims=True)
    db_esm2_norm = all_train_esm2_features / np.linalg.norm(all_train_esm2_features, axis=1, keepdims=True)

    # Build ProstT5 index
    d = db_prostt5_norm.shape[1]
    prostt5_index = faiss.IndexHNSWFlat(d, 32)
    prostt5_index.add(db_prostt5_norm.astype(np.float32))
    prostt5_index.hnsw.efSearch = max(50, now_k * 2)

    # Build ESM-2 index
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

    # Load test features
    test_prostt5 = np.load(os.path.join(parent_dir, f"hybrid_{now_task}_test_prostt5.npy"))
    test_esm2 = np.load(os.path.join(parent_dir, f"hybrid_{now_task}_test_esm2.npy"))

    # Normalize test features
    test_prostt5_norm = test_prostt5 / np.linalg.norm(test_prostt5, axis=1, keepdims=True)
    test_esm2_norm = test_esm2 / np.linalg.norm(test_esm2, axis=1, keepdims=True)

    print(f"\nLoaded {len(test_instructions)} test samples")

    # Set to evaluate all test samples
    infer_numbers = 256

    # ===== Perform Weighted Similarity Retrieval =====

    print(f"\n=== Performing weighted similarity retrieval (α={alpha}) ===")
    st_time = time.time()
    weighted_results = weighted_similarity_retrieval(
        test_prostt5_norm, test_esm2_norm,
        prostt5_index, esm2_index,
        top_k=now_k,
        alpha=alpha,
        distance_conversion=distance_conversion,
        score_normalization=score_normalization
    )
    print(f"Weighted similarity retrieval time: {time.time() - st_time:.4f} seconds")

    # ===== Build FAISS Index for In-Task Examples =====

    train_prostt5 = np.load(os.path.join(parent_dir, f"hybrid_{now_task}_train_prostt5.npy"))
    train_prostt5_norm = train_prostt5 / np.linalg.norm(train_prostt5, axis=1, keepdims=True)
    train_faiss_index = faiss.IndexHNSWFlat(train_prostt5_norm.shape[1], 32)
    train_faiss_index.hnsw.efSearch = max(50, now_k * 2)
    train_faiss_index.add(train_prostt5_norm.astype(np.float32))
    train_D, train_I = train_faiss_index.search(train_prostt5_norm.astype(np.float32), now_k)

    # ===== Generate Prompts and Run Inference =====

    all_answers = []
    all_labels = []
    all_meta_labels = []

    print(f"\n=== Running RAPM inference on {infer_numbers} samples ===")
    for i in tqdm(range(infer_numbers)):
        # Get retrieved information from weighted similarity
        retrieved_info = []
        for idx, score in weighted_results[i][:now_k]:
            retrieved_info.append({
                "db_label": all_train_labels[idx],
                "confidence level": score_to_confidence(score),
                "weighted_score": float(score)
            })

        # Get in-task examples
        train_examples = []
        for idx, score in zip(train_I[i], train_D[i]):
            train_examples.append({
                "example answer": train_labels[idx],
                "confidence level": score_to_confidence(1.0 / (1.0 + score))
            })

        # Construct RAG prompt
        RAG_prompt = (
            f"You are given a protein sequence and a list of related proteins retrieved from a database.\n"
            f"Instruction: {test_instructions[i]}\n"
            f"Protein sequence: {test_seqs[i]}\n"
            f"Retrieved proteins annotations by Weighted Similarity (α={alpha}, ProstT5 + ESM-2): {retrieved_info}\n"
            f"Here are some example input-output pairs for this task:\n"
            f"{train_examples}\n"
            "Based on the instruction, the protein sequence, the retrieved information, and the examples, "
            "output ONLY the functional description of this protein in the following JSON format:\n"
            '{"description": "..."}'
            "\nDo not output any other text or explanation. Only output the JSON answer."
        )

        # API inference
        LLM_answer = api_inference(RAG_prompt, model=model)

        all_input_prompt_len += len(RAG_prompt.split())
        all_answers.append(LLM_answer)
        all_labels.append(test_labels[i])
        all_meta_labels.append(test_metas[i])

    print(f"\nAverage input prompt length: {all_input_prompt_len / infer_numbers:.2f}")

    # ===== Evaluate Results =====

    print("\n=== Evaluating predictions ===")
    evaluation(all_answers, all_labels, all_meta_labels, result_file)

    # Save detailed results
    print(f"\n=== Saving detailed results ===")
    results_path = os.path.join(results_dir, f"WEIGHTED_SIM_alpha{alpha:.2f}_{now_task}_{now_k}_results.json")

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
    print(f"Metrics saved to: {os.path.join(results_dir, f'hybrid_weighted_sim_alpha{alpha:.2f}_256_rapm_results.txt')}")
    print("="*80)

    result_file.close()

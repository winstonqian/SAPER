"""
Weighted similarity score fusion for hybrid protein retrieval.
This script implements score-based fusion (instead of rank-based RRF) for combining
ProstT5 (structure-aware) and ESM-2 (sequence-based) embeddings.

Reference: RAPM paper uses: score = α * sim_1 + (1-α) * sim_2 with α=0.5
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'retrival_methods'))

from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt', quiet=True)
import json
import re
import faiss

from mmseq_utils import align_and_analyze

# Store results in structural_retrieval/results/
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
results_dir = os.path.join(parent_dir, "results")
os.makedirs(results_dir, exist_ok=True)
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def extract_words(text):
    words = re.findall(r'\w+', text)
    words = [w.lower() for w in words if w]
    return words


def evaluation(lines, labels, meta_labels, task_name, result_file):
    """
    Evaluate predictions using BLEU, Meta-BLEU, METEOR, ROUGE, and Exact Match.
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
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

        # Meta-BLEU: filter to only words present in metadata
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

    # BLEU scores
    bleu2 = corpus_bleu(references, hypotheses, weights=(.5, .5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25, .25, .25, .25))
    bleu2 *= 100
    bleu4 *= 100

    print('BLEU-2 score:', bleu2, file=result_file)
    print('BLEU-4 score:', bleu4, file=result_file)

    # Meta-BLEU scores
    bleu2 = corpus_bleu(meta_references, meta_hypotheses, weights=(.5, .5))
    bleu4 = corpus_bleu(meta_references, meta_hypotheses, weights=(.25, .25, .25, .25))
    bleu2 *= 100
    bleu4 *= 100
    print('Meta-BLEU-2 score:', bleu2, file=result_file)
    print('Meta-BLEU-4 score:', bleu4, file=result_file)

    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print('Average Meteor score:', _meteor_score, file=result_file)

    # ROUGE scores
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
    print("Exact Match:", total_exact_match / len(lines), file=result_file)

    # Also print to console for monitoring
    print(f"Task: {task_name}")
    print(f"  Meta-BLEU-2: {bleu2:.2f}, Meta-BLEU-4: {bleu4:.2f}")
    print(f"  ROUGE-L: {rouge_l:.2f}, Exact Match: {total_exact_match / len(lines):.4f}")


def distance_to_similarity(distances, method='negative', sigma=None):
    """
    Convert L2 distances from FAISS to similarity scores.

    Args:
        distances: Array of L2 distances (lower is better)
        method: Conversion method
            - 'negative': -distance (simple, preserves ordering)
            - 'inverse': 1 / (1 + distance) (bounded [0,1])
            - 'gaussian': exp(-distance^2 / (2*sigma^2)) (Gaussian kernel)
            - 'exp': exp(-distance) (exponential decay)
        sigma: Bandwidth for Gaussian kernel (auto-estimated if None)

    Returns:
        similarities: Array of similarity scores (higher is better)
    """
    if method == 'negative':
        # Simple negation: higher similarity for lower distance
        return -distances

    elif method == 'inverse':
        # Inverse transformation: maps [0, inf] -> [0, 1]
        return 1.0 / (1.0 + distances)

    elif method == 'gaussian':
        # Gaussian/RBF kernel
        if sigma is None:
            # Auto-estimate sigma as median distance
            sigma = np.median(distances[distances > 0])
        return np.exp(-distances**2 / (2 * sigma**2))

    elif method == 'exp':
        # Exponential decay
        return np.exp(-distances)

    else:
        raise ValueError(f"Unknown method: {method}")


def normalize_scores(scores, method='minmax'):
    """
    Normalize similarity scores to [0, 1] range.

    Args:
        scores: Array of similarity scores
        method: Normalization method
            - 'minmax': (x - min) / (max - min)
            - 'zscore': (x - mean) / std, then sigmoid
            - 'softmax': exponential normalization
            - 'none': no normalization

    Returns:
        normalized_scores: Normalized scores
    """
    if method == 'none':
        return scores

    elif method == 'minmax':
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-10:
            return np.ones_like(scores) * 0.5
        return (scores - min_score) / (max_score - min_score)

    elif method == 'zscore':
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score < 1e-10:
            return np.ones_like(scores) * 0.5
        z_scores = (scores - mean_score) / std_score
        # Apply sigmoid to map to [0, 1]
        return 1.0 / (1.0 + np.exp(-z_scores))

    elif method == 'softmax':
        # Exponential normalization (preserves relative differences)
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        return exp_scores / np.sum(exp_scores)

    else:
        raise ValueError(f"Unknown method: {method}")


def weighted_similarity_predict(train_prostt5_features, train_esm2_features, train_labels,
                                 test_prostt5_features, test_esm2_features,
                                 top_k=100, alpha=0.5,
                                 distance_conversion='inverse',
                                 score_normalization='minmax',
                                 sigma_prostt5=None, sigma_esm2=None):
    """
    Predict labels using weighted similarity score fusion of ProstT5 and ESM-2.

    Formula: final_score = α * sim_prostt5 + (1-α) * sim_esm2

    Args:
        train_prostt5_features: Training ProstT5 features [N_train, 1024]
        train_esm2_features: Training ESM-2 features [N_train, 1280]
        train_labels: Training labels
        test_prostt5_features: Test ProstT5 features [N_test, 1024]
        test_esm2_features: Test ESM-2 features [N_test, 1280]
        top_k: Number of candidates to retrieve
        alpha: Weight for ProstT5 (1-alpha for ESM-2). Range [0, 1]
               α=0.5: Equal weight (RAPM paper default)
               α>0.5: Emphasize structure (ProstT5)
               α<0.5: Emphasize sequence (ESM-2)
        distance_conversion: Method to convert L2 distance to similarity
        score_normalization: Method to normalize scores before fusion
        sigma_prostt5: Bandwidth for Gaussian kernel (ProstT5)
        sigma_esm2: Bandwidth for Gaussian kernel (ESM-2)

    Returns:
        pred_labels: Predicted labels for test set
        pred_scores: Combined similarity scores for analysis
    """
    # Build Faiss indices for both feature types
    d_prostt5 = train_prostt5_features.shape[1]
    d_esm2 = train_esm2_features.shape[1]

    prostt5_index = faiss.IndexFlatL2(d_prostt5)
    prostt5_index.add(train_prostt5_features.astype(np.float32))

    esm2_index = faiss.IndexFlatL2(d_esm2)
    esm2_index.add(train_esm2_features.astype(np.float32))

    print(f"Performing weighted similarity prediction on {len(test_prostt5_features)} test samples...")
    print(f"  α={alpha:.2f} (ProstT5 weight), 1-α={1-alpha:.2f} (ESM-2 weight)")
    print(f"  Distance conversion: {distance_conversion}")
    print(f"  Score normalization: {score_normalization}")

    # Retrieve top_k candidates from each modality
    D_prostt5, I_prostt5 = prostt5_index.search(
        test_prostt5_features.astype(np.float32), top_k
    )
    D_esm2, I_esm2 = esm2_index.search(
        test_esm2_features.astype(np.float32), top_k
    )

    pred_labels = []
    pred_scores = []

    # For each test sample, fuse similarity scores
    for i in tqdm(range(len(test_prostt5_features)), desc="Weighted similarity fusion"):
        # Get distances for this test sample
        prostt5_distances = D_prostt5[i]
        prostt5_indices = I_prostt5[i]
        esm2_distances = D_esm2[i]
        esm2_indices = I_esm2[i]

        # Convert distances to similarities
        prostt5_similarities = distance_to_similarity(
            prostt5_distances, method=distance_conversion, sigma=sigma_prostt5
        )
        esm2_similarities = distance_to_similarity(
            esm2_distances, method=distance_conversion, sigma=sigma_esm2
        )

        # Normalize scores to [0, 1] for fair combination
        prostt5_similarities_norm = normalize_scores(prostt5_similarities, method=score_normalization)
        esm2_similarities_norm = normalize_scores(esm2_similarities, method=score_normalization)

        # Combine scores from both modalities
        # We need to merge the two candidate sets
        combined_scores = {}

        # Add ProstT5 scores
        for idx, score in zip(prostt5_indices, prostt5_similarities_norm):
            combined_scores[idx] = alpha * score

        # Add ESM-2 scores (merge with existing)
        for idx, score in zip(esm2_indices, esm2_similarities_norm):
            if idx in combined_scores:
                combined_scores[idx] += (1 - alpha) * score
            else:
                combined_scores[idx] = (1 - alpha) * score

        # Sort by combined score (higher is better)
        sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Take the label of the top-ranked candidate
        best_idx = sorted_candidates[0][0]
        best_score = sorted_candidates[0][1]

        pred_labels.append(train_labels[best_idx])
        pred_scores.append(best_score)

    return pred_labels, pred_scores


def hybrid_weighted_sim_predict(train_seqs, test_seqs, train_labels, test_labels, task_name,
                                 top_k=100, alpha=0.5,
                                 distance_conversion='inverse',
                                 score_normalization='minmax'):
    """
    Full pipeline: load/extract features and perform weighted similarity prediction.

    Args:
        train_seqs: List of training sequences
        test_seqs: List of test sequences
        train_labels: List of training labels
        test_labels: List of test labels
        task_name: Name of the task
        top_k: Number of candidates for fusion
        alpha: Weight for ProstT5 (0.5 = equal weight)
        distance_conversion: Method to convert distance to similarity
        score_normalization: Method to normalize scores

    Returns:
        pred_labels: Predictions
        test_labels: True labels
        acc: Accuracy
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Load pre-computed hybrid features
    print(f"Loading hybrid features for {task_name}...")
    train_prostt5 = np.load(os.path.join(parent_dir, f"hybrid_{task_name}_train_prostt5.npy"))
    train_esm2 = np.load(os.path.join(parent_dir, f"hybrid_{task_name}_train_esm2.npy"))
    test_prostt5 = np.load(os.path.join(parent_dir, f"hybrid_{task_name}_test_prostt5.npy"))
    test_esm2 = np.load(os.path.join(parent_dir, f"hybrid_{task_name}_test_esm2.npy"))

    print(f"  Train: ProstT5 {train_prostt5.shape}, ESM-2 {train_esm2.shape}")
    print(f"  Test: ProstT5 {test_prostt5.shape}, ESM-2 {test_esm2.shape}")

    # Predict using weighted similarity fusion
    print(f"Performing weighted similarity prediction (α={alpha}, top_k={top_k})...")
    pred_labels, pred_scores = weighted_similarity_predict(
        train_prostt5, train_esm2, train_labels,
        test_prostt5, test_esm2,
        top_k=top_k, alpha=alpha,
        distance_conversion=distance_conversion,
        score_normalization=score_normalization
    )

    acc = sum(p == t for p, t in zip(pred_labels, test_labels)) / len(test_labels)

    print(f"Weighted similarity prediction accuracy: {acc:.4f}")
    print(f"  Average combined score: {np.mean(pred_scores):.4f}")
    print(f"  Score std: {np.std(pred_scores):.4f}")

    return pred_labels, test_labels, acc


if __name__ == "__main__":
    """
    Test different α values to find optimal weight between ProstT5 and ESM-2.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Hybrid retrieval with weighted similarity fusion')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight for ProstT5 (0.5=equal, >0.5=favor structure, <0.5=favor sequence)')
    parser.add_argument('--top_k', type=int, default=100,
                       help='Number of candidates to retrieve from each modality')
    parser.add_argument('--distance_conversion', type=str, default='inverse',
                       choices=['negative', 'inverse', 'gaussian', 'exp'],
                       help='Method to convert L2 distance to similarity')
    parser.add_argument('--score_normalization', type=str, default='minmax',
                       choices=['none', 'minmax', 'zscore', 'softmax'],
                       help='Method to normalize scores before fusion')

    args = parser.parse_args()

    JSON_FOLDER = "../../dataset"

    result_file = open(os.path.join(results_dir,
                                     f"hybrid_weighted_sim_alpha{args.alpha:.2f}_results.txt"), "w")

    print("="*80)
    print(f"Hybrid Weighted Similarity (α={args.alpha}) Protein Retrieval Evaluation")
    print(f"Distance conversion: {args.distance_conversion}")
    print(f"Score normalization: {args.score_normalization}")
    print("="*80)
    print(file=result_file)
    print("="*80, file=result_file)
    print(f"Hybrid Weighted Similarity (α={args.alpha}) Protein Retrieval Evaluation", file=result_file)
    print(f"Distance conversion: {args.distance_conversion}, Score norm: {args.score_normalization}", file=result_file)
    print("="*80, file=result_file)

    # Process each task
    for p in sorted(os.listdir(JSON_FOLDER)):
        if not p.endswith('.json'):
            continue

        print(f"\n{'='*60}")
        print(f"Processing task: {p[:-5]}")
        print(f"{'='*60}")
        print(f"\n{'='*60}", file=result_file)
        print(f"Task: {p[:-5]}", file=result_file)
        print(f"{'='*60}", file=result_file)

        JSON_PATH = os.path.join(JSON_FOLDER, p)
        dic = json.load(open(JSON_PATH, "r"))

        train_dic = [d for d in dic if d["split"] == "train"]
        test_dic = [d for d in dic if d["split"] == "test"]

        train_seqs = [d["sequence"] for d in train_dic]
        train_labels = [d["description"] for d in train_dic]
        test_seqs = [d["sequence"] for d in test_dic]
        test_labels = [d["description"] for d in test_dic]
        meta_list = [d["metadata"] for d in test_dic]

        print(f"Train size: {len(train_seqs)}, Test size: {len(test_seqs)}")

        # Weighted similarity fusion
        print(f"\nWeighted similarity (α={args.alpha}) retrieval for {p[:-5]}...")
        print(f"\nWeighted similarity (α={args.alpha}) retrieval for {p[:-5]}...", file=result_file)
        pred_labels, test_labels, acc = hybrid_weighted_sim_predict(
            train_seqs, test_seqs, train_labels, test_labels, p[:-5],
            top_k=args.top_k, alpha=args.alpha,
            distance_conversion=args.distance_conversion,
            score_normalization=args.score_normalization
        )
        evaluation(pred_labels, test_labels, meta_list, p[:-5], result_file)

    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"Results saved to: {os.path.join(results_dir, f'hybrid_weighted_sim_alpha{args.alpha:.2f}_results.txt')}")
    print(f"{'='*60}")

    result_file.close()

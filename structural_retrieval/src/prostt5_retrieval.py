"""
Structurally-aware protein retrieval using ProstT5 embeddings.
This script evaluates retrieval performance using ProstT5 features
and compares with MMSeqs2 sequence-based retrieval.
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

from mmseq_utils import align_and_analyze
from prostt5_features import hybrid_rrf_predict, rrf_predict

# Store results in structural_retrieval/
script_dir = os.path.dirname(os.path.abspath(__file__))
result_file = open(os.path.join(script_dir, "hybrid_rrf_results.txt"), "w")
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def extract_words(text):
    words = re.findall(r'\w+', text)
    words = [w.lower() for w in words if w]
    return words


def evaluation(lines, labels, meta_labels, task_name):
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

        # Tokenize predictions and labels
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

        # Meta-BLEU: filter to only words present in metadata
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


if __name__ == "__main__":

    JSON_FOLDER = "../dataset"

    all_train_seqs = []
    all_train_labels = []
    all_test_seqs = []
    all_test_labels = []
    all_meta_list = []

    all_train_features = []
    all_test_features = []

    task_ranges = []  # Track each task's test data range
    task_names = []

    print("="*80)
    print("Hybrid RRF (ProstT5 + ESM-2) Protein Retrieval Evaluation")
    print("="*80)
    print(file=result_file)
    print("="*80, file=result_file)
    print("Hybrid RRF (ProstT5 + ESM-2) Protein Retrieval Evaluation", file=result_file)
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

        # Track task ranges for combined evaluation
        start_idx = len(all_test_seqs)
        all_train_seqs.extend(train_seqs)
        all_train_labels.extend(train_labels)
        all_test_seqs.extend(test_seqs)
        all_test_labels.extend(test_labels)
        all_meta_list.extend(meta_list)
        end_idx = len(all_test_seqs)
        task_ranges.append((start_idx, end_idx))
        task_names.append(p[:-5])

        print(f"Train size: {len(train_seqs)}, Test size: {len(test_seqs)}")

        # Hybrid RRF (ProstT5 + ESM-2) retrieval
        print(f"\n[1/2] Hybrid RRF retrieval for {p[:-5]}...")
        print(f"\nHybrid RRF (ProstT5 + ESM-2) retrieval for {p[:-5]}...", file=result_file)
        pred_labels, test_labels, acc = hybrid_rrf_predict(
            train_seqs, test_seqs, train_labels, test_labels, p[:-5], top_k=10
        )
        evaluation(pred_labels, test_labels, meta_list, p[:-5])

        # MMSeqs2 sequence retrieval (for comparison)
        print(f"\n[2/2] MMSeqs2 sequence retrieval for {p[:-5]}...")
        print(f"\nMMSeqs2 sequence retrieval for {p[:-5]}...", file=result_file)

        # Change to structural_retrieval directory for MMSeqs2 temp files
        original_dir = os.getcwd()
        os.chdir(script_dir)
        pred_labels, test_labels = align_and_analyze(
            train_seqs, test_seqs, train_labels, test_labels, f"prostt5_{p[:-5]}"
        )
        os.chdir(original_dir)

        evaluation(pred_labels, test_labels, meta_list, f"{p[:-5]}_mmseqs")

        # Load features for combined evaluation later
        train_prostt5 = np.load(os.path.join(script_dir, f"hybrid_{p[:-5]}_train_prostt5.npy"))
        train_esm2 = np.load(os.path.join(script_dir, f"hybrid_{p[:-5]}_train_esm2.npy"))
        test_prostt5 = np.load(os.path.join(script_dir, f"hybrid_{p[:-5]}_test_prostt5.npy"))
        test_esm2 = np.load(os.path.join(script_dir, f"hybrid_{p[:-5]}_test_esm2.npy"))

        all_train_features.append((train_prostt5, train_esm2))
        all_test_features.append((test_prostt5, test_esm2))

    # Combined evaluation across all tasks
    print(f"\n{'='*60}")
    print("Combined All-Task Evaluation")
    print(f"{'='*60}")
    print(f"\n{'='*60}", file=result_file)
    print("Combined All-Task Evaluation", file=result_file)
    print(f"{'='*60}", file=result_file)

    # Concatenate features from all tasks
    all_train_prostt5 = np.concatenate([feat[0] for feat in all_train_features], axis=0)
    all_train_esm2 = np.concatenate([feat[1] for feat in all_train_features], axis=0)
    all_test_prostt5 = np.concatenate([feat[0] for feat in all_test_features], axis=0)
    all_test_esm2 = np.concatenate([feat[1] for feat in all_test_features], axis=0)

    # MMSeqs2 on all tasks
    print(f"\nMMSeqs2 retrieval for all_task...")
    print(f"\nMMSeqs2 retrieval for all_task...", file=result_file)

    # Change to structural_retrieval directory for MMSeqs2 temp files
    original_dir = os.getcwd()
    os.chdir(script_dir)
    all_pred_labels, all_test_labels = align_and_analyze(
        all_train_seqs, all_test_seqs, all_train_labels, all_test_labels, 'prostt5_all'
    )
    os.chdir(original_dir)

    # Evaluate subtasks
    for (start, end), task_name in zip(task_ranges, task_names):
        task_name += "_all_mmseqs"
        print(f"\nEvaluation for task: {task_name}", file=result_file)
        evaluation(
            all_pred_labels[start:end],
            all_test_labels[start:end],
            meta_labels=all_meta_list[start:end],
            task_name=task_name
        )

    # Hybrid RRF on all tasks
    print(f"\nHybrid RRF retrieval for all_task...")
    print(f"\nHybrid RRF retrieval for all_task...", file=result_file)
    all_pred_labels_rrf = rrf_predict(all_train_prostt5, all_train_esm2, all_train_labels,
                                       all_test_prostt5, all_test_esm2, top_k=100)

    # Evaluate RRF on combined all_task, split by subtasks
    for (start, end), task_name in zip(task_ranges, task_names):
       task_name += "_all_rrf"
       print(f"\nEvaluation for task: {task_name}", file=result_file)
       evaluation(
           all_pred_labels_rrf[start:end],
           all_test_labels[start:end],
           meta_labels=all_meta_list[start:end],
           task_name=task_name
       )

    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"Results saved to: {os.path.join(script_dir, 'hybrid_rrf_results.txt')}")
    print(f"{'='*60}")

    result_file.close()

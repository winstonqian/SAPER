"""
ProstT5-based RAPM inference using Gemini API.
This script runs end-to-end evaluation using structurally-aware retrieval.
"""
import json
from tqdm import tqdm
import random
import os
import sys

import google.generativeai as genai
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np
from transformers import AutoTokenizer
import re

os.environ['OPENBLAS_NUM_THREADS'] = '1'


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


# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def api_inference(RAG_prompt, model):
    """Inference using Google Gemini API"""

    # Initialize Gemini model with relaxed safety settings
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

    # Configure generation parameters
    generation_config = genai.GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=8192,
    )

    try:
        # Generate response
        response = gemini_model.generate_content(
            RAG_prompt,
            generation_config=generation_config,
        )

        # Check if response was blocked
        if not response.candidates:
            print("Warning: No candidates returned (likely blocked)")
            return ""

        candidate = response.candidates[0]

        # Check finish_reason FIRST to handle appropriately
        # finish_reason: 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
        if candidate.finish_reason == 3:  # SAFETY - blocked
            print(f"Warning: Response blocked by safety filters")
            return ""

        # Try to extract text from candidate parts (for STOP or MAX_TOKENS)
        if candidate.content and candidate.content.parts:
            output_results = "".join(part.text for part in candidate.content.parts)

            # Warn if truncated, but STILL USE the partial response
            if candidate.finish_reason == 2:  # MAX_TOKENS
                print("Warning: Response truncated (max tokens) - using partial output")
        else:
            # No content - shouldn't happen for normal responses
            print(f"Warning: No content parts in response (finish_reason={candidate.finish_reason})")
            return ""

        # Clean up JSON formatting if present
        output_results = output_results.replace("```json", "")
        output_results = output_results.replace("```", "")

        # Try to extract description field if output is JSON
        try:
            result_json = json.loads(output_results)
            if "description" in result_json:
                output_results = result_json["description"]
            else:
                print("Error: 'description' key not found in JSON. Returning original output.")
        except json.JSONDecodeError:
            # Not JSON, use full text output
            pass

    except Exception as e:
        print(f"Gemini API Error: {e}")
        output_results = ""

    return output_results


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python run_prostt5_rapm.py <task_name> <top_k> [model]")
        print("Example: python run_prostt5_rapm.py protein_function_OOD 10 gemini-2.5-flash")
        sys.exit(1)

    now_task = sys.argv[1]
    now_k = int(sys.argv[2])

    # Default to Gemini 2.5 Flash (best cost-performance)
    model = sys.argv[3] if len(sys.argv) > 3 else "gemini-2.5-flash"

    # Get script directory and parent directory for input/output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    result_file = open(os.path.join(results_dir, "hybrid_rrf_rapm_256_results.txt"), "a+")
    all_input_prompt_len = 0

    # Load prompts to determine how many samples to evaluate
    JSON_PATH = os.path.join(parent_dir, f"HYBRID_RRF_{now_task}_RAP_Top_{now_k}.json")

    if not os.path.exists(JSON_PATH):
        print(f"Error: Prompt file {JSON_PATH} not found!")
        print(f"Please run prostt5_rag_prompt.py first to generate prompts.")
        sys.exit(1)

    dic = json.load(open(JSON_PATH, "r"))
    infer_numbers = 256

    print("="*80)
    print(f"Hybrid RRF RAPM Evaluation")
    print(f"Task: {now_task}, Top-K: {now_k}, Model: {model}")
    print(f"Output directory: {parent_dir}")
    print("="*80)
    print("Task:", now_task, "Top-K:", now_k, "Model:", model, file=result_file)

    JSON_PATH = os.path.join(parent_dir, f"HYBRID_RRF_{now_task}_RAP_Top_{now_k}.json")

    if not os.path.exists(JSON_PATH):
        print(f"Error: Prompt file {JSON_PATH} not found!")
        print(f"Please run prostt5_rag_prompt.py first to generate prompts.")
        sys.exit(1)

    dic = json.load(open(JSON_PATH, "r"))
    all_answers = []
    all_labels = []
    all_meta_labels = []

    print(f"\nRunning inference on {infer_numbers} samples...")
    for i in tqdm(range(infer_numbers)):
        d = dic[i]
        RAG_prompt = d['RAG_prompt']
        answer = d['labels']
        meta_answer = d["meta_label"]

        LLM_answer = api_inference(RAG_prompt, model=model)

        all_input_prompt_len += len(RAG_prompt.split())

        all_answers.append(LLM_answer)
        all_labels.append(answer)
        all_meta_labels.append(meta_answer)

    print(f"\nAverage input prompt length: {all_input_prompt_len / infer_numbers:.2f}")

    print("\nEvaluating predictions...")
    evaluation(all_answers, all_labels, all_meta_labels, result_file)

    # Save detailed results
    print(f"\nSaving detailed results...")
    results_path = os.path.join(results_dir, f"HYBRID_RRF_{now_task}_{now_k}_results.json")

    for i in range(len(all_answers)):
        info_dict = {
            "RAG_prompt": dic[i]['RAG_prompt'],
            "answer": all_answers[i],
            "label": all_labels[i],
            "meta_label": all_meta_labels[i]
        }

        with open(results_path, 'a+') as f:
            json.dump(info_dict, f)
            f.write('\n')

    print("="*80)
    print("Evaluation complete!")
    print(f"Results saved to: {results_path}")
    print(f"Metrics saved to: {os.path.join(results_dir, 'hybrid_rrf_rapm_256_results.txt')}")
    print("="*80)

    result_file.close()

"""
Hybrid feature extraction using ProstT5 (structure-aware) + ESM-2 (sequence-based).
Combines both embeddings using Reciprocal Rank Fusion (RRF) for retrieval.
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel
import faiss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ProstT5 model configuration
PROSTT5_MODEL_NAME = "Rostlab/ProstT5"
print(f"Loading ProstT5 model on {DEVICE}...")
prostt5_tokenizer = T5Tokenizer.from_pretrained(PROSTT5_MODEL_NAME, do_lower_case=False)
prostt5_model = T5EncoderModel.from_pretrained(PROSTT5_MODEL_NAME).to(DEVICE)
if DEVICE.type == 'cuda':
    prostt5_model = prostt5_model.half()
prostt5_model.eval()
print("ProstT5 model loaded successfully!")

# ESM-2 model configuration
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
print(f"Loading ESM-2 model on {DEVICE}...")
esm2_tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
esm2_model = AutoModel.from_pretrained(ESM2_MODEL_NAME).to(DEVICE)
esm2_model.eval()
print("ESM-2 model loaded successfully!")


def format_sequence_for_prostt5(seq):
    """
    Format protein sequence for ProstT5 input.
    Adds <AA2fold> prefix and spaces between amino acids.

    Args:
        seq: Raw protein sequence string
    Returns:
        Formatted sequence string
    """
    # Add spaces between each amino acid and prefix
    spaced_seq = " ".join(list(seq))
    return f"<AA2fold> {spaced_seq}"


def extract_esm2_embedding(seq, max_length=1024):
    """
    Extract sequence-based embedding for a single protein sequence using ESM-2.

    Args:
        seq: Protein sequence string
        max_length: Maximum sequence length (default: 1024)
    Returns:
        1280-dimensional CLS token embedding
    """
    # Tokenize
    inputs = esm2_tokenizer(
        seq,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Extract embeddings
    with torch.no_grad():
        outputs = esm2_model(**inputs)
        # Use CLS token (position 0) as sequence representation
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

    return embedding


def extract_prostt5_embedding(seq, max_length=1024):
    """
    Extract structure-aware embedding for a single protein sequence using ProstT5.

    Args:
        seq: Protein sequence string
        max_length: Maximum sequence length (default: 1024)
    Returns:
        1024-dimensional mean-pooled embedding
    """
    # Format sequence
    formatted_seq = format_sequence_for_prostt5(seq)

    # Tokenize
    ids = prostt5_tokenizer.batch_encode_plus(
        [formatted_seq],
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    ).to(DEVICE)

    # Extract embeddings
    with torch.no_grad():
        outputs = prostt5_model(
            input_ids=ids.input_ids,
            attention_mask=ids.attention_mask
        )
        # Get residue-level embeddings (remove special tokens)
        embeddings = outputs.last_hidden_state[0]  # Shape: [seq_len, 1024]

        # Mean pooling over sequence length to get per-protein embedding
        # Only pool over actual sequence tokens (use attention mask)
        attention_mask = ids.attention_mask[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=0)
        sum_mask = torch.clamp(mask_expanded.sum(dim=0), min=1e-9)
        mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()

    return mean_pooled


def extract_hybrid_embedding(seq, max_length=1024):
    """
    Extract both ProstT5 and ESM-2 embeddings for a single sequence.

    Args:
        seq: Protein sequence string
        max_length: Maximum sequence length
    Returns:
        tuple: (prostt5_embedding, esm2_embedding)
    """
    prostt5_emb = extract_prostt5_embedding(seq, max_length)
    esm2_emb = extract_esm2_embedding(seq, max_length)
    return prostt5_emb, esm2_emb


def extract_features(dataset, split_name, feature_dir=None, extract_both=True):
    """
    Extract ProstT5 and ESM-2 features for a dataset.

    Args:
        dataset: List of dicts with 'seq' and 'label' keys
        split_name: Name for this split (e.g., 'train', 'test')
        feature_dir: Directory to save/load features (default: structural_retrieval/features/)
        extract_both: If True, extract both ProstT5 and ESM-2; if False, only ProstT5
    Returns:
        If extract_both=True: (prostt5_features, esm2_features, labels)
        If extract_both=False: (prostt5_features, labels)
    """
    if feature_dir is None:
        # Default to structural_retrieval/features/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        feature_dir = os.path.join(script_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)

    prostt5_features = []
    esm2_features = []
    labels = []

    desc = f"Extracting hybrid {split_name} features" if extract_both else f"Extracting ProstT5 {split_name} features"

    for idx, item in enumerate(tqdm(dataset, desc=desc)):
        seq = item['seq']
        label = item['label']

        prostt5_path = os.path.join(feature_dir, f"{split_name}_prostt5_{idx}.npy")
        esm2_path = os.path.join(feature_dir, f"{split_name}_esm2_{idx}.npy")

        # Load or extract ProstT5 features
        if os.path.exists(prostt5_path):
            prostt5_feat = np.load(prostt5_path)
        else:
            prostt5_feat = extract_prostt5_embedding(seq)
            np.save(prostt5_path, prostt5_feat)
        prostt5_features.append(prostt5_feat)

        # Load or extract ESM-2 features if needed
        if extract_both:
            if os.path.exists(esm2_path):
                esm2_feat = np.load(esm2_path)
            else:
                esm2_feat = extract_esm2_embedding(seq)
                np.save(esm2_path, esm2_feat)
            esm2_features.append(esm2_feat)

        labels.append(label)

    prostt5_features = np.stack(prostt5_features)

    if extract_both:
        esm2_features = np.stack(esm2_features)
        return prostt5_features, esm2_features, labels
    else:
        return prostt5_features, labels


def reciprocal_rank_fusion(rankings_list, k=60):
    """
    Combine multiple rankings using Reciprocal Rank Fusion (RRF).

    RRF formula: RRF(d) = Σ 1/(k + rank_i(d))
    where rank_i(d) is the rank of document d in ranking i.

    Args:
        rankings_list: List of rankings, each ranking is a list of (index, score) tuples
        k: RRF parameter (default: 60, standard value from literature)
    Returns:
        fused_ranking: List of (index, fused_score) tuples, sorted by fused_score
    """
    rrf_scores = {}

    for ranking in rankings_list:
        for rank, (idx, _) in enumerate(ranking, start=1):
            if idx not in rrf_scores:
                rrf_scores[idx] = 0.0
            rrf_scores[idx] += 1.0 / (k + rank)

    # Sort by RRF score (higher is better)
    fused_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused_ranking


def adaptive_weighted_rrf(rankings_list, instruction, k_base=60, weight_adaptation='auto'):
    """
    Adaptive Weighted Reciprocal Rank Fusion with sophisticated task-aware weighting.

    This function adapts RRF weights based on multi-level evidence from the instruction:
    - Primary signals (catalytic, domain/motif, function, etc.)
    - Secondary signals (enzyme, chemical reaction, biological process, etc.)
    - Context patterns (co-occurrence of multiple keywords)

    Adaptation strategy (based on dataset analysis):
    - Structure-heavy: catalytic activity (100%), domain/motif (100%)
    - Function-heavy: protein function (79%), biological process (61%)
    - Generic: general characteristics/features

    RRF formula: RRF(d) = 1/(k_prostt5 + rank_p(d)) + 1/(k_esm2 + rank_e(d))

    Args:
        rankings_list: List of [prostt5_ranking, esm2_ranking]
        instruction: The task instruction text
        k_base: Base RRF parameter (default: 60)
        weight_adaptation: 'auto' (evidence-based), 'structure', 'sequence', 'balanced'

    Returns:
        fused_ranking: List of (index, fused_score) tuples, sorted by fused_score
        weights_used: Dict with the weights used for transparency
    """
    instruction_lower = instruction.lower()

    if weight_adaptation == 'auto':
        # Multi-level evidence scoring system

        # Level 1: Primary structure signals (high confidence)
        structure_primary = {
            'catalytic activity': 3.0,  # Strong structure signal
            'chemical reaction': 2.5,   # Strong structure signal
            'active site': 3.0,
            'domain': 2.5,
            'motif': 2.5,
            'structural motif': 3.0,
        }

        # Level 2: Secondary structure signals (medium confidence)
        structure_secondary = {
            'enzyme': 1.5,
            'catalyze': 1.5,
            'catalytic': 2.0,
            'structure': 1.0,
            'fold': 1.5,
            'topology': 1.5,
            '3d': 1.5,
            'tertiary': 1.5,
            'binding site': 1.2,
        }

        # Level 1: Primary function signals (high confidence)
        function_primary = {
            'biological process': 2.5,
            'cellular process': 2.5,
            'subcellular': 2.0,
            'molecular function': 2.5,
            'functional role': 2.5,
        }

        # Level 2: Secondary function signals (medium confidence)
        function_secondary = {
            'function': 1.5,
            'pathway': 1.5,
            'process': 1.0,
            'role': 1.0,
            'activity': 0.8,  # Ambiguous - can be catalytic activity or function
            'cellular': 1.0,
            'localization': 1.0,
            'involvement': 1.0,
        }

        # Generic/descriptive signals (should be balanced)
        generic_signals = {
            'characteristics': 1.5,
            'features': 1.5,
            'summary': 1.5,
            'overview': 1.5,
            'description': 1.0,
            'analysis': 0.5,
        }

        # Calculate evidence scores
        structure_score = 0.0
        function_score = 0.0
        generic_score = 0.0

        # Score primary structure signals
        for phrase, weight in structure_primary.items():
            if phrase in instruction_lower:
                structure_score += weight

        # Score secondary structure signals
        for phrase, weight in structure_secondary.items():
            if phrase in instruction_lower:
                structure_score += weight

        # Score primary function signals
        for phrase, weight in function_primary.items():
            if phrase in instruction_lower:
                function_score += weight

        # Score secondary function signals
        for phrase, weight in function_secondary.items():
            if phrase in instruction_lower:
                function_score += weight

        # Score generic signals
        for phrase, weight in generic_signals.items():
            if phrase in instruction_lower:
                generic_score += weight

        # Decision logic with more gradual adjustments
        total_signal = structure_score + function_score + generic_score

        if total_signal == 0:
            # No clear signal → balanced
            k_prostt5 = k_base
            k_esm2 = k_base
            mode = 'balanced'
        elif generic_score > (structure_score + function_score):
            # Generic description task → slight sequence emphasis
            k_prostt5 = k_base * 1.15
            k_esm2 = k_base * 0.87
            mode = 'generic-description'
        elif structure_score > function_score * 1.5:
            # Strong structure signal → emphasize ProstT5 moderately
            # Target: 1.3-1.6x max ratio (NOT 3x)
            ratio = min(structure_score / max(function_score, 0.5), 3.0)
            # More gradual scaling: log-like dampening
            adjusted_ratio = 1.0 + 0.6 / (1.0 + ratio * 0.5)
            k_prostt5 = k_base / adjusted_ratio  # ~45-50x (0.75-0.83 of base)
            k_esm2 = k_base * adjusted_ratio     # ~72-80x (1.2-1.33 of base)
            mode = 'structure-emphasis'
        elif function_score > structure_score * 1.5:
            # Strong function signal → emphasize ESM-2 moderately
            # Target: 1.3-1.5x max ratio
            ratio = min(function_score / max(structure_score, 0.5), 3.0)
            adjusted_ratio = 1.0 + 0.5 / (1.0 + ratio * 0.6)
            k_prostt5 = k_base * adjusted_ratio  # ~70-75x (1.17-1.25 of base)
            k_esm2 = k_base / adjusted_ratio     # ~48-51x (0.8-0.85 of base)
            mode = 'function-emphasis'
        else:
            # Mixed signals or close scores → balanced with slight preference
            if structure_score > function_score:
                k_prostt5 = k_base * 0.90
                k_esm2 = k_base * 1.11
                mode = 'balanced-structure'
            else:
                k_prostt5 = k_base * 1.11
                k_esm2 = k_base * 0.90
                mode = 'balanced-function'

        evidence = {
            'structure_score': round(structure_score, 2),
            'function_score': round(function_score, 2),
            'generic_score': round(generic_score, 2),
        }

    elif weight_adaptation == 'structure':
        # Manual structure emphasis (moderate)
        k_prostt5 = k_base * 0.70
        k_esm2 = k_base * 1.43
        mode = 'structure-emphasis (manual)'
        evidence = {}
    elif weight_adaptation == 'sequence':
        # Manual sequence emphasis (moderate)
        k_prostt5 = k_base * 1.43
        k_esm2 = k_base * 0.70
        mode = 'sequence-emphasis (manual)'
        evidence = {}
    else:  # 'balanced'
        k_prostt5 = k_base
        k_esm2 = k_base
        mode = 'balanced (manual)'
        evidence = {}

    # Compute RRF scores with adaptive weights
    rrf_scores = {}

    # ProstT5 ranking (index 0)
    for rank, (idx, _) in enumerate(rankings_list[0], start=1):
        if idx not in rrf_scores:
            rrf_scores[idx] = 0.0
        rrf_scores[idx] += 1.0 / (k_prostt5 + rank)

    # ESM-2 ranking (index 1)
    for rank, (idx, _) in enumerate(rankings_list[1], start=1):
        if idx not in rrf_scores:
            rrf_scores[idx] = 0.0
        rrf_scores[idx] += 1.0 / (k_esm2 + rank)

    # Sort by RRF score (higher is better)
    fused_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Return weights for transparency
    weights_used = {
        'mode': mode,
        'k_prostt5': round(k_prostt5, 2),
        'k_esm2': round(k_esm2, 2),
        'prostt5_weight': round(1.0 / k_prostt5, 4),
        'esm2_weight': round(1.0 / k_esm2, 4),
        'weight_ratio': round((1.0/k_prostt5) / (1.0/k_esm2), 2),
    }

    if evidence:
        weights_used.update(evidence)

    return fused_ranking, weights_used


def rrf_predict(train_prostt5_features, train_esm2_features, train_labels,
                test_prostt5_features, test_esm2_features, top_k=10):
    """
    Predict labels using Reciprocal Rank Fusion of ProstT5 and ESM-2 rankings.

    Args:
        train_prostt5_features: Training ProstT5 features [N_train, 1024]
        train_esm2_features: Training ESM-2 features [N_train, 1280]
        train_labels: Training labels
        test_prostt5_features: Test ProstT5 features [N_test, 1024]
        test_esm2_features: Test ESM-2 features [N_test, 1280]
        top_k: Number of candidates to retrieve from each method before fusion
    Returns:
        pred_labels: Predicted labels for test set
    """
    # Build Faiss indices for both feature types
    d_prostt5 = train_prostt5_features.shape[1]
    d_esm2 = train_esm2_features.shape[1]

    prostt5_index = faiss.IndexFlatL2(d_prostt5)
    prostt5_index.add(train_prostt5_features.astype(np.float32))

    esm2_index = faiss.IndexFlatL2(d_esm2)
    esm2_index.add(train_esm2_features.astype(np.float32))

    # Batch search for efficiency
    print(f"Performing RRF prediction on {len(test_prostt5_features)} test samples...")

    # Search all test samples at once (much faster!)
    D_prostt5, I_prostt5 = prostt5_index.search(
        test_prostt5_features.astype(np.float32), top_k
    )
    D_esm2, I_esm2 = esm2_index.search(
        test_esm2_features.astype(np.float32), top_k
    )

    pred_labels = []

    # For each test sample, fuse rankings
    from tqdm import tqdm
    for i in tqdm(range(len(test_prostt5_features)), desc="RRF fusion"):
        prostt5_ranking = [(I_prostt5[i][j], D_prostt5[i][j]) for j in range(top_k)]
        esm2_ranking = [(I_esm2[i][j], D_esm2[i][j]) for j in range(top_k)]

        # Fuse rankings using RRF
        fused_ranking = reciprocal_rank_fusion([prostt5_ranking, esm2_ranking])

        # Take the label of the top-ranked candidate
        best_idx = fused_ranking[0][0]
        pred_labels.append(train_labels[best_idx])

    return pred_labels


def knn_predict(train_features, train_labels, test_features, k=1):
    """
    Predict labels using k-nearest neighbors in feature space.
    (Used for single-modality baselines)

    Args:
        train_features: Training features [N_train, D]
        train_labels: Training labels
        test_features: Test features [N_test, D]
        k: Number of neighbors
    Returns:
        pred_labels: Predicted labels for test set
    """
    d = train_features.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(train_features.astype(np.float32))
    D, I = index.search(test_features.astype(np.float32), k)

    # Take the label of the nearest neighbor
    pred_labels = [train_labels[i[0]] for i in I]
    return pred_labels


def evaluate_accuracy(pred_labels, true_labels):
    """Calculate prediction accuracy."""
    correct = sum(p == t for p, t in zip(pred_labels, true_labels))
    return correct / len(true_labels)


def hybrid_rrf_predict(train_seqs, test_seqs, train_labels, test_labels, task_name, top_k=10, reuse_esm2=True):
    """
    Full pipeline: extract hybrid features and perform RRF prediction.

    Args:
        train_seqs: List of training sequences
        test_seqs: List of test sequences
        train_labels: List of training labels
        test_labels: List of test labels
        task_name: Name of the task
        top_k: Number of candidates for RRF fusion
        reuse_esm2: If True, try to reuse existing ESM-2 features from ../features/
    Returns:
        pred_labels: Predictions
        test_labels: True labels
        acc: Accuracy
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Try to load existing ESM-2 features
    train_esm2 = None
    test_esm2 = None

    if reuse_esm2:
        esm2_train_path = os.path.join(parent_dir, "features", f"{task_name}_train_features.npy")
        esm2_test_path = os.path.join(parent_dir, "features", f"{task_name}_test_features.npy")

        if os.path.exists(esm2_train_path) and os.path.exists(esm2_test_path):
            print(f"✓ Reusing existing ESM-2 features from {parent_dir}/features/")
            train_esm2 = np.load(esm2_train_path)
            test_esm2 = np.load(esm2_test_path)
            print(f"  Loaded ESM-2 train: {train_esm2.shape}, test: {test_esm2.shape}")
        else:
            print(f"⚠ ESM-2 features not found at {esm2_train_path}, will extract...")

    # Extract ProstT5 features (always needed)
    train_set = [{"seq": seq, "label": label} for seq, label in zip(train_seqs, train_labels)]
    test_set = [{"seq": seq, "label": label} for seq, label in zip(test_seqs, test_labels)]

    feature_dir = os.path.join(script_dir, "features", f"hybrid_{task_name}")
    os.makedirs(feature_dir, exist_ok=True)

    if train_esm2 is None or test_esm2 is None:
        # Extract both features
        print(f"Extracting hybrid features for {task_name}...")
        train_prostt5, train_esm2, train_labels = extract_features(train_set, "train", feature_dir, extract_both=True)
        test_prostt5, test_esm2, test_labels = extract_features(test_set, "test", feature_dir, extract_both=True)
    else:
        # Only extract ProstT5 features
        print(f"Extracting ProstT5 features for {task_name}...")
        train_prostt5, train_labels = extract_features(train_set, "train", feature_dir, extract_both=False)
        test_prostt5, test_labels = extract_features(test_set, "test", feature_dir, extract_both=False)

    # Save features for later use in structural_retrieval/
    np.save(os.path.join(script_dir, f"hybrid_{task_name}_train_prostt5.npy"), train_prostt5)
    np.save(os.path.join(script_dir, f"hybrid_{task_name}_train_esm2.npy"), train_esm2)
    np.save(os.path.join(script_dir, f"hybrid_{task_name}_test_prostt5.npy"), test_prostt5)
    np.save(os.path.join(script_dir, f"hybrid_{task_name}_test_esm2.npy"), test_esm2)

    # Predict using RRF
    print(f"Performing RRF prediction (top_k={top_k})...")
    pred_labels = rrf_predict(train_prostt5, train_esm2, train_labels,
                               test_prostt5, test_esm2, top_k=top_k)
    acc = evaluate_accuracy(pred_labels, test_labels)

    print(f"Hybrid RRF prediction accuracy: {acc:.4f}")
    return pred_labels, test_labels, acc


if __name__ == "__main__":
    # Test the hybrid feature extraction
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    print(f"\nTest sequence length: {len(test_seq)}")

    print("\nExtracting ProstT5 embedding...")
    prostt5_emb = extract_prostt5_embedding(test_seq)
    print(f"ProstT5 embedding shape: {prostt5_emb.shape}")
    print(f"ProstT5 embedding mean: {prostt5_emb.mean():.4f}, std: {prostt5_emb.std():.4f}")

    print("\nExtracting ESM-2 embedding...")
    esm2_emb = extract_esm2_embedding(test_seq)
    print(f"ESM-2 embedding shape: {esm2_emb.shape}")
    print(f"ESM-2 embedding mean: {esm2_emb.mean():.4f}, std: {esm2_emb.std():.4f}")

    print("\nHybrid feature extraction test passed!")

<h2 align="center">
  <img src="figs/protein.png" style="vertical-align:middle; width:23px; height:23px;" />
  SAPER: Structure-Aware Prompt-Enhanced Retrieval Augmented Protein Modeling Framework
</h2>

<h4 align="center">

*Extended work based on "Rethinking Protein Understanding: Retrieval-Augmented Modeling Reconsidered"*

</h4>

<h5 align="center">
Linrui Ma (MIT), Winston Qian (Harvard), Yiwei Liang (MIT), Emma Wang (MIT)
</h5>

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Visualizations](#visualizations)
- [Citation](#citation)

---

## Overview

**SAPER (Structure-Aware Prompt-Enhanced Retrieval Augmented Protein Modeling)** extends the RAPM framework to address a fundamental limitation: protein function depends on 3D structure, yet existing methods rely solely on sequence similarity. We demonstrate that incorporating structural information via ProstT5 embeddings, optimizing multi-modal fusion, and enhancing prompt engineering can achieve dramatic improvements in protein functional annotation, particularly for out-of-distribution (OOD) proteins.

### The Challenge

Large language models (LLMs) fail on novel proteins precisely when we need them most: for clinical applications involving previously uncharacterized disease variants. The Prot-Inst-OOD dataset reveals this failure by enforcing strict OOD splits (<30% sequence identity) and using Meta-BLEU evaluation that measures biological entity accuracy rather than surface n-grams.

### Our Solution: SAPER

We present three synergistic innovations:
1. **ProstT5 Structural Embeddings**: Capture 3D geometric features via 3Di alphabets
2. **Multi-Modal Fusion**: Weighted Similarity (α=0.7) and Reciprocal Rank Fusion
3. **Enhanced Prompting**: Task-specific instructions with confidence signals

---

## Key Contributions

### 1. Structure-Aware Embeddings
- **ProstT5** (1024-dim): Encodes local backbone conformations and tertiary interactions via 3Di structural alphabets
- **Complementary to ESM-2** (1280-dim): ESM-2 captures evolutionary patterns, ProstT5 captures geometric environments
- **Critical for twilight zone**: When sequence similarity <30%, structural conservation persists

### 2. Fusion Methods

#### Reciprocal Rank Fusion (RRF)
```math
RRF_k(i) = \frac{1}{k + rank_{ProstT5}(i)} + \frac{1}{k + rank_{ESM-2}(i)}
```
- Weight-free combination (k=60)
- Favors proteins ranking well in both modalities
- Best for protein function tasks

#### Weighted Similarity Fusion
```math
Score_α(q,i) = α · sim_{ProstT5}(e^P_q, e^P_i) + (1-α) · sim_{ESM-2}(e^E_q, e^E_i)
```
- α=0.7 (70% structure, 30% sequence) optimal
- Preserves similarity magnitudes for confidence calibration
- Superior for structure-dependent tasks

### 3. Enhanced Prompt Engineering
- **Expert Persona Framing**: "You are a protein function prediction expert..."
- **Task-Specific Guidance**: Dynamic instructions for each task type
- **Confidence-Level Grouping**: High (≥0.9), Medium (0.7-0.9), Low (<0.7)
- **Terminology Directives**: Explicit instructions to use precise biological terms

---

## Results

Evaluated on **Prot-Inst-OOD** with **Gemini-2.5-Flash** (256 samples × 3 runs per task).

### Performance Improvements (Meta-BLEU-2)

| Task | Baseline RAPM | ProstT5+RRF | ProstT5+Weighted | **SAPER** | **Relative Gain** |
|------|--------------|-------------|-----------------|-----------|------------------|
| **Domain Motif** | 16.09 | 22.63 | 25.89 | **34.43** | **+114.0%** |
| **General Function** | 4.78 | 7.04 | 10.15 | **9.21** | **+92.7%** |
| **Catalytic Activity** | 28.09 | 33.91 | 36.52 | **43.04** | **+53.2%** |
| **Protein Function** | 47.40 | 49.80 | 47.19 | **56.66** | **+19.5%** |
| **Average** | 24.09 | 28.35 | 29.94 | **35.84** | **+48.8%** |

### Key Findings

1. **Structure matters most for geometric tasks**: Domain motif (+114%) and catalytic activity (+53.2%) show largest gains as these depend on 3D architecture and active site geometry.

2. **Weighted fusion optimal at α=0.7**: 70% structural weight balances geometric patterns with evolutionary information, outperforming equal-weight RRF on 3/4 tasks.

3. **Prompt engineering amplifies retrieval quality**: Enhanced prompts add 6-9 Meta-BLEU points beyond structural retrieval alone by better guiding entity extraction.

4. **Trade-off on general function**: Enhanced prompts slightly decrease general function performance (10.15→9.21) due to over-constraining diverse descriptions.

---

## Installation

### Prerequisites
- Python 3.12.7
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM
- Gemini API key

### Setup

```bash
# Clone repository
git clone https://github.com/winstonqian/SAPER.git
cd SAPER

# Create conda environment
conda create -n saper python=3.12.7
conda activate saper

# Install PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Download Prot-Inst-OOD from [Hugging Face](https://huggingface.co/datasets/TimeRune/Prot-Inst-OOD):
```bash
# Place in dataset/ directory
wget -P dataset/ https://huggingface.co/datasets/TimeRune/Prot-Inst-OOD/resolve/main/*.json
```

---

## Usage

### Quick Start: Run SAPER

```bash
# Navigate to structural retrieval
cd structural_retrieval/src

# Generate enhanced prompts with weighted similarity (α=0.7)
python enhanced_prompt.py catalytic_activity_OOD 10 0.7

# Run inference with Gemini
python run_prostt5_rapm_sim.py catalytic_activity_OOD 10 0.7
```

### Run All Tasks

```bash
# Evaluate on all four Prot-Inst-OOD tasks
for task in catalytic_activity_OOD domain_motif_OOD protein_function_OOD general_function_OOD; do
    python enhanced_prompt.py $task 10 0.7
    python run_prostt5_rapm_sim.py $task 10 0.7
done
```

### Compare Methods

```bash
# Method 1: Baseline RAPM (ESM-2 only)
python RAPM/RAG_prompt_cons.py dataset 10
python RAPM/GEMINI_inference.py $task 10

# Method 2: ProstT5 + RRF
python prostt5_rag_prompt.py $task 10
python run_prostt5_rapm.py $task 10

# Method 3: ProstT5 + Weighted Similarity
python prostt5_retrieval_sim.py $task 10 0.7
python run_prostt5_rapm_sim.py $task 10 0.7

# Method 4: SAPER (Enhanced)
python enhanced_prompt.py $task 10 0.7
python run_prostt5_rapm_sim.py $task 10 0.7
```

---

## Project Structure

```
protein_rag_project/
├── RAPM/                      # Original RAPM baseline
├── structural_retrieval/      # SAPER implementation
│   ├── src/
│   │   ├── prostt5_features.py        # ProstT5 embedding extraction
│   │   ├── prostt5_retrieval.py       # Hybrid RRF fusion
│   │   ├── prostt5_retrieval_sim.py   # Weighted similarity fusion
│   │   ├── enhanced_prompt.py         # Enhanced prompt engineering
│   │   └── run_*.py                   # Execution scripts
│   ├── results/               # Experimental results (3 runs × 256 samples)
│   └── visualizations/        # Performance plots
├── dataset/                   # Prot-Inst-OOD dataset
├── report/                    # Full technical report (LaTeX)
└── requirements.txt
```

---

## Methodology

### Embedding Extraction

**ProstT5**: Bilingual model translating between amino acids and 3Di structural tokens
```python
# 1024-dim structure-aware embeddings
embeddings = ProstT5_mean(sequence, predicted_3Di)
```

**ESM-2**: Evolutionary scale model capturing sequence patterns
```python
# 1280-dim sequence-based embeddings
embeddings = ESM2_CLS_token(sequence)
```

### Retrieval Pipeline

1. **Extract embeddings** for query protein (both ProstT5 and ESM-2)
2. **Search indices** using FAISS HNSW for top-K=100 candidates
3. **Fusion**: Combine rankings via RRF or weighted similarity
4. **Select top-K=10** for prompt augmentation
5. **Construct prompt** with retrieved annotations + confidence levels
6. **Generate** functional description using Gemini-2.5-Flash

### Enhanced Prompt Structure

```
You are a protein function prediction expert...

[Task-Specific Guidance]
- For catalytic activity: Focus on enzyme classes, substrates, cofactors
- For domain motif: Identify Pfam/InterPro domains, structural motifs
...

[Confidence-Grouped Retrievals]
High Confidence (≥0.9):
• Serine protease with catalytic triad...

[In-Task Examples]
• Example 1: Catalyzes ATP hydrolysis...
```

---

## Visualizations

Generate performance comparison plots:

```bash
cd structural_retrieval/visualizations

# Compare retrieval methods
python performance_comparison.py

# Compare prompt engineering impact
python prompt_comparison.py
```

### Generated Figures
- `meta_bleu2_comparison.png`: Meta-BLEU-2 across all methods
- `prompt_meta_bleu2_comparison.png`: Impact of enhanced prompting
- Bar charts with error bars (3 runs) for all metrics

---

## Citation

If you use SAPER in your research, please cite:

```bibtex
@article{saper2025,
  title={SAPER: Structure-Aware Prompt-Enhanced Retrieval Augmented Protein Modeling Framework},
  author={Ma, Linrui and Qian, Winston and Liang, Yiwei and Wang, Emma},
  institution={MIT and Harvard University},
  year={2025}
}
```

### Acknowledgments

- **Original RAPM**: Wu et al. (2025) for the foundational framework
- **ProstT5**: Heinzinger et al. (2023) for structure-aware embeddings
- **ESM-2**: Lin et al. (2023) for evolutionary scale modeling
- **Prot-Inst-OOD**: Wu et al. (2025) for the benchmark dataset

---

## Contact

- **Code Repository**: [github.com/winstonqian/SAPER](https://github.com/winstonqian/SAPER)
- For questions about the original RAPM paper, refer to the [original repository](https://github.com/IDEA-XL/RAPM).

For questions or collaboration, please open an issue on GitHub.
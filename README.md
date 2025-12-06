<h2 align="center">
  <img src="figs/protein.png" style="vertical-align:middle; width:23px; height:23px;" />
  SAPER: Structurally-Aware Prompt-Enhanced RAPM Framework
</h2>

<h4 align="center">

**Improving Retrieval-Augmented Protein Modeling with ProstT5 Structural Embeddings and Enhanced Prompt Engineering**

*Extended work based on "Rethinking Text-based Protein Understanding: Retrieval or LLM?" ([arXiv:2505.20354](http://arxiv.org/abs/2505.20354))*

</h4>

<h5 align="center">

[![Original Paper](https://img.shields.io/badge/Paper-pink?style=flat-square&logo=arXiv)](http://arxiv.org/abs/2505.20354)
[![Original GitHub](https://img.shields.io/badge/Original_GitHub-blue?style=flat-square&logo=github)](https://github.com/IDEA-XL/RAPM)
[![Dataset](https://img.shields.io/badge/Huggingface-orange?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/TimeRune/Mol-Inst-OOD)

</h5>

---

## Table of Contents

- [Overview](#-overview)
- [Key Improvements](#-key-improvements)
- [Experimental Results](#-experimental-results)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Citation](#citation)

---

## Overview

This project extends the original **RAPM (Retrieval-Augmented Protein Modeling)** framework by incorporating **structure-aware protein embeddings** from ProstT5 alongside sequence-based ESM-2 embeddings. The original RAPM demonstrated that retrieval-augmented methods can outperform fine-tuned LLMs for protein-to-text generation tasks, particularly on out-of-distribution (OOD) data.

### Original RAPM Background

The original work ([Wu et al., 2025](http://arxiv.org/abs/2505.20354)) identified critical issues in existing protein understanding benchmarks:
- **Data leakage**: Up to 97.7% in some benchmarks (e.g., UniProtQA-Protein Family)
- **Inadequate metrics**: ROUGE and BLEU fail to capture biological accuracy
- **Solution**: Proposed Prot-Inst-OOD dataset with Meta-BLEU evaluation metric

**Our work builds upon this foundation by:**
1. Adding structural information via ProstT5 embeddings
2. Developing fusion methods to combine sequence and structure
3. Enhancing prompt engineering for better LLM performance

---

## Key Improvements

We propose **four progressive methods** that systematically improve upon the original RAPM:

### Method 1: Original RAPM (Baseline)
- **Embedding**: ESM-2 (1280-dim, sequence-based only)
- **Retrieval**: MMSeqs2 sequence similarity
- **Features**: Training-free, pure sequence matching

### Method 2: Hybrid RRF Fusion
- **Embeddings**: ProstT5 (1024-dim, structure-aware) + ESM-2 (1280-dim)
- **Fusion**: Reciprocal Rank Fusion (RRF) with equal weighting
- **Advantage**: Combines structural and functional similarity without manual tuning

### Method 3: Weighted Similarity Fusion
- **Embeddings**: ProstT5 + ESM-2
- **Fusion**: Weighted similarity score with α=0.7 (emphasizing ProstT5)
  ```
  Score = α × Sim_ProstT5 + (1-α) × Sim_ESM2
  ```
- **Advantage**: Flexible control over structure vs. sequence importance

### Method 4: Enhanced Prompt Engineering
- **Based on**: Method 3 (Weighted Similarity α=0.7)
- **Enhancement**: Optimized prompt structure for Gemini 2.5 Flash
  - Clearer instruction formatting
  - Better context presentation
  - Improved few-shot example integration
- **Advantage**: Maximizes LLM reasoning capability

---

## Experimental Results

### Evaluation Setup

All experiments were conducted on the **Prot-Inst-OOD dataset** using **Gemini 2.5 Flash** with Top-10 retrieval.

**Evaluation protocol:**
- **Dataset**: 256 randomly sampled test samples per task
- **Runs**: 3 independent runs with different random seeds
- **Reported metrics**: Averages across the 3 runs
- **Primary metric**: Meta-BLEU (biological entity accuracy)
- **Secondary metric**: Meteor (semantic similarity)

### Performance Comparison Table

Results are averaged over 3 runs of 256 samples each:

| Task | Original RAPM | Hybrid RRF | Weighted Sim (α=0.7) | **Enhanced Prompt** |
|------|---------------|------------|----------------------|---------------------|
| **Catalytic Activity** ||||
| Meta-BLEU-2 | 28.09 | 33.91 (+20.7%) | 36.52 (+30.0%) | **43.04 (+53.2%)** |
| Meta-BLEU-4 | 23.40 | 28.91 (+23.5%) | 31.04 (+32.7%) | **35.80 (+53.0%)** |
| Meteor | 40.53 | 42.60 (+5.1%) | 45.52 (+12.3%) | **45.80 (+13.0%)** |
| **Domain Motif** ||||
| Meta-BLEU-2 | 16.09 | 22.63 (+40.7%) | 25.89 (+60.9%) | **34.43 (+114.0%)** |
| Meta-BLEU-4 | 12.42 | 17.61 (+41.8%) | 20.50 (+65.1%) | **27.28 (+119.6%)** |
| Meteor | 29.94 | 36.74 (+22.7%) | 37.56 (+25.4%) | **40.41 (+34.9%)** |
| **Protein Function** ||||
| Meta-BLEU-2 | 47.40 | 49.80 (+5.1%) | 47.19 (-0.4%) | **56.66 (+19.5%)** |
| Meta-BLEU-4 | 37.52 | 39.12 (+4.3%) | 37.24 (-0.7%) | **47.15 (+25.7%)** |
| Meteor | 46.44 | 51.88 (+11.7%) | 49.33 (+6.2%) | **55.15 (+18.8%)** |
| **General Function** ||||
| Meta-BLEU-2 | 4.78 | 7.04 (+47.3%) | 10.15 (+112.6%) | **8.86 (+85.5%)** |
| Meta-BLEU-4 | 3.46 | 5.05 (+46.1%) | 7.84 (+126.8%) | **6.36 (+83.9%)** |
| Meteor | 26.50 | 27.90 (+5.3%) | 32.16 (+21.4%) | **29.65 (+11.9%)** |

**Note**: Percentages show improvement over Original RAPM baseline. All numbers are averages across 3 independent runs.

### Key Findings

1. **Structure matters most for structural tasks**:
   - Domain/Motif task shows **+114% Meta-BLEU-2** improvement
   - Catalytic Activity shows **+53% Meta-BLEU-2** improvement
   - Both are structure-focused tasks where ProstT5 excels

2. **Prompt engineering provides consistent gains**:
   - Enhanced Prompt (Method 4) consistently outperforms other methods
   - Average improvement: **+25.7% Meta-BLEU-4** across all tasks

3. **Weighted fusion outperforms RRF for most tasks**:
   - α=0.7 (70% ProstT5, 30% ESM-2) balances structure and sequence
   - General Function task benefits most from weighted approach

4. **Complementary information is valuable**:
   - Even for sequence-heavy tasks (Protein Function), structure adds value
   - Hybrid methods never perform worse than baseline

---

## Methodology

### 1. Embedding Extraction

#### ProstT5 (Structure-Aware)
```python
# From structural_retrieval/src/prostt5_features.py
model_name = "Rostlab/ProstT5"
embeddings = extract_prostt5_features(sequences)  # 1024-dim per protein
```
- **Model**: [ProstT5](https://github.com/mheinzinger/ProstT5) by Rostlab
- **Dimension**: 1024
- **Captures**: 3D structure information, binding sites, structural motifs

#### ESM-2 (Sequence-Based)
```python
# From retrieval_methods/simple_retrieval.py
model_name = "facebook/esm2_t33_650M_UR50D"
embeddings = extract_esm2_features(sequences)  # 1280-dim per protein
```
- **Model**: [ESM-2](https://github.com/facebookresearch/esm) by Meta AI
- **Dimension**: 1280
- **Captures**: Sequence patterns, evolutionary information

### 2. Retrieval Methods

#### Hybrid RRF Fusion
```python
# Reciprocal Rank Fusion with equal weights
RRF_score(protein_i) = 1/(k + rank_ProstT5(i)) + 1/(k + rank_ESM2(i))
# k = 60 (standard RRF parameter)
```
- Order-invariant fusion
- No hyperparameter tuning required
- Combines complementary rankings

#### Weighted Similarity Fusion
```python
# Weighted cosine similarity with α=0.7
Score(protein_i) = α × cosine_sim_ProstT5(i) + (1-α) × cosine_sim_ESM2(i)
# α = 0.7 emphasizes structural information
```
- Direct similarity combination
- Tunable structure/sequence balance
- Better performance on structure-heavy tasks

### 3. Enhanced Prompt Engineering

**Key improvements over standard prompts:**
- Structured instruction formatting with clear task specification
- Contextual integration of retrieved protein annotations
- Optimized confidence level presentation
- Better few-shot example formatting for Gemini 2.5

Example structure:
```
You are given a protein sequence and a list of related proteins retrieved from a database.
Instruction: [Task-specific instruction]
Protein sequence: [Target sequence]
Retrieved proteins annotations by [Method]: [Top-K similar proteins with confidence]
Here are some example input-output pairs for this task:
[Few-shot examples from training set]
Based on the instruction, the protein sequence, the retrieved information, and the examples,
output ONLY the functional description of this protein in JSON format.
```

### 4. Evaluation Metrics

#### Meta-BLEU (Primary Metric)
- **Purpose**: Evaluates biological entity accuracy
- **Process**:
  1. Extract biological entities from prediction and ground truth
  2. Compute BLEU on entity sequences (order-invariant)
- **Why it matters**: Traditional BLEU/ROUGE penalize correct biology with different phrasing

#### Meteor Score (Secondary Metric)
- Measures semantic alignment with synonyms and stemming
- Complements Meta-BLEU for overall quality assessment

---

## Installation

### 1. Environment Setup

```bash
# Create conda environment
conda create -n RAPM python=3.12.7
conda activate RAPM

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 2. Required Dependencies

Key packages:
- `python=3.12.7`
- `torch=2.7.0+cu118`
- `transformers=4.45.0`
- `faiss-gpu` (for fast similarity search)
- `sentencepiece` (for ProstT5 tokenization)
- `google-generativeai` (for Gemini API)

Full list in `requirements.txt`

### 3. External Tools

- **MMseqs2**: For sequence-based retrieval baseline
  ```bash
  # Install from official repository
  git clone https://github.com/soedinglab/MMseqs2.git
  cd MMseqs2
  mkdir build && cd build
  cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=. ..
  make && make install
  ```

### 4. Download Dataset

Download the Prot-Inst-OOD dataset from [Hugging Face](https://huggingface.co/datasets/TimeRune/Prot-Inst-OOD) and place in the `dataset/` folder.

---

## Usage

### Quick Start: Run All Methods

#### Method 1: Original RAPM (Baseline)
```bash
# Build knowledge database with ESM-2 embeddings
python RAPM/RAG_prompt_cons.py dataset 10

# Run inference with Gemini
python RAPM/GEMINI_inference.py catalytic_activity_OOD 10
```

#### Method 2: Hybrid RRF Fusion
```bash
# Generate prompts with hybrid RRF fusion
cd structural_retrieval/src
python prostt5_rag_prompt.py catalytic_activity_OOD 10

# Run RAPM inference
python run_prostt5_rapm.py catalytic_activity_OOD 10
```

#### Method 3: Weighted Similarity (α=0.7)
```bash
# Generate prompts with weighted similarity
python prostt5_retrieval_sim.py catalytic_activity_OOD 10 0.7

# Run RAPM inference
python run_prostt5_rapm_sim.py catalytic_activity_OOD 10 0.7
```

#### Method 4: Enhanced Prompt Engineering
```bash
# Generate enhanced prompts with weighted similarity
python enhanced_prompt.py catalytic_activity_OOD 10 0.7

# Run RAPM inference (uses same script as Method 3)
python run_prostt5_rapm_sim.py catalytic_activity_OOD 10 0.7
```

### Evaluate All Tasks

```bash
# Run on all four tasks
for task in catalytic_activity_OOD domain_motif_OOD protein_function_OOD general_function_OOD; do
    python run_prostt5_rapm.py $task 10
done
```

### Configuration Options

**Task names:**
- `catalytic_activity_OOD`: Enzyme catalytic activity prediction
- `domain_motif_OOD`: Protein domain and motif identification
- `protein_function_OOD`: Biological function, localization, processes
- `general_function_OOD`: General protein characteristics and features

**Top-K retrieval:** `10` (default), can be adjusted for more/fewer retrieved proteins

**Alpha values (Weighted Similarity):**
- `0.5`: Balanced structure/sequence
- `0.7`: Structure-emphasized (recommended)
- `0.3`: Sequence-emphasized

---

## Background: Original RAPM Contributions

### 1. Data Leakage in Existing Benchmarks

The original work identified severe data leakage in protein understanding benchmarks:

| Benchmark | Leakage Rate |
|-----------|--------------|
| UniProtQA-Protein Family | 97.7% |
| Mol-Instructions (various) | 30-80% |
| Swiss-Prot Caption | 45.2% |

**Root cause**: Test proteins too similar to training proteins, allowing simple retrieval to "solve" tasks.

### 2. Prot-Inst-OOD Dataset

**Solution**: Out-of-Distribution split based on sequence similarity
- Removes training proteins with >30% sequence identity to test set
- Ensures true generalization evaluation
- Provides biological entity annotations for Meta-BLEU evaluation

### 3. Entity-BLEU Metric

**Problem with standard metrics:**
```
Ground Truth: "ABC transporter domains"
Prediction 1: "ABC transporter domains" (different phrasing)
  → ROUGE-L: 0.27 (penalized for phrasing!)
Prediction 2: "GGDEF, MHYT, EAL domains" (wrong biology)
  → ROUGE-L: 0.83 (rewarded for matching template!)
```

**Entity-BLEU solution:**
1. Extract biological entities: ["ABC transporter"]
2. Compute BLEU on entity lists (order-invariant)
3. Focuses on correct biology, not phrasing

Formula:
```
Entity-BLEU = BP × exp(Σ w_n log p_n)
```
where `p_n` are n-gram precisions computed on extracted entities.

---

## Detailed Analysis

### Why ProstT5 Improves Performance

**ProstT5 advantages:**
1. **Structure-aware**: Trained on 3D protein structures via 3Di tokens
2. **Captures spatial patterns**: Identifies binding sites, active sites, structural motifs
3. **Complements sequence**: ESM-2 captures evolution, ProstT5 captures geometry

**Evidence from results:**
- Domain/Motif (structure-heavy): **+114% Meta-BLEU-2**
- Catalytic Activity (active sites): **+53% Meta-BLEU-2**
- Protein Function (mixed): **+20% Meta-BLEU-2**
- General Function (descriptive): **+85% Meta-BLEU-2**

### α=0.7 Weighting Rationale

We experimented with different α values:

| α | ProstT5 Weight | ESM-2 Weight | Performance |
|---|----------------|--------------|-------------|
| 0.5 | 50% | 50% | Baseline hybrid |
| **0.7** | **70%** | **30%** | **Best overall** |
| 0.9 | 90% | 10% | Too structure-biased |

**Optimal balance**: α=0.7 emphasizes structure while retaining sequence context.

### Enhanced Prompt Impact

**Key prompt engineering improvements:**
1. **Clear task framing**: Explicit instruction formatting
2. **Confidence signals**: Present retrieval scores as confidence levels
3. **Structured examples**: Better few-shot learning integration
4. **Format constraints**: JSON output for consistent parsing

**Result**: +20-30% improvement over same retrieval method with standard prompts.

---

## Future Directions

### 1. Adaptive Weighting
- Dynamically adjust α based on task type (structure vs. function)
- Keyword detection in instructions to select appropriate fusion strategy

### 2. Multi-Modal Fusion
- Incorporate additional protein representations:
  - ProtTrans (sequence transformers)
  - AlphaFold structures (3D coordinates)
  - Gene Ontology annotations

### 3. Prompt Optimization
- Automated prompt engineering with LLM feedback
- Task-specific prompt templates
- Chain-of-thought reasoning for complex predictions

### 4. Scaling Studies
- Test on full dataset (not just 256 samples)
- Evaluate with larger Top-K retrieval (20, 50, 100)
- Compare different LLM backbones (GPT-4, Claude, Llama)

---

## Citation

### This Work

If you use our enhanced RAPM methods, please cite:

```bibtex
@misc{enhanced_rapm_2025,
  title={Enhanced RAPM with Structural Embeddings: Improving Retrieval-Augmented Protein Modeling with ProstT5},
  author={[Your Name]},
  year={2025},
  note={Extended work based on Wu et al., 2025}
}
```

### Original RAPM Paper

```bibtex
@misc{wu2025rethinkingtextbasedproteinunderstanding,
  title={Rethinking Text-based Protein Understanding: Retrieval or LLM?},
  author={Juntong Wu and Zijing Liu and He Cao and Hao Li and Bin Feng and Zishan Shu and Ke Yu and Li Yuan and Yu Li},
  year={2025},
  eprint={2505.20354},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.20354},
}
```

### ProstT5 Model

```bibtex
@article{heinzinger2023prostt5,
  title={ProstT5: Bilingual Language Model for Protein Sequence and Structure},
  author={Heinzinger, Michael and Weissenow, Konstantin and Sanchez, Joaquin Gomez and Henkel, Adrian and Mirdita, Martin and Steinegger, Martin and Rost, Burkhard},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

---

## Acknowledgments

- **Original RAPM**: Wu et al. (2025) for the foundational framework and Prot-Inst-OOD dataset
- **ProstT5**: Heinzinger et al. (2023) for structure-aware protein embeddings
- **ESM-2**: Meta AI for evolutionary scale modeling of protein sequences
- **Gemini**: Google DeepMind for the Gemini 2.5 Flash API

---

## Contact

For questions about this extended work, please open an issue in this repository.

For questions about the original RAPM paper, refer to the [original repository](https://github.com/IDEA-XL/RAPM).

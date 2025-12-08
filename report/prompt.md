You are an expert academic researcher and technical writer specializing in AI for Medicine. 

Your task is to write a rigorous 10-page (double-spaced) course project report based on the code and documentation in this current directory (SAPER project).

Do not write the full report yet. Just analyze the repository and provide the outline. However, the current repository could be a bit too messy, and include too many unnecessary files, so you can use https://github.com/winstonqian/SAPER as a reference.

First, read README.md and examine the repository (https://github.com/winstonqian/SAPER) to understand:
1. The Task and its Significance;
2. The Methods and improvements;
3. The Results;

Once you have analyzed the files, please output a detailed **Report Outline** for the following 4 sections. Estimate the word count needed for each to reach a total of ~2,500 words (approx. 10 pages double-spaced):

1. Introduction & Background:
    - Summarize the paper we're based on: https://arxiv.org/html/2505.20354, focusing on the following aspects:
        - The task
        - The previous methods (LLMs, simple retrieval), and how simple retrieval outperforms LLMs
        - The problem with the previous dataset and how Pro-Inst-OOD addresses it
        - The problem with the previous metric and how Entity-BLEU was introduced
        - The approach and architecture of the paper: RAPM
    This part should not be too long, but comprehensive. Capture all the key conclusions and innovations of the paper.
2. Methods (Architecture & Key improvements)
    - Improvement 1: Adding Structural Embeddings using ProstT5 https://github.com/mheinzinger/ProstT5
    - Improvement 2: Using rank fusion techniques (Hybrid RRF) and Weighted Sum of Similarity Scores
    - Improvement 3: Prompt Enhancement
3. Results (Quantitative metrics, qualitative visualizations description)
    - Replication with Gemini-2.5-flash due to limited resources
    - Effect of Improvement 1 & 2 (Either using Hybrid RRF or Weighted Sum of Similarity Scores)
    - Effect of Improvement 3 over Improvement 1 & 2
    The graphs are stored in the folder figs.
4. Discussion (Analysis of results, limitations, future work)
    - Analysis
    - Limitations: Gemini-2.5-flash is not the best-performing LLM for RAPM, but used due to resources limitations; etc.
    - Future work: Evaluate on GPT-4.1 and SOTA LLMs; etc.
5. References
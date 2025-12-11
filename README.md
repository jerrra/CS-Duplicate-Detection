# Scalable Product Duplicate Detection (MSMP+)

## Project Overview
This repository contains the Python implementation for the paper **"Model Word Pruning and Screen Size Selection in MSMP+"**.

The project implements a scalable duplicate detection pipeline TVs. It builds upon the Multi-component Similarity Method (MSM) by introducing two key efficiency improvements:
1.  **Model Word Pruning:** Removing high-frequency tokens (appearing in >10% of products) from the LSH binary vector representations.
2.  **Screen Size Selection:** A strict blocking pass that filters candidate pairs based on extracted screen diagonal size (inches).

The code performs a full evaluation using **bootstrapping** (100 iterations) to generate performance metrics ($F_1$, Pair Completeness, Pair Quality) and efficiency statistics.
To run this code, you need Python 3, numpy and matplotlib, and change the dataset path in the main code. The dataset is also in this repo. 

## Repository Structure

├── clustering.py       # Logic for MSM clustering and "can_merge" constraints
├── evaluation.py       # Metrics calculation (PC, PQ, F1) and bootstrapping logic
├── lsh.py              # Implementation of Locality Sensitive Hashing 
├── main code.py        # Entry point: runs the full pipeline and generates results
├── minhashing.py       # Minhash signature generation
├── model_words.py      # Feature extraction (Model Words, Brands, Inch parsing)
├── utils.py            # Data loading, binary vector creation, and plotting
└──  results.txt         # Log of the results run



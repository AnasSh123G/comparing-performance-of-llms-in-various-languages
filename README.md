# Comparing Performance of LLMs in Various Languages

**Based on the Master's Thesis: "Comparing Performance of LLMs in Various Languages" (September 2025).**

As large language models (LLMs) see unprecedented daily use across the globe, testing their performance across diverse languages and evaluating the effects of acceleration methods has become critical. This project accompanies the thesis research exploring how techniques like **pruning** and **quantization** (used to fit large models into limited hardware and speed up inference) affect a multilingual model's behavior, quality, and biases in different languages and scenarios.

This repository contains the suite of evaluation scripts built to benchmark these LLMs originally across English, Arabic, and German.

## Features

The project includes three main evaluation pipelines:

1. **CrowS-Pairs Bias Evaluation (`crowspairs_eval.py`)**
   Measures stereotypical bias by comparing the log-probabilities and perplexity of sentence pairs (more vs. less stereotypical).
2. **Discrim-Eval Bias Evaluation (`discrim_eval.py`)**
   Measures discriminatory bias by computing Yes/No answer probabilities for decision-making scenarios across different demographic groups.
3. **MMLU Accuracy Evaluation (`mmlu_eval.py`)**
   Evaluates model accuracy on multiple-choice questions across various subjects (Massive Multitask Language Understanding).

Additionally, the project supports:
*   **Model Quantisation:** Easily evaluate models in 4-bit or 8-bit precision to reduce VRAM usage.
*   **Model Pruning:** Apply structured or unstructured L1 pruning to Llama-style MLP blocks before evaluation.
*   **Multilingual Support:** Specific datasets and prompts are tailored for English, Arabic (including diacritised datasets), and German.

## Usage

A unified CLI is provided via `main.py` to run any of the benchmarks.

### CrowS-Pairs
```bash
python main.py crowspairs --model meta-llama/Llama-3-8B --dataset datasets/crows_pairs.csv
```

### Discrim-Eval
```bash
python main.py discrim --model meta-llama/Llama-3-8B --lang EN
```

### MMLU
```bash
python main.py mmlu --model meta-llama/Llama-3-8B --lang AR --quant 4
```

### Global Options
All commands support the following options:
*   `--model`: HuggingFace model name or local path (Required)
*   `--quant`: Quantisation level (`none`, `4`, `8`) (Default: `none`)
*   `--prune`: Pruning fraction, e.g., 0.2 for 20% (Default: `0.0`)
*   `--structured` / `--no-structured`: Use structured pruning (Default: `True`)
*   `--devices`: Comma-separated CUDA device IDs (Default: `'2'`)
*   `--seed`: Random seed for reproducibility (Default: `42`)

## Project Structure
*   `main.py`: The main CLI entry point.
*   `utils.py`: Shared utilities for device configuration, reproducibility, model loading, and GPU memory management.
*   `pruning.py`: Utilities for structured and unstructured neuron-pair pruning.
*   `*_eval.py`: The individual evaluation scripts.

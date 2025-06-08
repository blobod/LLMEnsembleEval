# LLMEnsembleEval

A modular framework for evaluating Large Language Model ensembles using the Generation As Classification (GAC) strategy, with seamless integration to lm-evaluation-harness.

## Overview

This project implements the GAC ensemble method for combining multiple LLMs at the token level, enabling improved performance on various NLP benchmarks. The framework features multi-GPU support, vocabulary unification across different tokenizers, and standardized evaluation protocols.

## Architecture

- **`ensemble.py`**: Core ensemble logic with vocabulary unification and multi-GPU model loading
- **`wrapper.py`**: Integration adapter for lm-evaluation-harness with benchmark-specific handlers

## Key Features

- **Multi-GPU Support**: Automatic distribution of models across available CUDA devices
- **Vocabulary Unification**: Handles different tokenization schemes (Ġ vs ▁ prefixes)
- **Standardized Evaluation**: Compatible with lm-evaluation-harness benchmarks
- **Modular Design**: Clean separation between ensemble logic and evaluation interface

## Supported Benchmarks

- MMLU (Massive Multitask Language Understanding)
- PIQA (Physical Interaction QA)
- ARC Challenge (AI2 Reasoning Challenge)
- WinoGrande

## Configuration

Models are configured via JSON files:

```json
{
  "model_names": ["model1", "model2"],
  "tokenizer_names": ["model1", "model2"],
  "devices": ["cuda:0", "cuda:1"],
  "weights": [0.5, 0.5]
}
```

## Status

**Work in Progress** - This repository is under active development.

## Related Work

Based on the paper "Breaking the Ceiling of the LLM Community by Treating Token Generation as a Classification for Ensembling" and implements findings from research on LLM ensemble effectiveness patterns.

---

*More documentation and examples coming soon.*

# LLMEnsembleEval

A modular framework for evaluating Large Language Model ensembles using the Generation As Classification (GAC) strategy, with seamless integration to lm-evaluation-harness.

## Overview

This project implements the GAC ensemble method for combining multiple LLMs at the token level, enabling improved performance on various NLP benchmarks. The framework features multi-GPU support, universal vocabulary unification across different tokenizers, and standardized evaluation protocols.

## Architecture

- **`ensemble.py`**: Core ensemble logic with universal vocabulary unification and multi-GPU model loading
- **`wrapper.py`**: Integration adapter for lm-evaluation-harness with benchmark-specific handlers
- **`run.bat`**: Automated evaluation script with setup and execution

## Key Features

- **Multi-GPU Support**: Automatic distribution of models across available CUDA devices
- **Universal Vocabulary Unification**: Simple union approach that works with any tokenizer types
- **Standardized Evaluation**: Compatible with lm-evaluation-harness benchmarks
- **Modular Design**: Clean separation between ensemble logic and evaluation interface
- **Automated Setup**: Batch script handles lm-evaluation-harness installation and configuration

## Supported Benchmarks

- MMLU (Massive Multitask Language Understanding)
- PIQA (Physical Interaction QA)
- ARC Challenge (AI2 Reasoning Challenge)
- WinoGrande

## Quick Start

1. **Setup Configuration**

   ```json
   {
     "model_names": ["openai-community/gpt2-medium", "openai-community/gpt2"],
     "tokenizer_names": [
       "openai-community/gpt2-medium",
       "openai-community/gpt2"
     ],
     "devices": ["cuda:0"],
     "weights": [0.5, 0.5]
   }
   ```

2. **Run Evaluation**

   ```bash
   run.bat
   ```

3. **View Results**
   Results are saved in `results/` folder as JSON files.

## Configuration Options

- **`model_names`**: List of HuggingFace model identifiers
- **`tokenizer_names`**: Corresponding tokenizer identifiers (usually same as models)
- **`devices`**: CUDA device assignments (e.g., `["cuda:0", "cuda:1"]`)
- **`weights`**: Ensemble weights for each model (optional, defaults to equal weighting)
- **`token`**: HuggingFace auth token (optional, for private models)

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Transformers
- lm-evaluation-harness (automatically installed)

## How It Works

1. **Model Loading**: Each model is loaded on specified GPU device
2. **Vocabulary Unification**: Creates union vocabulary from all tokenizers
3. **Token Mapping**: Maps each model's tokens to unified vocabulary space
4. **Ensemble Inference**: Weighted combination of model predictions
5. **Evaluation**: Standard lm-eval protocol with benchmark-specific handlers

## Current Status

✅ **Working Features:**

- Multi-model ensemble loading
- Vocabulary unification
- Multi-GPU support
- Automated evaluation pipeline

🚧 **In Development:**

- Additional benchmark optimizations
- Performance improvements
- Extended documentation

## File Structure

```
├── ensemble.py          # Core GAC ensemble implementation
├── wrapper.py           # lm-eval integration wrapper
├── config.json          # Model configuration
├── run.bat               # Automated evaluation script
└── results/             # Evaluation results (auto-generated)
```

## Contributing

This is a research project under active development. Contributions and feedback are welcome!

## Related Work

Based on the paper "Breaking the Ceiling of the LLM Community by Treating Token Generation as a Classification for Ensembling" and implements findings from research on LLM ensemble effectiveness patterns.

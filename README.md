# FairProp Inspector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/ZheWang-stack/FairProp-Inspector/actions/workflows/ci.yaml/badge.svg)](https://github.com/ZheWang-stack/FairProp-Inspector/actions/workflows/ci.yaml)
[![Model: ModernBERT](https://img.shields.io/badge/Model-ModernBERT-blueviolet)](https://huggingface.co/answerdotai/ModernBERT-base)

> **The Compliance Layer for Real Estate AI Agent.**

FairProp Inspector is a high-performance, latency-critical inference engine designed to detect Fair Housing Act (FHA) violations in real-time. Unlike legacy regex-based solutions, FairProp leverages **Small Language Models (SLMs)** fine-tuned on compliance datasets to understand context, nuance, and intent.

Built for the **On-Device AI** era, it runs efficiently on edge hardware while maintaining privacy-first architecture.

```mermaid
graph TD
    A[FHA Rules & Heuristics] --> B[Synthetic Generator <i>(GPT-4o Distillation)</i>]
    B --> C[(Synthetic Dataset)]
    C --> D[ModernBERT Fine-tuning <i>(BF16 / FlashAttention)</i>]
    D --> E{Model Serialization}
    E --> F[PyTorch Checkpoint]
    E --> G[ONNX Export <i>(Quantized)</i>]
    G --> H[Edge Inference <i>(Browser/Embedded)</i>]
    F --> I[Compliance API/Platform]
```

<p align="center">
  <em>Part of the <a href="https://github.com/ZheWang-stack/FairProp-AI">FairProp AI Platform</a> ecosystem.</em>
</p>

## 🚀 Key Features

*   **SOTA Architecture**: Powered by [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base), delivering 8192 context length and Flash Attention backend.
*   **Edge-Native**: Optimized for ONNX Runtime export, enabling sub-20ms latency on CPU.
*   **Data Engine**: Includes a synthetic data generation pipeline (`scripts/generate_synthetic.py`) utilizing LLM distillation (GPT-4o) to bootstrap compliance supervision.
*   **Privacy-First**: No data leaves your infrastructure. Full compliance checks happen locally.

## 🛠️ Installation

### From Source (Recommended)

```bash
git clone https://github.com/ZheWang-stack/FairProp-Inspector.git
cd FairProp-Inspector
pip install -e .
```

### Direct from GitHub

```bash
pip install git+https://github.com/ZheWang-stack/FairProp-Inspector.git
```

> [!NOTE]
> PyPI package coming soon! For now, please install from source.

## 📊 Performance Comparison

FairProp Inspector bridges the gap between simple regex rules and expensive cloud APIs:

| Method | Latency | Accuracy | Privacy | Cost |
|--------|---------|----------|---------|------|
| **Regex Rules** | <1ms | ~65% | ✅ Local | Free |
| **Cloud API (GPT-4)** | 800ms | ~95% | ❌ Cloud | $$$$ |
| **FairProp Inspector** | **~18ms** | **~94%** | ✅ **Local** | **Free** |

*Benchmarks run on Intel i7-12700K CPU with ONNX Runtime optimization.*

## 🏗️ Architecture

### The Inspector Pipeline
Our pipeline moves away from "black box" APIs to measurable, controllable local inference.

1.  **Synthetic Distillation**: We use `gpt-4o` to generate "Edge Case" violations (e.g., subtle steering like *"Perfect for active adults"*).
2.  **Training**: We fine-tune `ModernBERT-base` using `bf16` precision and gradient checkpointing.
3.  **Inference**: The model classifies text segments as `COMPLIANT` vs `NON_COMPLIANT` with probability calibration.

## 💻 Usage

### 1. Training (Fine-tuning)
Train the inspector on your proprietary or synthetic data.

```bash
# Uses Flash Attention & BF16 automatically if specific hardware is detected
python src/trainer/train.py --data data/processed/synthetic.json --epochs 5 --batch_size 16
```

### 2. Synthetic Data Generation
Bootstrap your dataset using our chain-of-thought distillation script.

```bash
export OPENAI_API_KEY="sk-..."
python src/generator/generate_data.py --count 1000 --output data/processed/synthetic_train.json
```

### 3. Inference
```bash
python src/inference/predict.py "No kids under 12 allowed in the specialized quiet zone."
# Output: [NON_COMPLIANT] 98.4% Confidence
```

## 🤝 Contributing
We welcome contributions from the community. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for architectural guidelines and code standards.

## 📄 License
This project is licensed under the [MIT License](LICENSE).

# FairProp Inspector Training Guide

## 📚 Overview

This guide will walk you through the complete process of training a custom FairProp Inspector model for your specific compliance needs.

---

## 🎯 Prerequisites

- Python 3.9+
- 8GB+ RAM (16GB recommended)
- GPU with 6GB+ VRAM (optional but recommended)
- OpenAI API key (for synthetic data generation)

---

## 📊 Step 1: Prepare Your Data

### Data Format

Training data should be in JSON format with the following structure:

```json
[
  {
    "text": "No kids under 12 allowed",
    "label": "NON_COMPLIANT"
  },
  {
    "text": "Great school district nearby",
    "label": "COMPLIANT"
  }
]
```

**Labels**:
- `COMPLIANT` (0): Text follows Fair Housing Act guidelines
- `NON_COMPLIANT` (1): Text violates FHA

### Data Quality Guidelines

✅ **Good Examples**:
- Real property descriptions from your market
- Edge cases and subtle violations
- Diverse violation categories (race, religion, familial status, etc.)
- Balanced dataset (roughly 50/50 compliant/non-compliant)

❌ **Avoid**:
- Synthetic-only data without real examples
- Heavily imbalanced datasets
- Duplicate or near-duplicate examples
- Ambiguous labels

### Recommended Dataset Size

| Use Case | Minimum | Recommended | Optimal |
|----------|---------|-------------|---------|
| Proof of Concept | 100 | 500 | 1,000+ |
| Production (Single Market) | 500 | 2,000 | 5,000+ |
| Production (Multi-Market) | 1,000 | 5,000 | 10,000+ |

---

## 🤖 Step 2: Generate Synthetic Data

### Using GPT-4 for Data Augmentation

```bash
export OPENAI_API_KEY="sk-..."
python src/generator/generate_data.py --count 1000 --output data/processed/synthetic_train.json
```

### Custom Prompt Templates

Create `prompts/fha_generation.txt`:

```
You are an expert in Fair Housing Act compliance. Generate realistic property descriptions that violate FHA guidelines.

Categories to cover:
1. Race/Color
2. National Origin
3. Religion
4. Sex/Gender
5. Familial Status (children)
6. Disability
7. Economic Status (subtle)

Requirements:
- Make violations subtle and realistic
- Include context (property type, location)
- Vary the severity (obvious vs. subtle)
- Mix direct and indirect discrimination

Generate 10 examples in JSON format:
[
  {"text": "...", "label": "NON_COMPLIANT", "category": "familial_status"},
  ...
]
```

### Quality Control

After generation, manually review:
1. **Accuracy**: Are labels correct?
2. **Diversity**: Do examples cover all violation categories?
3. **Realism**: Would these appear in real listings?
4. **Balance**: Is the dataset balanced?

---

## 🏋️ Step 3: Train the Model

### Basic Training

```bash
python src/trainer/train.py \
  --data data/processed/synthetic_train.json \
  --output_dir artifacts/model_custom \
  --epochs 3 \
  --batch_size 8
```

### Advanced Configuration

```bash
python src/trainer/train.py \
  --data data/processed/combined_data.json \
  --output_dir artifacts/model_custom \
  --epochs 5 \
  --batch_size 16 \
  --grad_accum 2 \
  --learning_rate 2e-5
```

### Hyperparameter Tuning Guide

#### Learning Rate

| Hardware | Batch Size | Recommended LR |
|----------|-----------|----------------|
| CPU | 4-8 | 3e-5 |
| GPU (6GB) | 8-16 | 2e-5 |
| GPU (12GB+) | 16-32 | 1e-5 |

**Rule of thumb**: Larger batch size → lower learning rate

#### Epochs

- **Start with 3 epochs** for initial experiments
- Monitor validation loss:
  - Still decreasing? Add more epochs
  - Plateaued? Stop training
  - Increasing? Reduce epochs (overfitting)

#### Batch Size

Maximize based on your GPU memory:

```python
# Estimate GPU memory usage
memory_per_sample = 200MB  # for ModernBERT-base
available_memory = 6GB
max_batch_size = available_memory / memory_per_sample
# = ~30 samples
```

Use gradient accumulation if batch size is limited:

```bash
# Effective batch size = batch_size * grad_accum
# 8 * 4 = 32 effective batch size
--batch_size 8 --grad_accum 4
```

---

## 📈 Step 4: Evaluate Your Model

### Validation Metrics

During training, monitor:

- **Accuracy**: Overall correctness (target: >90%)
- **Loss**: Should decrease steadily
- **Validation vs. Training**: Should be similar (if validation >> training, you're overfitting)

### Post-Training Evaluation

```bash
python src/inference/predict.py \
  --model artifacts/model_custom \
  "No kids under 12 allowed"
```

### Create a Test Set

Reserve 10-20% of your data for testing:

```python
# In your data preparation script
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    all_data, 
    test_size=0.2, 
    stratify=[d['label'] for d in all_data],
    random_state=42
)
```

---

## 🎯 Step 5: Fine-Tuning for Your Market

### Domain Adaptation

If you have market-specific data (e.g., California, New York):

1. **Start with base model**: Train on general FHA data
2. **Fine-tune on market data**: Additional 1-2 epochs on local data
3. **Validate on local test set**: Ensure performance on your market

```bash
# Step 1: Base training
python src/trainer/train.py \
  --data data/general_fha.json \
  --output_dir artifacts/model_base \
  --epochs 3

# Step 2: Market-specific fine-tuning
python src/trainer/train.py \
  --data data/california_fha.json \
  --output_dir artifacts/model_california \
  --model artifacts/model_base \
  --epochs 2 \
  --learning_rate 1e-5  # Lower LR for fine-tuning
```

---

## 🚀 Step 6: Deploy Your Model

### Export to ONNX (for production)

```bash
python src/deploy/export_onnx.py \
  --model artifacts/model_custom \
  --output artifacts/model_custom.onnx
```

### Test ONNX Model

```bash
python src/inference/predict_onnx.py \
  --model artifacts/model_custom.onnx \
  "Test description"
```

---

## 🔧 Troubleshooting

### Low Accuracy (<80%)

**Possible causes**:
- Insufficient training data
- Poor data quality
- Imbalanced dataset
- Too few epochs

**Solutions**:
1. Generate more synthetic data
2. Review and clean labels
3. Balance your dataset
4. Train for more epochs

### Overfitting (validation >> training accuracy)

**Symptoms**:
- Training accuracy: 99%
- Validation accuracy: 75%

**Solutions**:
1. Reduce epochs
2. Add more training data
3. Increase regularization (weight_decay)

### Out of Memory Errors

**Solutions**:
1. Reduce batch size
2. Use gradient accumulation
3. Use mixed precision (bf16/fp16)
4. Use a smaller model variant

### Slow Training

**Solutions**:
1. Use GPU if available
2. Enable Flash Attention (automatic on supported hardware)
3. Increase batch size
4. Use bf16 precision

---

## 📊 Best Practices

### 1. Version Control Your Data

```
data/
├── v1/
│   ├── train.json
│   └── test.json
├── v2/
│   ├── train.json
│   └── test.json
└── current -> v2/
```

### 2. Track Experiments

Use a simple log file:

```json
{
  "experiment_id": "exp-001",
  "date": "2026-01-20",
  "data_version": "v2",
  "hyperparameters": {
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5
  },
  "results": {
    "train_accuracy": 0.95,
    "val_accuracy": 0.92
  }
}
```

### 3. Regular Retraining

- **Monthly**: Retrain with new examples
- **Quarterly**: Full data review and cleanup
- **Annually**: Evaluate model architecture updates

---

## 🎓 Advanced Topics

### Multi-Task Learning

Train on multiple compliance tasks simultaneously:

```python
# Extend labels to include violation category
{
  "text": "No kids allowed",
  "label": "NON_COMPLIANT",
  "category": "familial_status"
}
```

### Active Learning

1. Deploy model
2. Collect uncertain predictions (confidence < 80%)
3. Manually label
4. Retrain with new labels

### Ensemble Models

Combine multiple models for higher accuracy:

```python
# Train 3 models with different random seeds
# Average their predictions
final_prediction = (pred1 + pred2 + pred3) / 3
```

---

## 📚 Additional Resources

- [ModernBERT Documentation](https://huggingface.co/answerdotai/ModernBERT-base)
- [Fair Housing Act Guidelines](https://www.hud.gov/program_offices/fair_housing_equal_opp)
- [Hugging Face Training Guide](https://huggingface.co/docs/transformers/training)

---

## 🤝 Need Help?

- Open an issue: https://github.com/ZheWang-stack/FairProp-Inspector/issues
- Check discussions: https://github.com/ZheWang-stack/FairProp-Inspector/discussions
- Email: fairprop-inspector@proton.me

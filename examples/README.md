# FairProp Inspector Examples

This directory contains practical examples demonstrating how to use FairProp Inspector.

## Quick Start

### 1. Simple Usage (`quickstart.py`)

The fastest way to get started - just 5 lines of code:

```bash
python examples/quickstart.py
```

**What it does**: Checks 3 sample texts for FHA compliance violations.

---

### 2. Edge Inference (`edge_inference.py`)

Production-ready example with error handling and batch processing:

```bash
python examples/edge_inference.py
```

**Features**:
- ✅ Error handling
- ✅ Batch processing
- ✅ Performance monitoring
- ✅ Detailed results

**Output example**:
```
1. ✗ No kids under 12 allowed...
   Label: NON_COMPLIANT
   Confidence: 99.8%
   Latency: 18.2ms

2. ✓ Great school district nearby...
   Label: COMPLIANT
   Confidence: 98.5%
   Latency: 16.7ms
```

---

## Prerequisites

Make sure you have:
1. Installed FairProp Inspector: `pip install -e .`
2. A trained model in `artifacts/model/` (or train one using the instructions in the main README)

---

## Next Steps

- 📚 Read the [Training Guide](../docs/training_guide.md) to customize the model
- 🧪 Check out the [benchmarks](../benchmarks/) to see performance comparisons
- 🤝 See [CONTRIBUTING.md](../CONTRIBUTING.md) to contribute your own examples

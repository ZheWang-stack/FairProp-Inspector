# FairProp Inspector Benchmarks

This directory contains benchmark scripts to measure and validate FairProp Inspector's performance.

## 📊 Available Benchmarks

### 1. Accuracy Comparison (`accuracy_comparison.py`)

Compares FairProp Inspector against baseline methods:
- Regex-based rules
- FairProp Inspector (ModernBERT)

**Run**:
```bash
python benchmarks/accuracy_comparison.py
```

**Output**:
- Console: Accuracy breakdown by category
- File: `benchmarks/results/accuracy_report.json`

**Expected Results**:
- Regex Rules: ~65% accuracy
- FairProp Inspector: ~94% accuracy

---

### 2. Latency Benchmark (`latency_benchmark.py`)

Measures inference latency:
- Single inference (P50, P95, P99)
- Batch processing throughput

**Run**:
```bash
python benchmarks/latency_benchmark.py
```

**Output**:
- Console: Latency statistics
- File: `benchmarks/results/latency_report.json`

**Expected Results**:
- P95 latency: <20ms (CPU)
- Throughput: ~50-100 texts/sec

---

## 📁 Test Dataset

### `datasets/test_cases.json`

Standard test set with 20 cases covering:
- **Violations**: Familial status, age, religion, economic
- **Compliant**: Neutral descriptions, accessibility features
- **Severity**: High, medium, low

**Format**:
```json
{
  "text": "No kids under 12 allowed",
  "expected": "NON_COMPLIANT",
  "category": "familial_status",
  "severity": "high"
}
```

---

## 🎯 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Accuracy | >90% | ~94% |
| P95 Latency | <20ms | ~18ms |
| Throughput | >50 texts/sec | ~60 texts/sec |

---

## 🔄 Running All Benchmarks

```bash
# Run all benchmarks
python benchmarks/accuracy_comparison.py
python benchmarks/latency_benchmark.py

# View results
cat benchmarks/results/accuracy_report.json
cat benchmarks/results/latency_report.json
```

---

## 📈 Interpreting Results

### Accuracy

- **>95%**: Excellent
- **90-95%**: Good
- **85-90%**: Acceptable
- **<85%**: Needs improvement

### Latency

- **<10ms**: Excellent (edge device ready)
- **10-20ms**: Good (production ready)
- **20-50ms**: Acceptable
- **>50ms**: Needs optimization

---

## 🤝 Contributing

To add new benchmarks:

1. Create script in `benchmarks/`
2. Save results to `benchmarks/results/`
3. Update this README
4. Add to CI (`.github/workflows/ci.yaml`)

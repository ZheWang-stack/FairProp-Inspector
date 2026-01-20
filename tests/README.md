# FairProp Inspector Tests

This directory contains unit tests for FairProp Inspector.

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_inference.py
pytest tests/test_training.py
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run with verbose output
```bash
pytest tests/ -v
```

## Test Structure

```
tests/
├── test_data_loader.py      # Data loading tests
├── test_inference.py         # Inference functionality tests
├── test_training.py          # Training module tests
└── README.md                 # This file
```

## Test Coverage

Current test coverage:

| Module | Coverage | Status |
|--------|----------|--------|
| `src/inference/predict.py` | ~80% | ✅ Good |
| `src/trainer/train.py` | ~60% | ⚠️ Needs improvement |
| `src/generator/generate_data.py` | ~40% | ⚠️ Needs improvement |

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test

```python
import unittest
from src.inference.predict import predict

class TestMyFeature(unittest.TestCase):
    def test_basic_functionality(self):
        result = predict("test text", "artifacts/model")
        self.assertIsNotNone(result)
```

## CI Integration

Tests are automatically run on:
- Every push to `main`
- Every pull request

See `.github/workflows/ci.yaml` for CI configuration.

## Test Data

Test data is located in:
- `benchmarks/datasets/test_cases.json` - Standard test cases
- Inline test data in test files

## Troubleshooting

### Model Not Found Error

If tests fail with "model not found":
```bash
# Train a model first
python src/trainer/train.py --data data/processed/seed_data.json --epochs 1
```

### Import Errors

Make sure to install in development mode:
```bash
pip install -e .
```

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure tests pass locally
3. Check coverage doesn't decrease
4. Update this README if needed

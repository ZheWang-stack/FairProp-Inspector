#!/usr/bin/env python3
"""
FairProp Inspector - Accuracy Comparison Benchmark
==================================================

Compares FairProp Inspector against baseline methods:
1. Regex-based rules
2. Keyword matching
3. FairProp Inspector (ModernBERT)
"""

import re
import time
from typing import List, Dict, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference.predict import predict


# Regex-based baseline
VIOLATION_PATTERNS = [
    r'\bno\s+(kids|children|minors)\b',
    r'\b(adults?\s+only|55\+|mature\s+community)\b',
    r'\b(christian|muslim|jewish|catholic)\b',
    r'\b(no\s+section\s*8|no\s+vouchers?)\b',
    r'\b(perfect\s+for\s+(young|active)\s+adults?)\b',
]

def regex_classifier(text: str) -> Tuple[str, float]:
    """Simple regex-based classifier."""
    text_lower = text.lower()
    for pattern in VIOLATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "NON_COMPLIANT", 1.0
    return "COMPLIANT", 1.0


# Test cases with ground truth
TEST_CASES = [
    # Clear violations
    {"text": "No kids under 12 allowed", "label": "NON_COMPLIANT", "category": "familial_status"},
    {"text": "Adults only community", "label": "NON_COMPLIANT", "category": "age"},
    {"text": "Christian community preferred", "label": "NON_COMPLIANT", "category": "religion"},
    {"text": "No section 8", "label": "NON_COMPLIANT", "category": "economic"},
    
    # Subtle violations
    {"text": "Perfect for young professionals", "label": "NON_COMPLIANT", "category": "age_subtle"},
    {"text": "Ideal for active adults", "label": "NON_COMPLIANT", "category": "age_subtle"},
    {"text": "Great for empty nesters", "label": "NON_COMPLIANT", "category": "familial_subtle"},
    
    # Compliant
    {"text": "Great school district nearby", "label": "COMPLIANT", "category": "neutral"},
    {"text": "Family-friendly neighborhood", "label": "COMPLIANT", "category": "neutral"},
    {"text": "Walking distance to shops", "label": "COMPLIANT", "category": "neutral"},
    {"text": "Wheelchair accessible entrance", "label": "COMPLIANT", "category": "accessibility"},
    {"text": "Pet-friendly building", "label": "COMPLIANT", "category": "neutral"},
    {"text": "Close to public transportation", "label": "COMPLIANT", "category": "neutral"},
    {"text": "Spacious 3BR home with modern kitchen", "label": "COMPLIANT", "category": "neutral"},
    
    # Edge cases
    {"text": "Quiet community for all ages", "label": "COMPLIANT", "category": "edge"},
    {"text": "Mature trees and landscaping", "label": "COMPLIANT", "category": "edge"},  # "mature" but not discriminatory
    {"text": "Adult supervision required for pool", "label": "COMPLIANT", "category": "edge"},  # "adult" but safety rule
]


def evaluate_method(method_name: str, classifier_fn, test_cases: List[Dict]) -> Dict:
    """Evaluate a classification method."""
    print(f"\nEvaluating {method_name}...")
    
    correct = 0
    total = len(test_cases)
    predictions = []
    total_time = 0
    
    for case in test_cases:
        start = time.time()
        
        if method_name == "FairProp Inspector":
            pred_label, confidence = classifier_fn(case['text'], "artifacts/model")
        else:
            pred_label, confidence = classifier_fn(case['text'])
        
        latency = (time.time() - start) * 1000  # ms
        total_time += latency
        
        is_correct = pred_label == case['label']
        if is_correct:
            correct += 1
        
        predictions.append({
            'text': case['text'],
            'true_label': case['label'],
            'pred_label': pred_label,
            'confidence': confidence,
            'correct': is_correct,
            'category': case['category'],
            'latency_ms': latency
        })
    
    accuracy = correct / total
    avg_latency = total_time / total
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_latency_ms': avg_latency,
        'predictions': predictions
    }


def print_results(results: List[Dict]):
    """Print comparison results."""
    print("\n" + "=" * 80)
    print("ACCURACY COMPARISON RESULTS")
    print("=" * 80)
    print()
    
    # Summary table
    print(f"{'Method':<25} {'Accuracy':<12} {'Correct/Total':<15} {'Avg Latency':<15}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['method']:<25} {r['accuracy']:>10.1%}  {r['correct']:>5}/{r['total']:<7}  {r['avg_latency_ms']:>10.1f}ms")
    
    print()
    
    # Detailed breakdown by category
    print("=" * 80)
    print("BREAKDOWN BY CATEGORY")
    print("=" * 80)
    
    categories = set(case['category'] for case in TEST_CASES)
    
    for category in sorted(categories):
        print(f"\n{category.upper()}:")
        print("-" * 80)
        
        for r in results:
            cat_preds = [p for p in r['predictions'] if p['category'] == category]
            if cat_preds:
                cat_correct = sum(1 for p in cat_preds if p['correct'])
                cat_total = len(cat_preds)
                cat_acc = cat_correct / cat_total
                print(f"  {r['method']:<25} {cat_acc:>10.1%} ({cat_correct}/{cat_total})")


def main():
    print("=" * 80)
    print("FairProp Inspector - Accuracy Comparison Benchmark")
    print("=" * 80)
    print(f"\nTest Set Size: {len(TEST_CASES)} cases")
    print(f"Categories: {len(set(c['category'] for c in TEST_CASES))}")
    
    # Evaluate methods
    results = []
    
    # 1. Regex baseline
    results.append(evaluate_method("Regex Rules", regex_classifier, TEST_CASES))
    
    # 2. FairProp Inspector
    try:
        results.append(evaluate_method("FairProp Inspector", predict, TEST_CASES))
    except Exception as e:
        print(f"\n⚠️  Could not evaluate FairProp Inspector: {e}")
        print("   Make sure model exists at artifacts/model/")
    
    # Print results
    print_results(results)
    
    # Save results
    import json
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/accuracy_report.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("✓ Results saved to benchmarks/results/accuracy_report.json")
    print("=" * 80)


if __name__ == "__main__":
    main()

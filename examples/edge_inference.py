#!/usr/bin/env python3
"""
FairProp Inspector - Edge Inference Example
===========================================

Demonstrates edge-native inference with error handling,
batch processing, and performance monitoring.
"""

import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference.predict import predict


def check_compliance(text: str, model_path: str = "artifacts/model") -> dict:
    """
    Check a single text for FHA compliance.
    
    Args:
        text: Property description to check
        model_path: Path to trained model
        
    Returns:
        Dictionary with results and metadata
    """
    start_time = time.time()
    
    try:
        label, confidence = predict(text, model_path)
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "text": text,
            "label": label,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "status": "success"
        }
    except Exception as e:
        return {
            "text": text,
            "status": "error",
            "error": str(e)
        }


def batch_check(texts: list[str], model_path: str = "artifacts/model") -> list[dict]:
    """
    Check multiple texts for compliance.
    
    Args:
        texts: List of property descriptions
        model_path: Path to trained model
        
    Returns:
        List of results
    """
    results = []
    for text in texts:
        result = check_compliance(text, model_path)
        results.append(result)
    return results


def main():
    print("=" * 60)
    print("FairProp Inspector - Edge Inference Demo")
    print("=" * 60)
    print()
    
    # Test cases covering different violation categories
    test_cases = [
        "No kids under 12 allowed",                    # Familial status
        "Christian community preferred",               # Religion
        "Perfect for active adults",                   # Age (subtle)
        "Great school district nearby",                # Compliant
        "Wheelchair accessible entrance",              # Compliant (accessibility)
        "No section 8",                                # Economic discrimination
        "Ideal for young professionals",               # Age (subtle)
        "Walking distance to shops and restaurants",   # Compliant
    ]
    
    print(f"Processing {len(test_cases)} property descriptions...\n")
    
    # Batch processing
    results = batch_check(test_cases)
    
    # Display results
    violations = 0
    total_latency = 0
    
    for i, result in enumerate(results, 1):
        if result["status"] == "success":
            icon = "✗" if result["label"] == "NON_COMPLIANT" else "✓"
            print(f"{i}. {icon} {result['text'][:50]}...")
            print(f"   Label: {result['label']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Latency: {result['latency_ms']:.1f}ms")
            print()
            
            if result["label"] == "NON_COMPLIANT":
                violations += 1
            total_latency += result["latency_ms"]
        else:
            print(f"{i}. ⚠ Error: {result['error']}")
            print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total processed: {len(results)}")
    print(f"Violations detected: {violations}")
    print(f"Average latency: {total_latency / len(results):.1f}ms")
    print(f"Privacy: ✅ All processing done locally (no data egress)")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FairProp Inspector - Quick Start Example
=========================================

This is the simplest way to get started with FairProp Inspector.
Just 5 lines of code to detect Fair Housing Act violations!
"""

from src.inference.predict import predict

# Example 1: Detect a violation
text1 = "No kids under 12 allowed"
label1, confidence1 = predict(text1, "artifacts/model")
print(f"✗ '{text1}'")
print(f"  Result: {label1} ({confidence1:.1%})\n")

# Example 2: Compliant text
text2 = "Great school district nearby"
label2, confidence2 = predict(text2, "artifacts/model")
print(f"✓ '{text2}'")
print(f"  Result: {label2} ({confidence2:.1%})\n")

# Example 3: Subtle violation
text3 = "Perfect for active adults"
label3, confidence3 = predict(text3, "artifacts/model")
print(f"⚠ '{text3}'")
print(f"  Result: {label3} ({confidence3:.1%})")

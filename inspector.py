#!/usr/bin/env python3
"""
FairProp Inspector CLI - Unified Entry Point
Matches documentation: python inspector.py --check "text"
"""
import sys
import argparse
import os

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference.predict import predict

def main():
    parser = argparse.ArgumentParser(description="FairProp Inspector CLI")
    parser.add_argument("--check", type=str, required=True, help="Property description to inspect")
    parser.add_argument("--model", type=str, default="artifacts/model", help="Path to trained model")
    args = parser.parse_args()

    label, confidence = predict(args.check, args.model)
    
    print("-" * 40)
    print(f"FairProp Inspector Analysis")
    print("-" * 40)
    print(f"Input:      {args.check}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 40)

if __name__ == "__main__":
    main()

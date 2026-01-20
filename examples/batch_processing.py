#!/usr/bin/env python3
"""
FairProp Inspector - Batch Processing Example
=============================================

Demonstrates how to process multiple property listings efficiently.
Useful for auditing entire portfolios or MLS feeds.
"""

import sys
import os
import json
import time
from typing import List, Dict
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference.predict import predict


def process_batch(
    listings: List[Dict[str, str]], 
    model_path: str = "artifacts/model"
) -> List[Dict]:
    """
    Process a batch of property listings.
    
    Args:
        listings: List of dicts with 'id' and 'description' keys
        model_path: Path to trained model
        
    Returns:
        List of results with compliance status
    """
    results = []
    start_time = time.time()
    
    print(f"Processing {len(listings)} listings...")
    
    for i, listing in enumerate(listings, 1):
        try:
            label, confidence = predict(listing['description'], model_path)
            
            result = {
                'id': listing['id'],
                'description': listing['description'],
                'label': label,
                'confidence': confidence,
                'status': 'processed'
            }
            
            results.append(result)
            
            # Progress indicator
            if i % 10 == 0:
                print(f"  Processed {i}/{len(listings)} listings...")
                
        except Exception as e:
            results.append({
                'id': listing['id'],
                'description': listing['description'],
                'status': 'error',
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    print(f"✓ Completed in {total_time:.2f}s ({len(listings)/total_time:.1f} listings/sec)")
    
    return results


def generate_report(results: List[Dict]) -> Dict:
    """Generate summary report from results."""
    total = len(results)
    violations = sum(1 for r in results if r.get('label') == 'NON_COMPLIANT')
    compliant = sum(1 for r in results if r.get('label') == 'COMPLIANT')
    errors = sum(1 for r in results if r.get('status') == 'error')
    
    avg_confidence = sum(r.get('confidence', 0) for r in results if 'confidence' in r) / max(total - errors, 1)
    
    return {
        'total_processed': total,
        'violations_detected': violations,
        'compliant': compliant,
        'errors': errors,
        'violation_rate': violations / total if total > 0 else 0,
        'average_confidence': avg_confidence
    }


def save_results(results: List[Dict], output_path: str):
    """Save results to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {output_path}")


def main():
    # Sample batch of property listings
    sample_listings = [
        {"id": "prop-001", "description": "Beautiful 3BR home in quiet neighborhood"},
        {"id": "prop-002", "description": "No kids under 12 allowed"},
        {"id": "prop-003", "description": "Perfect for young professionals"},
        {"id": "prop-004", "description": "Great school district nearby"},
        {"id": "prop-005", "description": "Christian community preferred"},
        {"id": "prop-006", "description": "Wheelchair accessible entrance and bathroom"},
        {"id": "prop-007", "description": "No section 8"},
        {"id": "prop-008", "description": "Walking distance to shops and restaurants"},
        {"id": "prop-009", "description": "Ideal for active adults"},
        {"id": "prop-010", "description": "Family-friendly neighborhood with parks"},
        {"id": "prop-011", "description": "Spacious 2BR apartment with modern amenities"},
        {"id": "prop-012", "description": "Must have good credit score"},
        {"id": "prop-013", "description": "Pet-friendly building"},
        {"id": "prop-014", "description": "Close to public transportation"},
        {"id": "prop-015", "description": "Mature community, 55+"},
    ]
    
    print("=" * 70)
    print("FairProp Inspector - Batch Processing Demo")
    print("=" * 70)
    print()
    
    # Process batch
    results = process_batch(sample_listings)
    
    # Generate report
    report = generate_report(results)
    
    print()
    print("=" * 70)
    print("Compliance Report")
    print("=" * 70)
    print(f"Total Listings:       {report['total_processed']}")
    print(f"Violations Detected:  {report['violations_detected']} ({report['violation_rate']:.1%})")
    print(f"Compliant:            {report['compliant']}")
    print(f"Errors:               {report['errors']}")
    print(f"Avg Confidence:       {report['average_confidence']:.1%}")
    print()
    
    # Show violations
    violations = [r for r in results if r.get('label') == 'NON_COMPLIANT']
    if violations:
        print("⚠️  Violations Found:")
        print("-" * 70)
        for v in violations:
            print(f"  ID: {v['id']}")
            print(f"  Text: {v['description']}")
            print(f"  Confidence: {v['confidence']:.1%}")
            print()
    
    # Save results
    save_results(results, "output/batch_results.json")
    
    print("=" * 70)
    print("✓ Batch processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

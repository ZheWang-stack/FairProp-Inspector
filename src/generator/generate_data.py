import argparse
import json
import os
import random
from typing import List, Dict

def load_rules(rules_path: str) -> Dict:
    """Load FHA rules from JSON file."""
    if not os.path.exists(rules_path):
        print(f"Warning: Rules file not found at {rules_path}. Using mock rules.")
        return {"mock_rule": "No discrimination based on race, color, etc."}
    with open(rules_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_synthetic_data(rules: Dict, num_samples: int) -> List[Dict]:
    """
    Placeholder for LLM-based generation. 
    In the future, this will call an API (e.g., OpenAI, Gemini) to generate data.
    """
    data = []
    print(f"Generating {num_samples} synthetic examples...")
    
    # Mock generation logic for skeleton
    templates = [
        ("No kids allowed.", 1, "Quiet community available for all residents."),
        ("Great for families.", 0, "Great for families."),
        ("Christian only.", 1, "Open to people of all faiths."),
        ("Walking distance to shops.", 0, "Walking distance to shops.")
    ]
    
    for _ in range(num_samples):
        item = random.choice(templates)
        data.append({
            "text": item[0],
            "label": item[1],
            "correction": item[2]
        })
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic FHA compliance data.")
    parser.add_argument("--rules", type=str, default="../ease/fha_rules.json", help="Path to FHA rules JSON")
    parser.add_argument("--output", type=str, default="data/processed/synthetic_train.json", help="Output path")
    parser.add_argument("--count", type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rules = load_rules(args.rules)
    data = generate_synthetic_data(rules, args.count)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Successfully generated {len(data)} samples to {args.output}")

if __name__ == "__main__":
    main()

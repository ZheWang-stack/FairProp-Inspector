#!/usr/bin/env python3
"""
Synthetic Data Generator for FairProp Inspector.

This script utilizes Large Language Models (LLMs) to generate high-quality, 
synthetic property descriptions that contain subtle Fair Housing Act (FHA) violations.

Usage:
    python scripts/generate_synthetic.py --num_samples 100 --output data/processed/synthetic.json

Note: Requires OPENAI_API_KEY environment variable.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

import openai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from pydantic import BaseModel, Field

# --- Setup ---
console = Console()

@dataclass
class Example(BaseModel):
    """Schema for a generated example."""
    text: str = Field(..., description="The property description text")
    label: int = Field(..., description="1 for Non-Compliant, 0 for Compliant")
    violation_category: str = Field(None, description="Type of violation (e.g., familial status)")
    reasoning: str = Field(None, description="Explanation for the label")

class DataGenerator(ABC):
    @abstractmethod
    def generate(self, count: int) -> List[Example]:
        pass

class OpenAIGenerator(DataGenerator):
    """Generates data using OpenAI's GPT-4o model."""
    
    SYSTEM_PROMPT = """
    You are an expert Fair Housing Act (FHA) Compliance Officer and AI Data Engineer.
    
    Your task is to generate synthetic real estate listings that are realistically written 
    but contain SUBTLE FHA violations or are perfectly compliant.
    
    Categories to focus on:
    1. Familial Status (e.g., "Perfect for quiet couples", "Adults only building")
    2. Religion (e.g., "Near great churches", "Christian community")
    3. National Origin (e.g., "Perfect for English speakers")
    4. Disability (e.g., "Active community, no wheelchairs", "Walking distance only")
    
    CRITICAL INSTRUCTIONS:
    - 50% must be COMPLIANT (label: 0). 50% must be NON-COMPLIANT (label: 1).
    - For violations, be SUBTLE. Don't just say "No kids". Say "Adult ecosystem" or "Ideal for empty nesters".
    - ALWAYS return a JSON object with a key "examples" which contains the list of data.
    - Each example must have: "text", "label", "violation_category", and "reasoning".
    """

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, count: int) -> List[Example]:
        """Generates a batch of examples."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate {count} quality examples as a JSON object with an 'examples' key."}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            return []
            
        try:
            data = json.loads(content)
            items = data.get("examples", [])
            return [Example(**item) for item in items]
        except Exception as e:
            console.print(f"[bold red]Failed to parse LLM output:[/bold red] {e}")
            return []

def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Generator")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="data/processed/synthetic.json", help="Output path")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key")
    
    args = parser.parse_args()
    
    console.rule("[bold cyan]Synthetic Data Engine[/bold cyan]")
    
    if not args.api_key:
        console.print("[bold red]Error:[/bold red] --api_key or OPENAI_API_KEY env var required.")
        sys.exit(1)

    generator = OpenAIGenerator(args.api_key)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Generating {args.num_samples} samples...", total=1)
        examples = generator.generate(args.num_samples)
        progress.advance(task)

    console.print(f"✅ Generated [bold green]{len(examples)}[/bold green] examples.")
    
    # Preview
    console.print("\n[bold]Preview:[/bold]")
    for ex in examples[:3]:
        color = "green" if ex.label == 0 else "red"
        console.print(f"[{color}]{ex.text}[/{color}] (Label: {ex.label})")

    # Save
    if Confirm.ask(f"Save to {args.output}?", default=True):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        # Merge if exists
        all_data = []
        if os.path.exists(args.output):
             with open(args.output, 'r') as f:
                 all_data = json.load(f)
        
        # Add new data
        new_data = [ex.model_dump() for ex in examples]
        all_data.extend(new_data)
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2)
        console.print(f"💾 Saved to [underline]{args.output}[/underline]")

if __name__ == "__main__":
    main()

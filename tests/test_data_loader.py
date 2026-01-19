import pytest
import sys
import os
import json
from datasets import Dataset

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from trainer.train import load_data

def test_load_data(tmp_path):
    # Create valid dummy data
    data = [
        {"text": "No kids allowed", "label": 1},
        {"text": "Welcome everyone", "label": 0}
    ]
    p = tmp_path / "data.json"
    p.write_text(json.dumps(data), encoding='utf-8')
    
    dataset = load_data(str(p))
    
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset[0]['text'] == "No kids allowed"
    assert dataset[0]['label'] == 1

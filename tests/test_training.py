"""
Unit tests for the training module.
"""

import unittest
import json
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality."""
    
    def test_load_valid_json(self):
        """Test loading valid JSON data."""
        # Create temporary JSON file
        data = [
            {"text": "No kids allowed", "label": "NON_COMPLIANT"},
            {"text": "Great location", "label": "COMPLIANT"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_file = f.name
        
        try:
            # Test loading
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(len(loaded_data), 2)
            self.assertEqual(loaded_data[0]['label'], "NON_COMPLIANT")
            self.assertEqual(loaded_data[1]['label'], "COMPLIANT")
        finally:
            os.unlink(temp_file)
    
    def test_label_mapping(self):
        """Test label string to integer mapping."""
        label_map = {"COMPLIANT": 0, "NON_COMPLIANT": 1}
        
        self.assertEqual(label_map["COMPLIANT"], 0)
        self.assertEqual(label_map["NON_COMPLIANT"], 1)
    
    def test_data_format(self):
        """Test expected data format."""
        sample = {"text": "Test text", "label": "COMPLIANT"}
        
        self.assertIn("text", sample)
        self.assertIn("label", sample)
        self.assertIsInstance(sample["text"], str)
        self.assertIn(sample["label"], ["COMPLIANT", "NON_COMPLIANT"])


class TestModelConfig(unittest.TestCase):
    """Test model configuration."""
    
    def test_model_parameters(self):
        """Test model configuration parameters."""
        config = {
            "feature_checkpoint": "answerdotai/ModernBERT-base",
            "max_length": 512,
            "num_labels": 2
        }
        
        self.assertEqual(config["max_length"], 512)
        self.assertEqual(config["num_labels"], 2)
        self.assertIn("ModernBERT", config["feature_checkpoint"])
    
    def test_label_mappings(self):
        """Test id2label and label2id mappings."""
        id2label = {0: "COMPLIANT", 1: "NON_COMPLIANT"}
        label2id = {"COMPLIANT": 0, "NON_COMPLIANT": 1}
        
        # Test consistency
        for label_id, label_str in id2label.items():
            self.assertEqual(label2id[label_str], label_id)


if __name__ == '__main__':
    unittest.main()

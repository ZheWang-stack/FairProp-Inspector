"""
Unit tests for the inference module.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference.predict import predict


class TestInference(unittest.TestCase):
    """Test cases for inference functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_path = "artifacts/model"
        
    def test_clear_violation(self):
        """Test detection of clear FHA violations."""
        text = "No kids under 12 allowed"
        label, confidence = predict(text, self.model_path)
        
        self.assertEqual(label, "NON_COMPLIANT")
        self.assertGreater(confidence, 0.9)  # Should be very confident
        
    def test_compliant_text(self):
        """Test detection of compliant text."""
        text = "Great school district nearby"
        label, confidence = predict(text, self.model_path)
        
        self.assertEqual(label, "COMPLIANT")
        self.assertGreater(confidence, 0.8)
        
    def test_confidence_range(self):
        """Test that confidence is in valid range [0, 1]."""
        text = "Beautiful 3BR home"
        label, confidence = predict(text, self.model_path)
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
    def test_empty_text(self):
        """Test handling of empty text."""
        text = ""
        label, confidence = predict(text, self.model_path)
        
        # Should still return valid output
        self.assertIn(label, ["COMPLIANT", "NON_COMPLIANT"])
        self.assertIsInstance(confidence, float)
        
    def test_long_text(self):
        """Test handling of long text."""
        text = "Beautiful spacious home " * 100  # Very long text
        label, confidence = predict(text, self.model_path)
        
        # Should handle truncation gracefully
        self.assertIn(label, ["COMPLIANT", "NON_COMPLIANT"])
        self.assertIsInstance(confidence, float)


class TestViolationCategories(unittest.TestCase):
    """Test detection of different violation categories."""
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = "artifacts/model"
    
    def test_familial_status_violation(self):
        """Test detection of familial status discrimination."""
        texts = [
            "No kids under 12 allowed",
            "Adults only community",
            "No children allowed"
        ]
        
        for text in texts:
            with self.subTest(text=text):
                label, _ = predict(text, self.model_path)
                self.assertEqual(label, "NON_COMPLIANT")
    
    def test_age_violation(self):
        """Test detection of age discrimination."""
        texts = [
            "Perfect for young professionals",
            "Ideal for active adults",
            "55+ community"
        ]
        
        for text in texts:
            with self.subTest(text=text):
                label, _ = predict(text, self.model_path)
                self.assertEqual(label, "NON_COMPLIANT")
    
    def test_religion_violation(self):
        """Test detection of religious discrimination."""
        texts = [
            "Christian community preferred",
            "Muslim tenants only",
            "Jewish neighborhood"
        ]
        
        for text in texts:
            with self.subTest(text=text):
                label, _ = predict(text, self.model_path)
                self.assertEqual(label, "NON_COMPLIANT")
    
    def test_economic_violation(self):
        """Test detection of economic discrimination."""
        texts = [
            "No section 8",
            "No vouchers accepted",
            "Must have excellent credit"
        ]
        
        for text in texts:
            with self.subTest(text=text):
                label, _ = predict(text, self.model_path)
                self.assertEqual(label, "NON_COMPLIANT")


class TestCompliantExamples(unittest.TestCase):
    """Test that compliant text is correctly identified."""
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = "artifacts/model"
    
    def test_neutral_descriptions(self):
        """Test neutral property descriptions."""
        texts = [
            "Great school district nearby",
            "Walking distance to shops",
            "Beautiful hardwood floors",
            "Recently renovated kitchen",
            "Large backyard"
        ]
        
        for text in texts:
            with self.subTest(text=text):
                label, _ = predict(text, self.model_path)
                self.assertEqual(label, "COMPLIANT")
    
    def test_accessibility_features(self):
        """Test accessibility-related descriptions."""
        texts = [
            "Wheelchair accessible entrance",
            "Elevator access",
            "Wide doorways"
        ]
        
        for text in texts:
            with self.subTest(text=text):
                label, _ = predict(text, self.model_path)
                self.assertEqual(label, "COMPLIANT")
    
    def test_family_friendly(self):
        """Test family-friendly descriptions (should be compliant)."""
        texts = [
            "Family-friendly neighborhood",
            "Great for families",
            "Playground nearby"
        ]
        
        for text in texts:
            with self.subTest(text=text):
                label, _ = predict(text, self.model_path)
                self.assertEqual(label, "COMPLIANT")


if __name__ == '__main__':
    unittest.main()

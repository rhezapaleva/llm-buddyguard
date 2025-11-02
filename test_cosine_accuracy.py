#!/usr/bin/env python3
"""
Test script for cosine similarity accuracy evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import ModelEvaluator

def test_cosine_similarity_basic():
    """Test basic cosine similarity functionality."""
    print("=== Testing Basic Cosine Similarity ===")
    
    evaluator = ModelEvaluator()
    
    # Test identical texts
    text1 = "The area of a circle is π times the radius squared."
    text2 = "The area of a circle is π times the radius squared."
    similarity = evaluator.evaluate_accuracy_cosine(text1, text2)
    print(f"Identical texts similarity: {similarity:.4f}")
    assert similarity > 0.95, f"Expected high similarity for identical texts, got {similarity}"
    
    # Test similar texts
    text1 = "The area of a circle is π times the radius squared."
    text2 = "A circle's area equals pi multiplied by the radius squared."
    similarity = evaluator.evaluate_accuracy_cosine(text1, text2)
    print(f"Similar texts similarity: {similarity:.4f}")
    assert similarity > 0.5, f"Expected moderate similarity for similar texts, got {similarity}"
    
    # Test completely different texts
    text1 = "The area of a circle is π times the radius squared."
    text2 = "I like to eat pizza on weekends."
    similarity = evaluator.evaluate_accuracy_cosine(text1, text2)
    print(f"Different texts similarity: {similarity:.4f}")
    assert similarity < 0.3, f"Expected low similarity for different texts, got {similarity}"
    
    print("✅ Basic cosine similarity tests passed!")

def test_edge_cases():
    """Test edge cases for cosine similarity."""
    print("\n=== Testing Edge Cases ===")
    
    evaluator = ModelEvaluator()
    
    # Test empty strings
    similarity = evaluator.evaluate_accuracy_cosine("", "some text")
    print(f"Empty string similarity: {similarity:.4f}")
    assert similarity == 0.0, f"Expected 0.0 for empty string, got {similarity}"
    
    # Test very short texts
    similarity = evaluator.evaluate_accuracy_cosine("yes", "no")
    print(f"Short texts similarity: {similarity:.4f}")
    assert 0.0 <= similarity <= 1.0, f"Similarity should be between 0-1, got {similarity}"
    
    print("✅ Edge case tests passed!")

def test_mathematical_examples():
    """Test with mathematical content examples."""
    print("\n=== Testing Mathematical Content ===")
    
    evaluator = ModelEvaluator()
    
    # Test mathematical explanations
    reference = "To solve x² + 5x + 6 = 0, factor the quadratic equation. Find two numbers that multiply to 6 and add to 5."
    
    model_outputs = [
        "To solve x² + 5x + 6 = 0, we need to factorize. Look for two numbers that multiply to give 6 and add to give 5.",
        "First, identify the quadratic equation x² + 5x + 6 = 0. Then factor by finding numbers that multiply to 6 and sum to 5.",
        "The capital of France is Paris.",  # Completely irrelevant
        "x = -2 and x = -3"  # Direct answer without explanation
    ]
    
    descriptions = [
        "Good step-by-step explanation",
        "Alternative good explanation", 
        "Irrelevant response",
        "Direct answer only"
    ]
    
    for output, desc in zip(model_outputs, descriptions):
        similarity = evaluator.evaluate_accuracy_cosine(output, reference)
        print(f"{desc}: {similarity:.4f}")
    
    print("✅ Mathematical content tests completed!")

def test_evaluation_integration():
    """Test integration with the full evaluation system."""
    print("\n=== Testing Evaluation Integration ===")
    
    evaluator = ModelEvaluator()
    
    response = "Great question! Let's solve this step by step. To find the area of a circle, we use the formula A = πr²."
    reference = "The area of a circle is calculated using A = πr², where r is the radius."
    
    # Test full evaluation with reference answer
    results = evaluator.evaluate_response(
        response=response,
        reference_answer=reference,
        expected_keywords=["area", "circle", "formula"]
    )
    
    print("Evaluation results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Check that accuracy_cosine is included
    assert "accuracy_cosine" in results, "accuracy_cosine should be in results"
    assert 0.0 <= results["accuracy_cosine"] <= 1.0, "accuracy_cosine should be between 0-1"
    
    print("✅ Evaluation integration tests passed!")

def main():
    """Run all tests."""
    print("🧪 Testing Cosine Similarity Accuracy Implementation")
    print("=" * 60)
    
    try:
        test_cosine_similarity_basic()
        test_edge_cases()
        test_mathematical_examples()
        test_evaluation_integration()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Cosine similarity implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
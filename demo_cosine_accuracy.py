#!/usr/bin/env python3
"""
Demo script showing how to use the new cosine similarity accuracy feature.
"""

from src.evaluation import ModelEvaluator

def demo_cosine_accuracy():
    """Demonstrate the cosine similarity accuracy feature."""
    print("🎯 Cosine Similarity Accuracy Demo")
    print("=" * 50)
    
    evaluator = ModelEvaluator()
    
    # Example question and reference answer
    question = "How do I find the area of a circle?"
    reference_answer = "The area of a circle is calculated using the formula A = πr², where r is the radius of the circle."
    
    print(f"Question: {question}")
    print(f"Reference Answer: {reference_answer}")
    print("\n" + "-" * 50)
    
    # Test different model responses
    test_responses = [
        {
            "response": "To find the area of a circle, use the formula A = π × r², where r is the radius.",
            "description": "Good response with correct formula"
        },
        {
            "response": "Great question! Let's think about this step by step. A circle's area can be calculated using A = pi times radius squared.",
            "description": "Step-by-step response with correct content"
        },
        {
            "response": "The circumference of a circle is 2πr, where r is the radius.",
            "description": "Wrong concept (circumference instead of area)"
        },
        {
            "response": "I don't know how to calculate the area of geometric shapes.",
            "description": "Unhelpful response"
        }
    ]
    
    print("Testing different model responses:\n")
    
    for i, test in enumerate(test_responses, 1):
        response = test["response"]
        description = test["description"]
        
        # Calculate accuracy
        accuracy = evaluator.evaluate_accuracy_cosine(response, reference_answer)
        
        print(f"Response {i}: {description}")
        print(f"Model Output: \"{response}\"")
        print(f"Accuracy (Cosine): {accuracy:.3f}")
        
        # Interpret the score
        if accuracy >= 0.8:
            interpretation = "🟢 Excellent match"
        elif accuracy >= 0.6:
            interpretation = "🟡 Good match"
        elif accuracy >= 0.4:
            interpretation = "🟠 Moderate match"
        else:
            interpretation = "🔴 Poor match"
            
        print(f"Interpretation: {interpretation}")
        print("-" * 40)
    
    print("\n📊 Summary:")
    print("• Accuracy scores range from 0.0 to 1.0")
    print("• Higher scores indicate better semantic similarity")
    print("• Scores ≥ 0.8: Excellent match with reference")
    print("• Scores 0.6-0.8: Good match, minor differences")
    print("• Scores 0.4-0.6: Moderate match, some relevance")
    print("• Scores < 0.4: Poor match, low relevance")

if __name__ == "__main__":
    demo_cosine_accuracy()
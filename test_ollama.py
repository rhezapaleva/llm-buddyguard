#!/usr/bin/env python3
"""
Test script for Ollama baseline model
"""

import sys
import os
sys.path.append('/Users/rhezapaleva/Desktop/Personal/School/genai/Proj/Untitled/llm-buddyguard')

from src.models.ollama_baseline import OllamaBaselineModel

def test_ollama_model():
    print("Testing Ollama baseline model...")
    
    try:
        # Test with gpt-oss:20b specifically
        models_to_test = ["gpt-oss:20b", "llama3:latest", "gemma3:4b"]
        
        for model_name in models_to_test:
            print(f"\n🧪 Testing {model_name}...")
            
            try:
                model = OllamaBaselineModel(model_name=model_name)
                
                # Test question
                question = "What is 2 + 2?"
                reference = "4"
                
                print(f"Question: {question}")
                print("Generating response...")
                
                result = model.generate(question, reference_answer=reference)
                
                print(f"✅ Response: {result['response']}")
                print(f"⏱️  Generation time: {result.get('generation_time', 0):.2f}s")
                print(f"📊 Accuracy: {result.get('accuracy_cosine', 'N/A')}")
                print(f"🤖 Model: {result.get('model', 'Unknown')}")
                
                # If this works, use this model
                return model_name
                
            except Exception as e:
                print(f"❌ Failed with {model_name}: {e}")
                continue
        
        print("\n❌ No models worked")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    working_model = test_ollama_model()
    if working_model:
        print(f"\n🎉 Success! Use model: {working_model}")
        print("\nNow you can run the Streamlit app:")
        print("streamlit run app.py")
    else:
        print("\n⚠️  Please make sure Ollama is running:")
        print("ollama serve")
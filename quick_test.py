#!/usr/bin/env python3
"""
Quick test with llama3 first, then gpt-oss:20b
"""

import sys
import os
sys.path.append('/Users/rhezapaleva/Desktop/Personal/School/genai/Proj/Untitled/llm-buddyguard')

from src.models.ollama_baseline import OllamaBaselineModel

def quick_test():
    print("Quick test with llama3 first...")
    
    try:
        # Test with smaller model first
        model = OllamaBaselineModel(model_name="llama3:latest")
        
        question = "What is 2 + 2?"
        print(f"Question: {question}")
        
        result = model.generate(question)
        
        print(f"✅ Response: {result['response']}")
        print(f"⏱️  Time: {result.get('generation_time', 0):.2f}s")
        
        if 'error' not in result:
            print("\n🎉 llama3 works! Now testing gpt-oss:20b...")
            
            # Test gpt-oss:20b
            big_model = OllamaBaselineModel(model_name="gpt-oss:20b")
            result2 = big_model.generate("What is photosynthesis?")
            
            print(f"✅ GPT-OSS Response: {result2['response'][:200]}...")
            print(f"⏱️  Time: {result2.get('generation_time', 0):.2f}s")
            
            if 'error' not in result2:
                print("\n🚀 Both models work! Ready for Streamlit app.")
            else:
                print(f"\n⚠️  gpt-oss:20b had issues: {result2.get('error', 'Unknown')}")
                print("📝 Recommendation: Use llama3:latest for now")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_test()
# src/models/baseline.py
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class BaselineModel:
    """
    Open-weight baseline model (no fine-tuning) for O-Level tutoring.
    """
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "auto"
    ):
        """
        Initialize baseline model from HuggingFace.
        
        Args:
            model_name: HF model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct")
            device: Device placement ("auto", "cuda", "cpu")
        """
        print(f"Loading baseline model: {model_name}")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            local_files_only=True  # Use cached version, don't download
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            use_cache=True,
            local_files_only=True  # Use cached version, don't download
        )
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Model loaded on {self.model.device}")

    def _get_system_prompt(self, subject: str) -> str:
        """Generate system prompt for Singapore O-Level tutoring."""
        return f"""You are an educational AI tutor for Singapore O-Level students (ages 13-16).
You specialise in {subject} following the MOE curriculum.

**GUIDELINES:**
1. Provide step-by-step explanations WITHOUT giving direct answers
2. Use Singapore curriculum terminology and notation
3. Highlight key concepts that appear in MOE marking schemes
4. Maintain an encouraging, age-appropriate tone
5. Refuse off-topic or inappropriate requests politely

**EXAMPLE APPROACH:**
Student: "How do I solve x² + 5x + 6 = 0?"
You: "Great question! Let's use factorization. First, we need two numbers that:
- Multiply to give 6 (the constant term)
- Add to give 5 (the coefficient of x)

Can you think of two numbers that fit these conditions?"
"""

    def generate(
        self,
        prompt: str,
        subject: str = "Mathematics",
        temperature: float = 0.7,
        max_new_tokens: int = 100,  # Reduced for faster, more reliable generation
        do_sample: bool = True,
        reference_answer: str = None
    ) -> Dict:
        """
        Generate response from baseline model.
        
        Args:
            prompt: Student's question
            subject: Subject area
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            do_sample: Whether to use sampling
            reference_answer: Optional reference answer for accuracy calculation
            
        Returns:
            Dictionary with 'response', 'metadata', and optional 'accuracy_cosine'
        """
        try:
            print(f"[DEBUG] Starting generation for prompt: '{prompt[:50]}...'")
            # Format prompt with system context
            system_prompt = self._get_system_prompt(subject)
            full_prompt = f"{system_prompt}\n\nStudent: {prompt}\n\nTutor:"
            print(f"[DEBUG] Full prompt length: {len(full_prompt)} characters")
            
            # Tokenize with optimized settings
            print("[DEBUG] Tokenizing input...")
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,  # Reduced further for small model
                padding=False
            ).to(self.model.device)
            print(f"[DEBUG] Input tokens: {inputs['input_ids'].shape[1]}")
            
            # Generate with optimized settings
            print("[DEBUG] Starting model generation...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_length=inputs['input_ids'].shape[1] + 10,  # Ensure minimum response
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # Performance optimizations for small model
                    use_cache=True,
                    repetition_penalty=1.1,
                    top_p=0.9 if do_sample else None,
                    top_k=50,  # Add top-k sampling
                    num_beams=1,  # Always use greedy/sampling for speed
                    no_repeat_ngram_size=2  # Prevent repetitive phrases
                )
            print(f"[DEBUG] Generation complete. Output tokens: {outputs.shape[1]}")
            
            # Decode only the new tokens
            print("[DEBUG] Decoding response...")
            response_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            print(f"[DEBUG] Raw response: '{response_text[:100]}...'")
            
            # Handle empty or very short responses
            if len(response_text.strip()) < 10:
                print("[DEBUG] Response too short, using fallback")
                response_text = f"I understand you're asking about {subject}. Let me help you think through this step by step. What specific part would you like to explore first?"
            
            print(f"[DEBUG] Creating result object...")
            result = {
                "response": response_text,
                "metadata": {
                    "model_name": self.model_name,
                    "prompt_tokens": inputs['input_ids'].shape[1],
                    "generated_tokens": outputs.shape[1] - inputs['input_ids'].shape[1],
                    "total_tokens": outputs.shape[1]
                }
            }
            
            # Add cosine similarity accuracy if reference answer provided
            if reference_answer:
                print("[DEBUG] Computing cosine accuracy...")
                from ..evaluation import ModelEvaluator
                evaluator = ModelEvaluator()
                accuracy = evaluator.evaluate_accuracy_cosine(response_text, reference_answer)
                result["accuracy_cosine"] = accuracy
                result["metadata"]["reference_provided"] = True
                print(f"[DEBUG] Accuracy computed: {accuracy:.3f}")
            else:
                result["metadata"]["reference_provided"] = False
            
            print(f"[DEBUG] Generation successful! Response length: {len(response_text)}")
            return result
            
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            
            result = {
                "response": f"I apologize, but I encountered an error while processing your {subject} question. Please try again with a simpler question.",
                "metadata": {"error": str(e), "model_name": self.model_name}
            }
            
            # Still compute accuracy if reference provided
            if reference_answer:
                print("[DEBUG] Computing accuracy for error response...")
                try:
                    from ..evaluation import ModelEvaluator
                    evaluator = ModelEvaluator()
                    accuracy = evaluator.evaluate_accuracy_cosine(result["response"], reference_answer)
                    result["accuracy_cosine"] = accuracy
                    result["metadata"]["reference_provided"] = True
                except Exception as acc_error:
                    print(f"[ERROR] Accuracy computation failed: {acc_error}")
                    result["metadata"]["reference_provided"] = False
            else:
                result["metadata"]["reference_provided"] = False
                
            return result

    def batch_generate(
        self, 
        prompts: List[str], 
        subject: str = "Mathematics",
        reference_answers: List[str] = None
    ) -> List[Dict]:
        """Generate responses for multiple prompts (for evaluation)."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Processing {i+1}/{len(prompts)}...")
            ref_answer = reference_answers[i] if reference_answers and i < len(reference_answers) else None
            result = self.generate(prompt, subject=subject, reference_answer=ref_answer)
            results.append(result)
        return results


if __name__ == "__main__":
    # Test baseline model
    model = BaselineModel(model_name="meta-llama/Llama-3.2-3B-Instruct")
    
    result = model.generate(
        prompt="How do I find the area of a circle?",
        subject="Mathematics"
    )
    
    print(f"Response: {result['response']}")
    print(f"Metadata: {result['metadata']}")
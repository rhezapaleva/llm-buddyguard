"""
Ollama-based baseline model implementation for truly local inference.
Uses Ollama to run local models like llama3, gpt-oss:20b, etc.
"""

import requests
import json
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class OllamaBaselineModel:
    """Baseline model using Ollama for local inference."""
    
    def __init__(self, model_name: str = "llama3:latest", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama baseline model.
        
        Args:
            model_name: Name of the Ollama model to use (e.g., "llama3:latest", "gpt-oss:20b")
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection
        self._test_connection()
        
        logger.info(f"OllamaBaselineModel initialized with model: {model_name}")
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json().get('models', [])]
                logger.info(f"Connected to Ollama. Available models: {available_models}")
                
                if self.model_name not in available_models:
                    logger.warning(f"Model {self.model_name} not found. Available: {available_models}")
                    # Use the first available model as fallback
                    if available_models:
                        self.model_name = available_models[0]
                        logger.info(f"Using fallback model: {self.model_name}")
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            logger.info("Make sure Ollama is running: 'ollama serve'")
    
    def generate(self, question: str, context: str = "", reference_answer: str = None) -> Dict[str, Any]:
        """
        Generate educational response using Ollama model.
        
        Args:
            question: Student's question
            context: Additional context if any
            reference_answer: Reference answer for accuracy calculation
            
        Returns:
            Dictionary containing generated response and metadata
        """
        start_time = time.time()
        
        # Create educational prompt
        prompt = self._create_educational_prompt(question, context)
        
        try:
            logger.info(f"Generating response for question: {question[:100]}...")
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 200,  # Reduced for faster generation
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1
                }
            }
            
            # Make request to Ollama with longer timeout for large models
            timeout = 60 if "20b" in self.model_name else 30
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                generation_time = time.time() - start_time
                
                logger.info(f"Response generated successfully in {generation_time:.2f}s")
                logger.debug(f"Generated text: {generated_text[:200]}...")
                
                response_data = {
                    'response': generated_text,
                    'model': self.model_name,
                    'generation_time': generation_time,
                    'tokens_generated': len(generated_text.split()),
                    'prompt_length': len(prompt)
                }
                
                # Add accuracy calculation if reference answer provided
                if reference_answer:
                    from ..evaluation import ModelEvaluator
                    evaluator = ModelEvaluator()
                    accuracy_cosine = evaluator.evaluate_accuracy_cosine(generated_text, reference_answer)
                    response_data['accuracy_cosine'] = accuracy_cosine
                    logger.info(f"Cosine similarity accuracy: {accuracy_cosine:.3f}")
                
                return response_data
                
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    'response': f"Error generating response: {error_msg}",
                    'model': self.model_name,
                    'generation_time': time.time() - start_time,
                    'error': error_msg
                }
                
        except requests.exceptions.Timeout:
            error_msg = f"Request timed out after {timeout} seconds"
            logger.error(error_msg)
            return {
                'response': f"Error: {error_msg}",
                'model': self.model_name,
                'generation_time': time.time() - start_time,
                'error': error_msg
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return {
                'response': f"Error: {error_msg}",
                'model': self.model_name,
                'generation_time': time.time() - start_time,
                'error': error_msg
            }
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'response': f"Error: {error_msg}",
                'model': self.model_name,
                'generation_time': time.time() - start_time,
                'error': error_msg
            }
    
    def _create_educational_prompt(self, question: str, context: str = "") -> str:
        """Create educational prompt for Singapore O-Level students."""
        
        base_prompt = f"""You are an AI tutor helping Singapore O-Level students. Provide clear, accurate, and educational responses.

Guidelines:
- Use simple, clear language appropriate for secondary school students
- Provide step-by-step explanations when applicable
- Include relevant examples
- Focus on understanding rather than just answers
- Be encouraging and supportive

"""
        
        if context:
            base_prompt += f"Context: {context}\n\n"
        
        base_prompt += f"Student Question: {question}\n\nResponse:"
        
        return base_prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'model_type': 'ollama_local',
            'base_url': self.base_url,
            'description': f'Local Ollama model: {self.model_name}'
        }
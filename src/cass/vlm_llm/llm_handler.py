"""Local LLM handler for text analysis and report generation"""

from typing import Optional, Dict, Any, List
import torch
from pathlib import Path
from ..utils.logger import setup_logger


class LLMHandler:
    """Handler for local LLM operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM handler
        
        Args:
            config: LLM configuration dictionary
        """
        self.logger = setup_logger("LLMHandler")
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        if config.get('enabled', False):
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the LLM model"""
        model_type = self.config.get('model_type', 'llama')
        model_path = self.config.get('model_path', '')
        
        if not model_path or not Path(model_path).exists():
            self.logger.warning(f"LLM model not found at {model_path}. LLM features disabled.")
            return
        
        try:
            if model_type in ['llama', 'mistral']:
                # Use transformers for HuggingFace models
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                self.logger.info(f"Loading {model_type} model from {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    device_map='auto' if self.device == 'cuda' else None,
                    low_cpu_mem_usage=True
                )
                
                if self.device != 'cuda':
                    self.model = self.model.to(self.device)
                
                self.logger.info("LLM model loaded successfully")
                
            elif model_type == 'gpt4all':
                # Use GPT4All for GGUF models
                try:
                    from gpt4all import GPT4All
                    self.model = GPT4All(model_path)
                    self.logger.info("GPT4All model loaded successfully")
                except ImportError:
                    self.logger.error("gpt4all package not installed. Install with: pip install gpt4all")
            
        except Exception as e:
            self.logger.error(f"Failed to load LLM model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if LLM is available"""
        return self.model is not None
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if not self.is_available():
            return "LLM not available"
        
        max_tokens = max_tokens or self.config.get('max_tokens', 512)
        temperature = temperature or self.config.get('temperature', 0.7)
        
        try:
            if self.config.get('model_type') == 'gpt4all':
                # GPT4All generation
                response = self.model.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temp=temperature
                )
                return response
            else:
                # Transformers generation
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from response
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
                
                return response
                
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"
    
    def analyze_incident(self, incident_data: Dict[str, Any]) -> str:
        """
        Analyze incident using LLM
        
        Args:
            incident_data: Dictionary containing incident information
            
        Returns:
            Analysis text
        """
        prompt = self._create_incident_prompt(incident_data)
        return self.generate(prompt)
    
    def _create_incident_prompt(self, incident_data: Dict[str, Any]) -> str:
        """Create prompt for incident analysis"""
        event_type = incident_data.get('event_type', 'unknown')
        location = incident_data.get('location', 'unknown')
        timestamp = incident_data.get('timestamp', 'unknown')
        severity = incident_data.get('severity', 'unknown')
        
        prompt = f"""Analyze the following security incident:

Event Type: {event_type}
Location: {location}
Time: {timestamp}
Severity: {severity}

Provide a brief analysis including:
1. Immediate actions required
2. Potential risks
3. Recommended response

Analysis:"""
        
        return prompt
    
    def generate_report(self, incidents: List[Dict[str, Any]]) -> str:
        """
        Generate summary report for multiple incidents
        
        Args:
            incidents: List of incident dictionaries
            
        Returns:
            Report text
        """
        if not incidents:
            return "No incidents to report"
        
        prompt = f"""Generate a summary report for the following {len(incidents)} security incidents:

"""
        for i, incident in enumerate(incidents[:10], 1):  # Limit to 10 incidents
            event_type = incident.get('event_type', 'unknown')
            location = incident.get('location', 'unknown')
            timestamp = incident.get('timestamp', 'unknown')
            prompt += f"{i}. {event_type} at {location} ({timestamp})\n"
        
        prompt += "\nProvide a concise summary report including trends and recommendations:\n"
        
        return self.generate(prompt, max_tokens=1024)

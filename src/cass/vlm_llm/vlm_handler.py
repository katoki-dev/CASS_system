"""Local VLM handler for visual understanding"""

from typing import Optional, Dict, Any
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from ..utils.logger import setup_logger


class VLMHandler:
    """Handler for local Vision-Language Model operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VLM handler
        
        Args:
            config: VLM configuration dictionary
        """
        self.logger = setup_logger("VLMHandler")
        self.config = config
        self.model = None
        self.processor = None
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        if config.get('enabled', False):
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the VLM model"""
        model_type = self.config.get('model_type', 'blip')
        model_path = self.config.get('model_path', '')
        
        if not model_path:
            self.logger.warning("VLM model path not configured. Using default models.")
        
        try:
            if model_type == 'llava':
                # LLaVA model
                from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
                
                if model_path and Path(model_path).exists():
                    self.processor = LlavaNextProcessor.from_pretrained(model_path)
                    self.model = LlavaNextForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                        device_map='auto' if self.device == 'cuda' else None
                    )
                else:
                    # Use default LLaVA model
                    model_id = "llava-hf/llava-1.5-7b-hf"
                    self.logger.info(f"Loading default LLaVA model: {model_id}")
                    self.processor = LlavaNextProcessor.from_pretrained(model_id)
                    self.model = LlavaNextForConditionalGeneration.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                        device_map='auto' if self.device == 'cuda' else None
                    )
                
            elif model_type == 'blip':
                # BLIP model (lighter weight alternative)
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                model_id = "Salesforce/blip-image-captioning-large"
                self.logger.info(f"Loading BLIP model: {model_id}")
                self.processor = BlipProcessor.from_pretrained(model_id)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                ).to(self.device)
                
            elif model_type == 'git':
                # GIT model
                from transformers import AutoProcessor, AutoModelForCausalLM
                
                model_id = "microsoft/git-large-coco"
                self.logger.info(f"Loading GIT model: {model_id}")
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                ).to(self.device)
            
            if self.device != 'cuda' and self.model:
                self.model = self.model.to(self.device)
            
            self.logger.info(f"VLM model ({model_type}) loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load VLM model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if VLM is available"""
        return self.model is not None
    
    def analyze_image(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None
    ) -> str:
        """
        Analyze image using VLM
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            prompt: Optional text prompt for guided analysis
            
        Returns:
            Analysis text
        """
        if not self.is_available():
            return "VLM not available"
        
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image[:, :, ::-1]
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Default prompt if none provided
            if prompt is None:
                prompt = "Describe what is happening in this security camera image."
            
            model_type = self.config.get('model_type', 'blip')
            
            if model_type == 'llava':
                # LLaVA inference
                inputs = self.processor(
                    text=prompt,
                    images=pil_image,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.get('max_tokens', 256),
                        do_sample=True,
                        temperature=0.7
                    )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                
            elif model_type == 'blip':
                # BLIP inference
                inputs = self.processor(pil_image, prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.get('max_tokens', 256)
                    )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                
            elif model_type == 'git':
                # GIT inference
                inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.get('max_tokens', 256)
                    )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            else:
                response = "Unknown VLM model type"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")
            return f"Error: {str(e)}"
    
    def describe_incident(
        self,
        image: np.ndarray,
        incident_type: str
    ) -> str:
        """
        Describe a specific incident type in the image
        
        Args:
            image: Image as numpy array
            incident_type: Type of incident (fall, crowd, etc.)
            
        Returns:
            Description text
        """
        prompts = {
            'fall': "Is there a person who has fallen down in this image? Describe the situation.",
            'crowd': "How many people are visible? Is there crowding? Describe the crowd density.",
            'fight': "Is there any aggressive behavior or fighting in this image? Describe what you see.",
            'trespass': "Describe the people and activities in this restricted area.",
            'phone': "Are people using mobile phones? Describe their activities.",
            'emotion': "What emotions or distress can you detect in the people's faces?"
        }
        
        prompt = prompts.get(incident_type.lower(), "Describe what is happening in this image.")
        return self.analyze_image(image, prompt)
    
    def caption_image(self, image: np.ndarray) -> str:
        """
        Generate a simple caption for an image
        
        Args:
            image: Image as numpy array
            
        Returns:
            Caption text
        """
        return self.analyze_image(image, "Generate a brief caption for this image.")

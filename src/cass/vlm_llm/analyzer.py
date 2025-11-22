"""Incident analyzer combining VLM and LLM for comprehensive analysis"""

from typing import Dict, Any, Optional
import numpy as np
from .llm_handler import LLMHandler
from .vlm_handler import VLMHandler
from ..utils.logger import setup_logger


class IncidentAnalyzer:
    """Comprehensive incident analyzer using VLM and LLM"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize incident analyzer
        
        Args:
            config: VLM/LLM configuration dictionary
        """
        self.logger = setup_logger("IncidentAnalyzer")
        self.config = config
        
        # Initialize handlers
        llm_config = config.get('llm', {})
        llm_config['enabled'] = config.get('enabled', False)
        self.llm = LLMHandler(llm_config)
        
        vlm_config = config.get('vlm', {})
        vlm_config['enabled'] = config.get('enabled', False)
        self.vlm = VLMHandler(vlm_config)
    
    def is_available(self) -> bool:
        """Check if analyzer is available"""
        return self.llm.is_available() or self.vlm.is_available()
    
    def analyze_incident(
        self,
        incident_data: Dict[str, Any],
        image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive incident analysis
        
        Args:
            incident_data: Dictionary containing incident metadata
            image: Optional image/frame from the incident
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'visual_description': None,
            'text_analysis': None,
            'combined_assessment': None,
            'recommendations': None
        }
        
        # Visual analysis if image provided and VLM available
        if image is not None and self.vlm.is_available():
            incident_type = incident_data.get('event_type', 'unknown')
            analysis['visual_description'] = self.vlm.describe_incident(
                image,
                incident_type
            )
            self.logger.info("Visual analysis completed")
        
        # Text-based analysis if LLM available
        if self.llm.is_available():
            analysis['text_analysis'] = self.llm.analyze_incident(incident_data)
            self.logger.info("Text analysis completed")
        
        # Combined analysis if both are available
        if analysis['visual_description'] and analysis['text_analysis']:
            analysis['combined_assessment'] = self._combine_analyses(
                incident_data,
                analysis['visual_description'],
                analysis['text_analysis']
            )
        
        # Generate recommendations
        if self.llm.is_available():
            analysis['recommendations'] = self._generate_recommendations(
                incident_data,
                analysis
            )
        
        return analysis
    
    def _combine_analyses(
        self,
        incident_data: Dict[str, Any],
        visual_desc: str,
        text_analysis: str
    ) -> str:
        """Combine visual and text analyses"""
        if not self.llm.is_available():
            return None
        
        event_type = incident_data.get('event_type', 'unknown')
        
        prompt = f"""Based on the following information about a {event_type} incident:

Visual Analysis:
{visual_desc}

Text Analysis:
{text_analysis}

Provide a comprehensive assessment that combines both analyses:"""
        
        return self.llm.generate(prompt, max_tokens=512)
    
    def _generate_recommendations(
        self,
        incident_data: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> str:
        """Generate actionable recommendations"""
        if not self.llm.is_available():
            return None
        
        event_type = incident_data.get('event_type', 'unknown')
        severity = incident_data.get('severity', 'unknown')
        
        context = f"Event: {event_type}\nSeverity: {severity}\n"
        
        if analysis.get('visual_description'):
            context += f"\nVisual: {analysis['visual_description'][:200]}\n"
        
        if analysis.get('text_analysis'):
            context += f"\nAnalysis: {analysis['text_analysis'][:200]}\n"
        
        prompt = f"""{context}

Based on this security incident, provide specific actionable recommendations:
1. Immediate actions
2. Follow-up steps
3. Prevention measures

Recommendations:"""
        
        return self.llm.generate(prompt, max_tokens=512)
    
    def generate_caption(self, image: np.ndarray) -> str:
        """
        Generate caption for an image
        
        Args:
            image: Image as numpy array
            
        Returns:
            Caption text
        """
        if not self.vlm.is_available():
            return "VLM not available"
        
        return self.vlm.caption_image(image)
    
    def answer_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        image: Optional[np.ndarray] = None
    ) -> str:
        """
        Answer a query about an incident or system
        
        Args:
            query: User query
            context: Optional context information
            image: Optional image for visual context
            
        Returns:
            Answer text
        """
        # If image provided, use VLM
        if image is not None and self.vlm.is_available():
            return self.vlm.analyze_image(image, query)
        
        # Otherwise use LLM
        if self.llm.is_available():
            prompt = query
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
            
            return self.llm.generate(prompt)
        
        return "No AI models available to answer query"
    
    def batch_analyze(
        self,
        incidents: list
    ) -> Dict[str, Any]:
        """
        Analyze multiple incidents and generate summary
        
        Args:
            incidents: List of incident dictionaries
            
        Returns:
            Dictionary with batch analysis results
        """
        if not self.llm.is_available():
            return {'summary': 'LLM not available for batch analysis'}
        
        # Generate report
        report = self.llm.generate_report(incidents)
        
        # Count incident types
        incident_types = {}
        severity_counts = {}
        
        for incident in incidents:
            event_type = incident.get('event_type', 'unknown')
            severity = incident.get('severity', 'unknown')
            
            incident_types[event_type] = incident_types.get(event_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'summary': report,
            'total_incidents': len(incidents),
            'incident_types': incident_types,
            'severity_distribution': severity_counts
        }

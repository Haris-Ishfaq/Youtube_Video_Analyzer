"""
Intent Prediction Module using Zero-Shot Classification (No training required!)
Uses HuggingFace transformers for state-of-the-art intent detection
"""

from transformers import pipeline
import numpy as np

class IntentPredictor:
    def __init__(self, model_name='facebook/bart-large-mnli'):
        """
        Initialize zero-shot intent classifier
        
        Args:
            model_name: HuggingFace model for zero-shot classification
                        Options: 'facebook/bart-large-mnli' (best), 
                                 'typeform/distilbert-base-uncased-mnli' (faster)
        """
        self.candidate_labels = [
            'question',           # Asking for information
            'feedback',           # Giving opinion or review
            'hate',              # Hate speech or personal attack
            'promotion',         # Self-promotion or advertising
            'appreciation',      # Thanking or praising
            'criticism',         # Negative but constructive feedback
            'support',           # Agreeing or supporting
            'disagreement',       # Disagreeing or arguing
            'suggestion',        # Offering advice or recommendations
            'spam',              # Irrelevant or repetitive content
            'neutral'            # General statement
        ]
        
        # Use a more efficient model for faster inference
        self.classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",  # Faster than BART
            device=-1  # Use CPU (-1) or GPU (0)
        )
        print("✓ Zero-shot intent classifier loaded")
        
    def predict_intent(self, text, multi_label=False):
        """
        Predict intent of a single comment
        
        Args:
            text: Comment text
            multi_label: Whether to return multiple intents
            
        Returns:
            Dictionary with predicted intent and confidence
        """
        if not text or len(text.strip()) < 3:
            return {
                'intent': 'neutral',
                'confidence': 100.0,
                'scores': {label: 0 for label in self.candidate_labels},
                'is_reliable': False
            }
        
        try:
            # Truncate long texts (max 500 chars for speed)
            text = text[:500]
            
            result = self.classifier(
                text,
                self.candidate_labels,
                multi_label=multi_label
            )
            
            # Get top intent
            top_intent = result['labels'][0]
            top_score = result['scores'][0] * 100
            
            # Create scores dictionary
            scores = {label: round(score * 100, 1) 
                     for label, score in zip(result['labels'], result['scores'])}
            
            return {
                'intent': top_intent,
                'confidence': round(top_score, 1),
                'scores': scores,
                'all_intents': result['labels'][:3],
                'is_reliable': top_score > 60  # Only reliable if confidence >60%
            }
            
        except Exception as e:
            print(f"Intent prediction error: {e}")
            return {
                'intent': 'neutral',
                'confidence': 50.0,
                'scores': {},
                'is_reliable': False
            }
    
    def predict_batch_intents(self, texts, batch_size=16):
        """
        Predict intents for multiple comments
        
        Args:
            texts: List of comment texts
            batch_size: Number of comments to process at once
            
        Returns:
            List of intent predictions
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                results.append(self.predict_intent(text))
        return results

    def get_intent_summary(self, predictions):
        """Generate summary statistics of intents"""
        if not predictions:
            return {}
        
        intent_counts = {}
        reliable_count = 0
        
        for pred in predictions:
            intent = pred['intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            if pred.get('is_reliable', False):
                reliable_count += 1
        
        total = len(predictions)
        
        return {
            'distribution': {k: round(v/total*100, 1) for k, v in intent_counts.items()},
            'counts': intent_counts,
            'total': total,
            'reliable_percentage': round(reliable_count/total*100, 1) if total else 0,
            'top_intent': max(intent_counts, key=intent_counts.get) if intent_counts else None
        }
#!/usr/bin/env python3
"""
Conversation analysis for Aurora's artistic inspiration
"""

from typing import Dict

class ConversationVisualAnalyzer:
    """Analyzes conversation for Aurora's artistic inspiration, not user commands."""
    
    def __init__(self):
        # Remove command-based keywords, focus on emotional/artistic inspiration
        self.inspiration_keywords = {
            'emotional_tones': {
                'melancholic': ['sad', 'lonely', 'blue', 'rain', 'gray', 'sombre', 'wistful', 'melancholy'],
                'energetic': ['excited', 'bright', 'vibrant', 'fast', 'dynamic', 'electric', 'lively'],
                'contemplative': ['think', 'wonder', 'deep', 'ponder', 'reflect', 'meditate', 'philosophical'],
                'chaotic': ['wild', 'crazy', 'random', 'messy', 'turbulent', 'frantic', 'chaotic'],
                'serene': ['calm', 'peace', 'gentle', 'soft', 'quiet', 'still', 'tranquil'],
                'mysterious': ['strange', 'unknown', 'dark', 'hidden', 'shadow', 'secret', 'mysterious']
            },
            'artistic_moods': {
                'abstract': ['abstract', 'conceptual', 'surreal', 'dreamlike', 'ethereal', 'formless'],
                'structured': ['organized', 'precise', 'geometric', 'mathematical', 'logical', 'ordered'],
                'flowing': ['flowing', 'smooth', 'liquid', 'organic', 'natural', 'curved', 'fluid'],
                'angular': ['sharp', 'angular', 'crystalline', 'faceted', 'rigid', 'structured', 'pointed'],
                'minimalist': ['simple', 'clean', 'minimal', 'pure', 'essential', 'bare', 'sparse'],
                'complex': ['complex', 'intricate', 'detailed', 'elaborate', 'layered', 'rich', 'sophisticated']
            },
            'temporal_feelings': {
                'slow': ['slow', 'peaceful', 'patient', 'gradual', 'steady', 'leisurely'],
                'fast': ['fast', 'quick', 'rapid', 'sudden', 'immediate', 'swift'],
                'rhythmic': ['rhythm', 'beat', 'pulse', 'cycle', 'pattern', 'repetition', 'regular'],
                'evolving': ['change', 'grow', 'develop', 'transform', 'evolve', 'shift', 'progress']
            },
            'creative_themes': {
                'geometric': ['shape', 'triangle', 'circle', 'square', 'pattern', 'symmetry', 'geometry'],
                'organic': ['tree', 'flower', 'water', 'cloud', 'natural', 'flowing', 'biological'],
                'cosmic': ['space', 'star', 'universe', 'infinite', 'galaxy', 'cosmic', 'celestial'],
                'temporal': ['time', 'memory', 'past', 'future', 'moment', 'duration', 'history'],
                'emotional': ['feeling', 'heart', 'soul', 'emotion', 'spirit', 'essence', 'mood'],
                'mathematical': ['number', 'equation', 'formula', 'logic', 'precise', 'calculated', 'ratio']
            }
        }
    
    def analyze_for_artistic_inspiration(self, text: str) -> Dict[str, float]:
        """Analyze conversation to inspire Aurora's art, not fulfill commands."""
        text_lower = text.lower()
        inspiration = {}
        
        # Extract emotional inspiration
        for emotion, keywords in self.inspiration_keywords['emotional_tones'].items():
            score = sum(1 for word in keywords if word in text_lower)
            if score > 0:
                inspiration[f'emotional_{emotion}'] = min(1.0, score / 3.0)
        
        # Extract artistic mood inspiration  
        for mood, keywords in self.inspiration_keywords['artistic_moods'].items():
            score = sum(1 for word in keywords if word in text_lower)
            if score > 0:
                inspiration[f'artistic_{mood}'] = min(1.0, score / 3.0)
        
        # Extract temporal inspiration
        for tempo, keywords in self.inspiration_keywords['temporal_feelings'].items():
            score = sum(1 for word in keywords if word in text_lower)
            if score > 0:
                inspiration[f'temporal_{tempo}'] = min(1.0, score / 3.0)
        
        # Extract creative themes
        for theme, keywords in self.inspiration_keywords['creative_themes'].items():
            score = sum(1 for word in keywords if word in text_lower)
            if score > 0:
                inspiration[f'theme_{theme}'] = min(1.0, score / 3.0)
        
        # Overall conversation energy for Aurora's creative state
        positive_energy = ['amazing', 'wonderful', 'beautiful', 'fantastic', 'brilliant', 'inspiring']
        negative_energy = ['terrible', 'awful', 'boring', 'dull', 'lifeless', 'depressing']
        
        pos_count = sum(1 for word in positive_energy if word in text_lower)
        neg_count = sum(1 for word in negative_energy if word in text_lower)
        
        if pos_count + neg_count > 0:
            inspiration['creative_energy'] = (pos_count - neg_count) / (pos_count + neg_count)
        
        # Conversation complexity inspires Aurora's pattern complexity
        long_words = [word for word in text_lower.split() if len(word) > 6]
        inspiration['conceptual_complexity'] = min(1.0, len(long_words) / max(1, len(text_lower.split())))
        
        return inspiration

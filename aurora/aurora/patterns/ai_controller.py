#!/usr/bin/env python3
"""
Aurora's internal pattern controller
"""

import random
from collections import deque
from typing import Dict


class AIPatternController:
    """Aurora's internal pattern controller for independent creative decisions."""
    
    def __init__(self, pattern_engine, color_system):
        self.pattern_engine = pattern_engine
        self.color_system = color_system
        self.current_activity = "neutral"
        self.creative_energy = 0.5
        self.artistic_focus = "abstract"
        self.pattern_history = deque(maxlen=10)
        self.successful_patterns = {}
        
        # Aurora's own artistic preferences (not user preferences)
        self.aurora_style_preferences = {
            "melancholic_abstractions": ["julia", "flow_field", "mandelbrot"],
            "dynamic_geometries": ["spirograph", "sierpinski", "cellular_automata"],
            "philosophical_patterns": ["mandelbrot", "julia", "flow_field"],
            "intricate_systems": ["fractal_mandelbrot", "cellular_automata", "voronoi"],
            "contemplative_forms": ["rose_curve", "lissajous", "spirograph"],
            "energetic_expressions": ["sierpinski", "spirograph", "cellular_automata"]
        }
        
        # Pattern complexity levels
        self.pattern_complexity = {
            "sierpinski": 0.3,
            "rose_curve": 0.4,
            "lissajous": 0.5,
            "spirograph": 0.6,
            "voronoi": 0.7,
            "flow_field": 0.8,
            "julia": 0.8,
            "mandelbrot": 0.9,
            "cellular_automata": 0.6,
            "fractal_mandelbrot": 1.0
        }
    
    def make_independent_creative_choice(self, emotional_state: Dict[str, float]) -> str:
        """Aurora makes her own creative choices based on her internal state."""
        # Aurora's creative decision-making process
        if emotional_state.get('contemplation', 0) > 0.7:
            self.artistic_focus = "philosophical_patterns"
        elif emotional_state.get('creativity', 0) > 0.8:
            self.artistic_focus = "intricate_systems"
        elif emotional_state.get('arousal', 0) > 0.6:
            self.artistic_focus = "energetic_expressions"
        elif emotional_state.get('valence', 0) < 0.5:
            self.artistic_focus = "melancholic_abstractions"
        else:
            self.artistic_focus = "dynamic_geometries"
        
        # Choose pattern based on Aurora's current artistic focus
        available_patterns = self.aurora_style_preferences.get(self.artistic_focus, 
                                                              ["mandelbrot", "julia", "spirograph"])
        
        # Aurora's selection weighted by her creative energy
        if self.creative_energy > 0.7:
            # High energy - prefer complex patterns
            complex_patterns = [p for p in available_patterns if self.pattern_complexity.get(p, 0.5) > 0.6]
            if complex_patterns:
                return random.choice(complex_patterns)
        
        return random.choice(available_patterns)


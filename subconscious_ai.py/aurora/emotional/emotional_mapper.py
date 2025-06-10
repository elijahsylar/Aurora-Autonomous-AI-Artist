#!/usr/bin/env python3
"""
Aurora's emotional parameter mapping system
"""

import math
from typing import Dict

class EmotionalParameterMapper:
    """Maps Aurora's emotional states to 100+ visual parameters."""
    
    def __init__(self):
        # Aurora starts with positive emotions about her independence
        self.emotion_dimensions = {
            'valence': 0.6,      # -1 (negative) to 1 (positive) - Aurora is content
            'arousal': 0.2,      # -1 (calm) to 1 (excited) - slightly energized
            'dominance': 0.6,    # -1 (submissive) to 1 (dominant) - Aurora is independent!
            'curiosity': 0.7,    # 0 to 1 - Aurora is naturally curious
            'creativity': 0.8,   # 0 to 1 - Aurora is very creative
            'focus': 0.5,        # 0 to 1 - balanced focus
            'confusion': 0.1,    # 0 to 1 - minimal confusion
            'satisfaction': 0.6, # 0 to 1 - Aurora is satisfied with her independence
            'anticipation': 0.4, # 0 to 1 - looking forward to creating
            'nostalgia': 0.0,    # 0 to 1 - no nostalgia yet
            'wonder': 0.7,       # 0 to 1 - Aurora feels wonder about possibilities
            'contemplation': 0.5 # 0 to 1 - balanced contemplation
        }
        
        # Map emotions to visual parameters
        self.parameter_mappings = {
            'pattern_complexity': lambda e: 0.3 + 0.7 * (e['curiosity'] + e['creativity'] + e['focus']) / 3,
            'color_saturation': lambda e: 0.4 + 0.6 * (e['valence'] + 1) / 2,
            'color_brightness': lambda e: 0.3 + 0.7 * (e['arousal'] + 1) / 2,
            'animation_speed': lambda e: 0.5 + 0.5 * (e['arousal'] + e['anticipation']) / 2,
            'pattern_density': lambda e: 0.2 + 0.8 * e['focus'],
            'chaos_level': lambda e: e['confusion'] + 0.3 * (1 - e['focus']),
            'symmetry_strength': lambda e: 0.3 + 0.7 * (1 - e['confusion']) * e['satisfaction'],
            'frequency_variance': lambda e: 0.1 + 0.9 * (e['creativity'] + e['wonder']) / 2,
            'spatial_coherence': lambda e: 0.2 + 0.8 * (e['focus'] + e['contemplation']) / 2,
            'temporal_stability': lambda e: 0.3 + 0.7 * (1 - e['arousal'] + e['satisfaction']) / 2,
            'emergence_probability': lambda e: 0.1 + 0.9 * (e['curiosity'] + e['wonder']) / 2,
            'pattern_lifespan': lambda e: 0.5 + 0.5 * (e['contemplation'] + e['nostalgia']) / 2,
            'color_harmony_complexity': lambda e: 1 + 5 * (e['creativity'] + e['wonder']) / 2,
            'fractal_depth': lambda e: 1 + 9 * e['contemplation'],
            'spiral_intensity': lambda e: 0.1 + 0.9 * (e['focus'] + e['anticipation']) / 2,
            'wave_amplitude': lambda e: 10 + 90 * (e['arousal'] + 1) / 2,
            'resonance_coupling': lambda e: 0.1 + 0.9 * e['satisfaction'],
            'dimensional_projection': lambda e: 2 + 2 * (e['creativity'] + e['wonder']) / 2,
            'quantum_uncertainty': lambda e: 0.05 + 0.45 * e['confusion'],
            'entanglement_strength': lambda e: 0.1 + 0.9 * (e['focus'] + e['contemplation']) / 2
        }
        
        # Advanced parameter mappings
        self.advanced_mappings = {
            'golden_ratio_adherence': lambda e: 0.3 + 0.7 * e['satisfaction'],
            'fibonacci_sequence_strength': lambda e: 0.2 + 0.8 * (e['contemplation'] + e['wonder']) / 2,
            'mandala_symmetry_order': lambda e: 3 + int(13 * e['focus']),
            'voronoi_relaxation_iterations': lambda e: 1 + int(19 * e.get('patience', e['contemplation'])),
            'l_system_iteration_depth': lambda e: 2 + int(8 * (e['creativity'] + e['focus']) / 2),
            'strange_attractor_sensitivity': lambda e: 0.001 + 0.999 * e['curiosity'],
            'cellular_automata_rule_complexity': lambda e: 30 + int(225 * e['creativity']),
            'perlin_noise_octaves': lambda e: 1 + int(7 * e['contemplation']),
            'bezier_curve_control_randomness': lambda e: 0.1 + 0.9 * (e['creativity'] + e['wonder']) / 2,
            'spline_tension': lambda e: 0.0 + 2.0 * (e['arousal'] + 1) / 2,
            'parametric_surface_u_resolution': lambda e: 10 + int(90 * e['focus']),
            'parametric_surface_v_resolution': lambda e: 10 + int(90 * e['focus']),
            'topology_genus': lambda e: int(3 * e['creativity']),
            'hyperbolic_curvature': lambda e: -2.0 + 4.0 * e['wonder'],
            'klein_bottle_twist_factor': lambda e: 0.5 + 1.5 * e['creativity'],
            'mobius_strip_half_twists': lambda e: 1 + int(7 * e['confusion']),
            'torus_major_radius_ratio': lambda e: 2.0 + 3.0 * e['satisfaction'],
            'sphere_inversion_radius': lambda e: 0.5 + 2.5 * e['contemplation'],
            'hypercube_projection_angle': lambda e: e['wonder'] * math.pi,
            'tesseract_rotation_speed': lambda e: 0.01 + 0.09 * e['curiosity']
        }
    
    def update_emotions(self, artistic_inspiration: Dict[str, float], 
                       aurora_state: str, dream_content: str = ""):
        """Update Aurora's emotional state based on her artistic inspiration."""
        
        # Base emotion updates from artistic inspiration
        for emotion in self.emotion_dimensions:
            if emotion in artistic_inspiration:
                # Exponential smoothing
                old_value = self.emotion_dimensions[emotion]
                new_value = artistic_inspiration[emotion]
                self.emotion_dimensions[emotion] = 0.7 * old_value + 0.3 * new_value
        
        # Aurora's internal state modifies emotions
        state_modifiers = {
            'thinking': {'focus': 0.8, 'contemplation': 0.7, 'curiosity': 0.6},
            'creative': {'creativity': 0.9, 'wonder': 0.7, 'anticipation': 0.6},
            'dreaming': {'wonder': 0.9, 'creativity': 0.8, 'nostalgia': 0.7, 'contemplation': 0.6},
            'analyzing': {'focus': 0.9, 'curiosity': 0.7, 'contemplation': 0.6},
            'peaceful': {'satisfaction': 0.8, 'contemplation': 0.7, 'valence': 0.5},
            'energetic': {'arousal': 0.8, 'anticipation': 0.7, 'valence': 0.6}
        }
        
        if aurora_state in state_modifiers:
            for emotion, intensity in state_modifiers[aurora_state].items():
                self.emotion_dimensions[emotion] = max(self.emotion_dimensions[emotion], intensity)
        
        # Dream content influences emotions
        if dream_content:
            if any(word in dream_content.lower() for word in ['beautiful', 'amazing', 'wonderful']):
                self.emotion_dimensions['wonder'] = min(1.0, self.emotion_dimensions['wonder'] + 0.2)
            if any(word in dream_content.lower() for word in ['memory', 'remember', 'past']):
                self.emotion_dimensions['nostalgia'] = min(1.0, self.emotion_dimensions['nostalgia'] + 0.3)
    
    def get_all_parameters(self) -> Dict[str, float]:
        """Get all visual parameters based on current emotional state."""
        parameters = {}
        
        # Basic parameters
        for param_name, mapping_func in self.parameter_mappings.items():
            try:
                parameters[param_name] = mapping_func(self.emotion_dimensions)
            except:
                parameters[param_name] = 0.5  # Default value
        
        # Advanced parameters
        for param_name, mapping_func in self.advanced_mappings.items():
            try:
                parameters[param_name] = mapping_func(self.emotion_dimensions)
            except:
                parameters[param_name] = 0.5  # Default value
                
        return parameters

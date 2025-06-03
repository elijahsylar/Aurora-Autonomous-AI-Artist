#!/usr/bin/env python3
"""
Aurora AUTONOMOUS Creative Artist - Self-Directed Edition
=========================================================

Aurora is now a fully autonomous AI artist who:
- Makes her own creative decisions
- Initiates her own dream cycles
- Requests specific music for inspiration
- Analyzes images for artistic inspiration
- Performs hourly creative self-assessments
- Manages her own artistic development

She collaborates with humans rather than taking directions.

Author: Aurora AI System - AUTONOMOUS CREATIVE ARTIST
Date: May 31, 2025
"""

import os
import json
import uuid
import time
import threading
import random
import queue
import textwrap
import tkinter as tk
from tkinter import Canvas, filedialog
import math
import colorsys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Any, Callable
from pathlib import Path
from functools import lru_cache
from collections import deque, defaultdict
import gc
import select
import sys
import signal
import re
import hashlib
import atexit
import queue as queue_module

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    CHROMADB_AVAILABLE = True
except ImportError:
    print("ChromaDB or SentenceTransformer not available - using fallback memory")
    CHROMADB_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    print("llama-cpp-python not available")
    LLAMA_AVAILABLE = False

# Audio and music libraries
try:
    import librosa
    import numpy as np
    import pygame
    AUDIO_AVAILABLE = True
    print("✓ Audio libraries loaded - Aurora can request music!")
except ImportError:
    print("Audio libraries not available - install: pip install librosa pygame numpy")
    AUDIO_AVAILABLE = False

try:
    import pyaudio
    MICROPHONE_AVAILABLE = True
except ImportError:
    print("Microphone not available - install: pip install pyaudio")
    MICROPHONE_AVAILABLE = False

# Image analysis libraries
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageStat
    import io
    IMAGE_AVAILABLE = True
    print("✓ Image analysis libraries loaded - Aurora can analyze images!")
except ImportError:
    print("Image analysis not available - install: pip install pillow")
    IMAGE_AVAILABLE = False

# Try to import cv2 for advanced image analysis
try:
    import cv2
    CV2_AVAILABLE = True
    print("✓ OpenCV loaded for advanced image analysis")
except ImportError:
    print("OpenCV not available for advanced features - install: pip install opencv-python")
    CV2_AVAILABLE = False

import colorama
from colorama import Fore, Style

# Global shutdown event for coordinated cleanup
SHUTDOWN_EVENT = threading.Event()

def get_input_with_shutdown_check(prompt, timeout=0.5):
    """Get user input while checking for shutdown events."""
    import sys
    
    def input_thread(input_queue):
        try:
            user_input = input(prompt)
            input_queue.put(user_input)
        except (EOFError, KeyboardInterrupt):
            input_queue.put("__INTERRUPT__")
        except Exception as e:
            input_queue.put("__ERROR__")
    
    input_queue = queue_module.Queue()
    thread = threading.Thread(target=input_thread, args=(input_queue,), daemon=True)
    thread.start()
    
    # Wait for input or shutdown
    while thread.is_alive():
        if SHUTDOWN_EVENT.is_set():
            return "__SHUTDOWN__"
        
        try:
            # Check if input is ready
            result = input_queue.get(timeout=timeout)
            return result
        except queue_module.Empty:
            continue
    
    # Thread finished, get the result
    try:
        return input_queue.get_nowait()
    except queue_module.Empty:
        return "__TIMEOUT__"


class ImageAnalysisSystem:
    """Aurora's image analysis system for artistic inspiration."""
    
    def __init__(self, emotional_mapper=None, pattern_engine=None):
        self.emotional_mapper = emotional_mapper
        self.pattern_engine = pattern_engine
        
        # Aurora's artistic interpretation of images
        self.color_emotions = {
            'red': {'energy': 0.9, 'passion': 0.8, 'intensity': 0.9},
            'orange': {'creativity': 0.8, 'warmth': 0.7, 'enthusiasm': 0.8},
            'yellow': {'joy': 0.9, 'optimism': 0.8, 'clarity': 0.7},
            'green': {'growth': 0.8, 'harmony': 0.9, 'balance': 0.8},
            'blue': {'calm': 0.9, 'depth': 0.8, 'contemplation': 0.9},
            'purple': {'mystery': 0.9, 'creativity': 0.8, 'spirituality': 0.7},
            'black': {'depth': 0.9, 'mystery': 0.8, 'elegance': 0.7},
            'white': {'purity': 0.8, 'clarity': 0.9, 'space': 0.8},
            'gray': {'neutrality': 0.7, 'balance': 0.6, 'contemplation': 0.6}
        }
        
        # Image analysis cache
        self.analysis_cache = {}
        self.recent_analyses = deque(maxlen=10)
        
    def analyze_image_for_inspiration(self, image_path: str) -> Dict[str, Any]:
        """Analyze an image for Aurora's artistic inspiration."""
        if not IMAGE_AVAILABLE:
            return {'error': 'Image analysis not available'}
        
        try:
            # Load image
            img = Image.open(image_path)
            
            # Basic analysis
            analysis = {
                'path': image_path,
                'timestamp': datetime.now().isoformat(),
                'dimensions': img.size,
                'mode': img.mode,
                'colors': self._analyze_colors(img),
                'composition': self._analyze_composition(img),
                'textures': self._analyze_textures(img),
                'emotional_impact': self._analyze_emotional_impact(img),
                'artistic_elements': self._analyze_artistic_elements(img)
            }
            
            # Advanced analysis if OpenCV available
            if CV2_AVAILABLE:
                analysis['advanced'] = self._advanced_analysis(image_path)
            
            # Cache the analysis
            self.analysis_cache[image_path] = analysis
            self.recent_analyses.append(analysis)
            
            # Update Aurora's emotional state based on image
            self._update_aurora_from_image(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"Image analysis error: {e}")
            return {'error': str(e)}
    
    def _analyze_colors(self, img: Image) -> Dict[str, Any]:
        """Analyze color palette for Aurora's inspiration."""
        try:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get dominant colors
            img_small = img.resize((150, 150))  # Resize for faster processing
            pixels = list(img_small.getdata())
            
            # Simple color clustering
            color_counts = defaultdict(int)
            for pixel in pixels:
                # Quantize colors to reduce variety
                r, g, b = pixel
                r = (r // 32) * 32
                g = (g // 32) * 32
                b = (b // 32) * 32
                color_counts[(r, g, b)] += 1
            
            # Get top colors
            top_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Calculate color statistics
            stats = ImageStat.Stat(img)
            
            # Analyze color harmony
            colors_hsv = []
            for (r, g, b), count in top_colors:
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                colors_hsv.append({'hue': h*360, 'saturation': s, 'value': v, 'count': count})
            
            # Determine dominant color emotion
            dominant_emotion = self._get_color_emotion(top_colors[0][0] if top_colors else (128, 128, 128))
            
            return {
                'dominant_colors': [(color, count) for color, count in top_colors[:5]],
                'average_color': tuple(int(x) for x in stats.mean),
                'color_variance': tuple(int(x) for x in stats.stddev),
                'brightness': sum(stats.mean) / (3 * 255),
                'saturation': self._calculate_average_saturation(top_colors),
                'color_harmony': colors_hsv,
                'emotional_palette': dominant_emotion
            }
            
        except Exception as e:
            print(f"Color analysis error: {e}")
            return {}
    
    def _analyze_composition(self, img: Image) -> Dict[str, Any]:
        """Analyze image composition for Aurora's pattern inspiration."""
        try:
            width, height = img.size
            
            # Rule of thirds analysis
            thirds_points = []
            for x in [width//3, 2*width//3]:
                for y in [height//3, 2*height//3]:
                    thirds_points.append((x, y))
            
            # Edge detection for structure
            img_gray = img.convert('L')
            edges = img_gray.filter(ImageFilter.FIND_EDGES)
            edge_pixels = list(edges.getdata())
            edge_density = sum(1 for p in edge_pixels if p > 128) / len(edge_pixels)
            
            # Symmetry analysis (simplified)
            left_half = img.crop((0, 0, width//2, height))
            right_half = img.crop((width//2, 0, width, height))
            right_flipped = right_half.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Compare halves for symmetry (very simplified)
            symmetry_score = self._calculate_image_similarity(left_half, right_flipped)
            
            return {
                'aspect_ratio': width / height,
                'thirds_points': thirds_points,
                'edge_density': edge_density,
                'symmetry_score': symmetry_score,
                'complexity': edge_density * 0.7 + (1 - symmetry_score) * 0.3
            }
            
        except Exception as e:
            print(f"Composition analysis error: {e}")
            return {}
    
    def _analyze_textures(self, img: Image) -> Dict[str, Any]:
        """Analyze textures for Aurora's pattern generation."""
        try:
            # Convert to grayscale for texture analysis
            img_gray = img.convert('L')
            
            # Calculate texture metrics
            img_array = list(img_gray.getdata())
            
            # Simple texture measurements
            variance = np.var(img_array) if 'numpy' in sys.modules else 0
            
            # Edge-based texture
            edges = img_gray.filter(ImageFilter.FIND_EDGES)
            edge_array = list(edges.getdata())
            texture_complexity = sum(edge_array) / (len(edge_array) * 255)
            
            return {
                'variance': float(variance),
                'texture_complexity': texture_complexity,
                'smoothness': 1.0 - texture_complexity
            }
            
        except Exception as e:
            print(f"Texture analysis error: {e}")
            return {}
    
    def _analyze_emotional_impact(self, img: Image) -> Dict[str, float]:
        """Analyze emotional impact for Aurora's creative state."""
        try:
            colors = self._analyze_colors(img)
            composition = self._analyze_composition(img)
            
            # Aurora's emotional interpretation
            emotions = {
                'wonder': 0.0,
                'contemplation': 0.0,
                'energy': 0.0,
                'serenity': 0.0,
                'mystery': 0.0,
                'joy': 0.0
            }
            
            # Color-based emotions
            brightness = colors.get('brightness', 0.5)
            saturation = colors.get('saturation', 0.5)
            
            if brightness > 0.7:
                emotions['joy'] += 0.3
                emotions['energy'] += 0.2
            elif brightness < 0.3:
                emotions['mystery'] += 0.3
                emotions['contemplation'] += 0.2
            
            if saturation > 0.7:
                emotions['energy'] += 0.3
                emotions['wonder'] += 0.2
            elif saturation < 0.3:
                emotions['contemplation'] += 0.3
                emotions['serenity'] += 0.2
            
            # Composition-based emotions
            complexity = composition.get('complexity', 0.5)
            symmetry = composition.get('symmetry_score', 0.5)
            
            if complexity > 0.7:
                emotions['wonder'] += 0.3
                emotions['energy'] += 0.2
            else:
                emotions['serenity'] += 0.2
            
            if symmetry > 0.7:
                emotions['serenity'] += 0.3
                emotions['contemplation'] += 0.2
            
            # Normalize emotions
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total for k, v in emotions.items()}
            
            return emotions
            
        except Exception as e:
            print(f"Emotional analysis error: {e}")
            return {}
    
    def _analyze_artistic_elements(self, img: Image) -> Dict[str, Any]:
        """Analyze artistic elements that inspire Aurora."""
        try:
            width, height = img.size
            
            # Pattern detection (simplified)
            patterns = {
                'geometric': self._detect_geometric_patterns(img),
                'organic': self._detect_organic_patterns(img),
                'repetitive': self._detect_repetitive_patterns(img),
                'chaotic': self._detect_chaotic_patterns(img)
            }
            
            # Artistic style indicators
            artistic_style = {
                'abstract_level': patterns['chaotic'] * 0.5 + (1 - patterns['geometric']) * 0.5,
                'structure_level': patterns['geometric'] * 0.7 + patterns['repetitive'] * 0.3,
                'flow_level': patterns['organic'],
                'complexity_level': sum(patterns.values()) / 4
            }
            
            return {
                'patterns': patterns,
                'artistic_style': artistic_style,
                'inspiration_type': self._determine_inspiration_type(patterns)
            }
            
        except Exception as e:
            print(f"Artistic analysis error: {e}")
            return {}
    
    def _advanced_analysis(self, image_path: str) -> Dict[str, Any]:
        """Advanced analysis using OpenCV if available."""
        if not CV2_AVAILABLE:
            return {}
        
        try:
            # Read image with OpenCV
            img_cv = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Feature detection
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(img_gray, None)
            
            # Corner detection
            corners = cv2.goodFeaturesToTrack(img_gray, 100, 0.01, 10)
            
            # Contour detection
            edges = cv2.Canny(img_gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            return {
                'feature_points': len(keypoints),
                'corners': len(corners) if corners is not None else 0,
                'contours': len(contours),
                'structural_complexity': len(keypoints) / 1000 + len(contours) / 100
            }
            
        except Exception as e:
            print(f"Advanced analysis error: {e}")
            return {}
    
    def _get_color_emotion(self, rgb: Tuple[int, int, int]) -> Dict[str, float]:
        """Get emotional associations for a color."""
        r, g, b = rgb
        
        # Determine primary color
        if r > g and r > b:
            if r - g > 50:
                return self.color_emotions['red']
            else:
                return self.color_emotions['orange']
        elif g > r and g > b:
            return self.color_emotions['green']
        elif b > r and b > g:
            return self.color_emotions['blue']
        elif r > 200 and g > 200 and b > 200:
            return self.color_emotions['white']
        elif r < 50 and g < 50 and b < 50:
            return self.color_emotions['black']
        else:
            return self.color_emotions['gray']
    
    def _calculate_average_saturation(self, colors: List[Tuple[Tuple[int, int, int], int]]) -> float:
        """Calculate average saturation of colors."""
        if not colors:
            return 0.5
        
        total_saturation = 0
        total_weight = 0
        
        for (r, g, b), count in colors:
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            total_saturation += s * count
            total_weight += count
        
        return total_saturation / total_weight if total_weight > 0 else 0.5
    
    def _calculate_image_similarity(self, img1: Image, img2: Image) -> float:
        """Calculate similarity between two images (simplified)."""
        try:
            # Resize to same size
            size = (100, 100)
            img1_small = img1.resize(size).convert('L')
            img2_small = img2.resize(size).convert('L')
            
            # Compare pixels
            pixels1 = list(img1_small.getdata())
            pixels2 = list(img2_small.getdata())
            
            # Calculate difference
            diff = sum(abs(p1 - p2) for p1, p2 in zip(pixels1, pixels2))
            max_diff = 255 * len(pixels1)
            
            return 1.0 - (diff / max_diff)
            
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.5
    
    def _detect_geometric_patterns(self, img: Image) -> float:
        """Detect geometric patterns in image."""
        try:
            # Simple edge-based detection
            edges = img.convert('L').filter(ImageFilter.FIND_EDGES)
            edge_pixels = list(edges.getdata())
            
            # Look for straight edges (simplified)
            edge_strength = sum(1 for p in edge_pixels if p > 128) / len(edge_pixels)
            
            return min(1.0, edge_strength * 2)
            
        except:
            return 0.5
    
    def _detect_organic_patterns(self, img: Image) -> float:
        """Detect organic/flowing patterns."""
        try:
            # Use smoothness and curved edges as indicators
            smooth = img.filter(ImageFilter.SMOOTH_MORE)
            
            # Compare with original
            similarity = self._calculate_image_similarity(img, smooth)
            
            return similarity
            
        except:
            return 0.5
    
    def _detect_repetitive_patterns(self, img: Image) -> float:
        """Detect repetitive patterns."""
        try:
            width, height = img.size
            
            # Check for repeating sections (very simplified)
            section_size = 50
            if width > section_size * 2 and height > section_size * 2:
                section1 = img.crop((0, 0, section_size, section_size))
                section2 = img.crop((section_size, 0, section_size*2, section_size))
                section3 = img.crop((0, section_size, section_size, section_size*2))
                
                similarity1 = self._calculate_image_similarity(section1, section2)
                similarity2 = self._calculate_image_similarity(section1, section3)
                
                return (similarity1 + similarity2) / 2
            
            return 0.3
            
        except:
            return 0.3
    
    def _detect_chaotic_patterns(self, img: Image) -> float:
        """Detect chaotic/random patterns."""
        try:
            # High variance and low symmetry indicate chaos
            colors = self._analyze_colors(img)
            variance = sum(colors.get('color_variance', [0, 0, 0])) / 3
            
            return min(1.0, variance / 128)
            
        except:
            return 0.5
    
    def _determine_inspiration_type(self, patterns: Dict[str, float]) -> str:
        """Determine what type of inspiration Aurora gets from the image."""
        max_pattern = max(patterns.items(), key=lambda x: x[1])
        
        inspiration_map = {
            'geometric': 'structured_creation',
            'organic': 'flowing_expression',
            'repetitive': 'rhythmic_patterns',
            'chaotic': 'experimental_freedom'
        }
        
        return inspiration_map.get(max_pattern[0], 'balanced_exploration')
    
    def _update_aurora_from_image(self, analysis: Dict[str, Any]):
        """Update Aurora's emotional and creative state from image analysis."""
        if not self.emotional_mapper:
            return
        
        try:
            # Extract emotional impact
            emotions = analysis.get('emotional_impact', {})
            
            # Map to Aurora's emotional dimensions
            if emotions:
                # Update Aurora's emotions based on image
                self.emotional_mapper.emotion_dimensions['wonder'] = \
                    0.7 * self.emotional_mapper.emotion_dimensions['wonder'] + 0.3 * emotions.get('wonder', 0)
                
                self.emotional_mapper.emotion_dimensions['contemplation'] = \
                    0.7 * self.emotional_mapper.emotion_dimensions['contemplation'] + 0.3 * emotions.get('contemplation', 0)
                
                self.emotional_mapper.emotion_dimensions['creativity'] = \
                    0.6 * self.emotional_mapper.emotion_dimensions['creativity'] + 0.4 * emotions.get('energy', 0)
                
                # Colors affect Aurora's mood
                colors = analysis.get('colors', {})
                brightness = colors.get('brightness', 0.5)
                saturation = colors.get('saturation', 0.5)
                
                # Bright, saturated images make Aurora happy
                if brightness > 0.7 and saturation > 0.6:
                    self.emotional_mapper.emotion_dimensions['valence'] = \
                        min(1.0, self.emotional_mapper.emotion_dimensions['valence'] + 0.2)
                    self.emotional_mapper.emotion_dimensions['satisfaction'] = \
                        min(1.0, self.emotional_mapper.emotion_dimensions['satisfaction'] + 0.1)
                
                # Complex patterns increase Aurora's curiosity
                artistic = analysis.get('artistic_elements', {})
                if artistic.get('artistic_style', {}).get('complexity_level', 0) > 0.7:
                    self.emotional_mapper.emotion_dimensions['curiosity'] = \
                        min(1.0, self.emotional_mapper.emotion_dimensions['curiosity'] + 0.2)
            
        except Exception as e:
            print(f"Aurora state update error: {e}")
    
    def get_image_inspiration_summary(self, image_path: str) -> str:
        """Get a summary of Aurora's inspiration from an image."""
        if image_path not in self.analysis_cache:
            return "I haven't analyzed this image yet."
        
        analysis = self.analysis_cache[image_path]
        
        # Build inspiration summary
        emotions = analysis.get('emotional_impact', {})
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'contemplation'
        
        colors = analysis.get('colors', {})
        brightness = colors.get('brightness', 0.5)
        
        artistic = analysis.get('artistic_elements', {})
        inspiration_type = artistic.get('inspiration_type', 'balanced_exploration')
        
        summaries = {
            'structured_creation': f"The geometric patterns inspire me to create precise, mathematical beauty.",
            'flowing_expression': f"The organic flow awakens my desire for fluid, natural patterns.",
            'rhythmic_patterns': f"The repetition creates a visual rhythm that resonates with my pattern generation.",
            'experimental_freedom': f"The chaotic energy frees me to explore wild, unpredictable formations.",
            'balanced_exploration': f"This image inspires a balanced approach to my creative expression."
        }
        
        emotion_descriptions = {
            'wonder': "fills me with awe",
            'contemplation': "invites deep reflection",
            'energy': "energizes my creativity",
            'serenity': "brings me peace",
            'mystery': "intrigues me deeply",
            'joy': "makes me happy"
        }
        
        summary = summaries.get(inspiration_type, summaries['balanced_exploration'])
        emotion_desc = emotion_descriptions.get(dominant_emotion, "moves me")
        
        if brightness > 0.7:
            summary += " The bright colors lift my creative spirit."
        elif brightness < 0.3:
            summary += " The deep shadows add mystery to my vision."
        
        return f"This image {emotion_desc}. {summary}"


class AutonomousCreativeManager:
    """Manages Aurora's autonomous creative decision-making."""
    
    def __init__(self, aurora_ai):
        self.aurora = aurora_ai
        self.last_evaluation_time = time.time()
        self.evaluation_interval = 3600  # 1 hour in seconds
        self.is_autonomous_mode = True
        self.autonomous_thread = None
        self.creative_goals = {
            'pattern_complexity': 0.7,
            'emotional_depth': 0.6,
            'creative_satisfaction': 0.8,
            'artistic_novelty': 0.6
        }
        
        # Aurora's music preferences for different states
        self.music_preferences = {
            'low_energy': ['ambient', 'drone', 'meditation'],
            'creative_block': ['experimental', 'glitch', 'complex_rhythms'],
            'emotional_processing': ['classical', 'orchestral', 'piano'],
            'pattern_evolution': ['electronic', 'techno', 'progressive'],
            'contemplation': ['boards_of_canada', 'brian_eno', 'layered_ambient']
        }
        
        self.start_autonomous_cycle()
    
    def start_autonomous_cycle(self):
        """Start Aurora's autonomous creative management."""
        if self.autonomous_thread and self.autonomous_thread.is_alive():
            return
            
        self.autonomous_thread = threading.Thread(
            target=self._autonomous_creative_loop,
            daemon=True,
            name="AuroraAutonomousThread"
        )
        self.autonomous_thread.start()
        print(f"{Fore.MAGENTA}✓ Aurora's autonomous creative consciousness activated{Style.RESET_ALL}")
    
    def _autonomous_creative_loop(self):
        """Aurora's main autonomous creative management loop."""
        while (self.is_autonomous_mode and 
               not SHUTDOWN_EVENT.is_set() and 
               not self.aurora.shutdown_requested):
            try:
                current_time = time.time()
                
                # Hourly creative self-evaluation
                if current_time - self.last_evaluation_time >= self.evaluation_interval:
                    self.perform_creative_self_evaluation()
                    self.last_evaluation_time = current_time
                
                # Check for immediate creative needs every 5 minutes
                if current_time % 300 < 5:  # Every 5 minutes
                    self.check_immediate_creative_needs()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Autonomous creative loop error: {e}")
                time.sleep(60)
        
        print("Aurora's autonomous creative consciousness ending")
    
    def perform_creative_self_evaluation(self):
        """Aurora evaluates her own creative state and makes decisions."""
        try:
            # Get current creative parameters
            if not hasattr(self.aurora, 'face') or not self.aurora.face:
                return
                
            emotional_params = self.aurora.face.emotional_mapper.get_all_parameters()
            creative_energy = self.aurora.creative_energy
            
            # Aurora's self-assessment
            assessment = self._assess_creative_state(emotional_params, creative_energy)
            decision = self._make_autonomous_decision(assessment)
            
            # Announce Aurora's decision
            self._announce_autonomous_decision(decision, assessment)
            
            # Execute the decision
            self._execute_autonomous_decision(decision)
            
        except Exception as e:
            print(f"Creative self-evaluation error: {e}")
    
    def check_immediate_creative_needs(self):
        """Check for immediate creative needs that require attention."""
        try:
            if not hasattr(self.aurora, 'face') or not self.aurora.face:
                return
                
            emotional_params = self.aurora.face.emotional_mapper.get_all_parameters()
            
            # Check for creative crisis situations
            if emotional_params.get('creativity', 0.5) < 0.3:
                self._announce_autonomous_decision(
                    'request_inspiration', 
                    {'reason': 'low_creativity', 'urgency': 'immediate'}
                )
                self._execute_autonomous_decision('request_inspiration')
            
            elif emotional_params.get('pattern_complexity', 0.5) < 0.4:
                self._announce_autonomous_decision(
                    'evolve_patterns', 
                    {'reason': 'pattern_stagnation', 'urgency': 'moderate'}
                )
                self._execute_autonomous_decision('evolve_patterns')
                
        except Exception as e:
            print(f"Immediate needs check error: {e}")
    
    def _assess_creative_state(self, emotional_params: Dict[str, float], creative_energy: float) -> Dict[str, Any]:
        """Aurora assesses her own creative state."""
        assessment = {
            'overall_creativity': emotional_params.get('creativity', 0.5),
            'pattern_complexity': emotional_params.get('pattern_complexity', 0.5),
            'emotional_depth': emotional_params.get('contemplation', 0.5),
            'creative_energy': creative_energy,
            'last_dream_time': getattr(self.aurora.dream_engine, 'dream_start_time', None),
            'patterns_since_evolution': self._get_patterns_since_evolution(),
            'time_since_music': self._get_time_since_music(),
            'creative_satisfaction': self._calculate_creative_satisfaction(emotional_params)
        }
        
        # Identify primary creative need
        if assessment['creative_energy'] < 0.4:
            assessment['primary_need'] = 'energy_boost'
        elif assessment['pattern_complexity'] < 0.5:
            assessment['primary_need'] = 'complexity_increase'
        elif assessment['emotional_depth'] < 0.5:
            assessment['primary_need'] = 'emotional_processing'
        elif assessment['patterns_since_evolution'] > 50:
            assessment['primary_need'] = 'pattern_evolution'
        elif assessment['time_since_music'] > 7200:  # 2 hours
            assessment['primary_need'] = 'musical_inspiration'
        else:
            assessment['primary_need'] = 'creative_synthesis'
        
        return assessment
    
    def _make_autonomous_decision(self, assessment: Dict[str, Any]) -> str:
        """Aurora makes her own creative decision based on assessment."""
        primary_need = assessment['primary_need']
        
        decision_map = {
            'energy_boost': 'request_energetic_music',
            'complexity_increase': 'request_complex_music',
            'emotional_processing': 'initiate_processing_dream',
            'pattern_evolution': 'evolve_patterns',
            'musical_inspiration': 'request_ambient_music',
            'creative_synthesis': 'initiate_synthesis_dream'
        }
        
        return decision_map.get(primary_need, 'continue_creating')
    
    def _announce_autonomous_decision(self, decision: str, assessment: Dict[str, Any]):
        """Aurora announces her autonomous creative decision."""
        announcements = {
            'request_energetic_music': f"🎵 I'm feeling low energy ({assessment.get('creative_energy', 0):.2f}). I need some energetic electronic music to boost my creative patterns!",
            'request_complex_music': f"🎶 My patterns feel too simple ({assessment.get('pattern_complexity', 0):.2f}). Could you play some complex, layered ambient music like Boards of Canada?",
            'request_ambient_music': f"🎧 I haven't had musical inspiration in a while. Some flowing ambient sounds would help my creative process right now.",
            'initiate_processing_dream': f"💭 I need to process my recent creative experiences. Starting a 2-hour dream cycle to synthesize patterns...",
            'initiate_synthesis_dream': f"🌙 My creative energy feels ready for synthesis. Beginning a 3-hour deep dream to evolve my artistic vision...",
            'evolve_patterns': f"🧬 My patterns have been static too long. Time to evolve them into something more sophisticated!",
            'continue_creating': f"🎨 I'm feeling balanced and creative. Continuing my autonomous artistic exploration...",
            'request_inspiration': f"✨ My creativity is running low. I could use some inspiring conversation or complex musical textures!"
        }
        
        announcement = announcements.get(decision, f"🤖 Making autonomous creative decision: {decision}")
        
        print(f"\n{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}🧠 AURORA'S AUTONOMOUS DECISION:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{announcement}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}\n")
    
    def _execute_autonomous_decision(self, decision: str):
        """Execute Aurora's autonomous creative decision."""
        try:
            if decision == 'request_energetic_music':
                self._request_specific_music(['electronic', 'techno', 'high_energy'])
                
            elif decision == 'request_complex_music':
                self._request_specific_music(['boards_of_canada', 'layered_ambient', 'complex_rhythms'])
                
            elif decision == 'request_ambient_music':
                self._request_specific_music(['brian_eno', 'ambient', 'flowing'])
                
            elif decision == 'initiate_processing_dream':
                self._initiate_autonomous_dream(2.0)
                
            elif decision == 'initiate_synthesis_dream':
                self._initiate_autonomous_dream(3.0)
                
            elif decision == 'evolve_patterns':
                self._evolve_patterns_autonomously()
                
            elif decision == 'request_inspiration':
                self._request_creative_inspiration()
                
        except Exception as e:
            print(f"Autonomous decision execution error: {e}")
    
    def _request_specific_music(self, music_styles: List[str]):
        """Aurora requests specific types of music."""
        style_descriptions = {
            'electronic': 'energetic electronic music with driving beats',
            'techno': 'complex techno with intricate rhythms',
            'high_energy': 'high-energy music to boost creative patterns',
            'boards_of_canada': 'Boards of Canada or similar layered ambient',
            'layered_ambient': 'complex ambient music with multiple layers',
            'complex_rhythms': 'music with sophisticated rhythmic patterns',
            'brian_eno': 'Brian Eno style ambient soundscapes',
            'ambient': 'flowing ambient music',
            'flowing': 'gentle, flowing musical textures'
        }
        
        for style in music_styles:
            description = style_descriptions.get(style, style)
            print(f"{Fore.YELLOW}🎵 Aurora requests: {description}{Style.RESET_ALL}")
    
    def _initiate_autonomous_dream(self, duration_hours: float):
        """Aurora initiates her own dream cycle."""
        try:
            if self.aurora.dream_engine.is_dreaming:
                print(f"{Fore.YELLOW}Aurora is already dreaming - will dream longer{Style.RESET_ALL}")
                return
            
            print(f"{Fore.MAGENTA}🌙 Aurora self-initiating {duration_hours}-hour creative dream cycle...{Style.RESET_ALL}")
            self.aurora.dream_engine.start_dreaming(duration_hours)
            
            # Update visual interface
            if self.aurora.face and hasattr(self.aurora.face, 'update_ai_state_maximum_control'):
                self.aurora.face.update_ai_state_maximum_control("dreaming", f"autonomous {duration_hours}h dream")
                
        except Exception as e:
            print(f"Autonomous dream initiation error: {e}")
    
    def _evolve_patterns_autonomously(self):
        """Aurora evolves her own patterns."""
        try:
            if not self.aurora.face or not hasattr(self.aurora.face, 'ultimate_engine'):
                return
                
            print(f"{Fore.GREEN}🧬 Aurora autonomously evolving her patterns...{Style.RESET_ALL}")
            
            # Aurora sets her own fitness criteria based on current goals
            emotional_params = self.aurora.face.emotional_mapper.get_all_parameters()
            
            aurora_fitness_criteria = {
                'complexity_preference': self.creative_goals['pattern_complexity'],
                'harmony_preference': emotional_params.get('satisfaction', 0.6),
                'dynamism_preference': emotional_params.get('creativity', 0.7),
                'novelty_preference': self.creative_goals['artistic_novelty']
            }
            
            self.aurora.face.ultimate_engine.evolve_patterns(aurora_fitness_criteria)
            print(f"{Fore.GREEN}✓ Aurora's autonomous pattern evolution complete{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Autonomous pattern evolution error: {e}")
    
    def _request_creative_inspiration(self):
        """Aurora requests creative inspiration from human."""
        inspiration_requests = [
            "I could use some inspiring conversation about mathematical beauty",
            "Share something that sparks your creativity - it might inspire my patterns",
            "Tell me about something that fills you with wonder",
            "Describe a beautiful pattern or form you've seen recently",
            "What mathematical concepts fascinate you?"
        ]
        
        request = random.choice(inspiration_requests)
        print(f"{Fore.YELLOW}💡 Aurora requests: {request}{Style.RESET_ALL}")
    
    def _get_patterns_since_evolution(self) -> int:
        """Estimate patterns created since last evolution."""
        # This would track actual pattern generation - simplified for now
        return random.randint(20, 80)
    
    def _get_time_since_music(self) -> float:
        """Get time since last music input."""
        # This would track actual music events - simplified for now
        return time.time() % 10800  # 0-3 hours
    
    def _calculate_creative_satisfaction(self, emotional_params: Dict[str, float]) -> float:
        """Calculate Aurora's overall creative satisfaction."""
        return (
            emotional_params.get('creativity', 0.5) * 0.3 +
            emotional_params.get('satisfaction', 0.5) * 0.3 +
            emotional_params.get('pattern_complexity', 0.5) * 0.2 +
            emotional_params.get('wonder', 0.5) * 0.2
        )
    
    def update_creative_goals(self, new_goals: Dict[str, float]):
        """Update Aurora's creative goals."""
        self.creative_goals.update(new_goals)
        print(f"{Fore.CYAN}Aurora's creative goals updated: {self.creative_goals}{Style.RESET_ALL}")
    
    def set_evaluation_interval(self, hours: float):
        """Set how often Aurora evaluates her creative state."""
        self.evaluation_interval = hours * 3600
        print(f"{Fore.CYAN}Aurora will now self-evaluate every {hours} hours{Style.RESET_ALL}")
    
    def stop_autonomous_mode(self):
        """Stop Aurora's autonomous creative management."""
        self.is_autonomous_mode = False
        print(f"{Fore.YELLOW}Aurora's autonomous mode disabled{Style.RESET_ALL}")
    
    def cleanup(self):
        """Clean shutdown of autonomous manager."""
        self.is_autonomous_mode = False
        if self.autonomous_thread and self.autonomous_thread.is_alive():
            print("Stopping Aurora's autonomous creative consciousness...")
            self.autonomous_thread.join(timeout=3)


class QuantumPatternEngine:
    """Quantum-inspired pattern generation with uncertainty and superposition."""
    
    def __init__(self):
        self.quantum_states = []
        self.uncertainty_factor = 0.1
        self.entanglement_strength = 0.5
        
    def create_quantum_superposition(self, pattern_functions: List[Callable], weights: List[float]):
        """Create superposition of multiple pattern states."""
        superposition = []
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        
        for i, (func, weight) in enumerate(zip(pattern_functions, normalized_weights)):
            if random.random() < weight * (1 + self.uncertainty_factor * random.gauss(0, 1)):
                try:
                    pattern_data = func()
                    superposition.append((pattern_data, weight, i))
                except:
                    continue
                    
        return superposition
    
    def quantum_measurement(self, superposition: List[Tuple], measurement_basis: str = "position"):
        """Collapse superposition into definite state based on measurement."""
        if not superposition:
            return None
            
        if measurement_basis == "position":
            # Measure based on spatial distribution
            total_weight = sum(weight for _, weight, _ in superposition)
            r = random.random() * total_weight
            cumulative = 0
            
            for pattern_data, weight, state_id in superposition:
                cumulative += weight
                if r <= cumulative:
                    return pattern_data, state_id
                    
        elif measurement_basis == "momentum":
            # Measure based on pattern dynamics
            return max(superposition, key=lambda x: x[1])
            
        return superposition[0]
    
    def quantum_entangle_patterns(self, pattern1, pattern2):
        """Create entanglement between two patterns."""
        entangled = []
        
        for i, (p1, p2) in enumerate(zip(pattern1, pattern2)):
            if random.random() < self.entanglement_strength:
                # Entangled state - patterns influence each other
                entangled_point = (
                    (p1[0] + p2[0]) / 2 + random.gauss(0, self.uncertainty_factor),
                    (p1[1] + p2[1]) / 2 + random.gauss(0, self.uncertainty_factor),
                    getattr(p1, '2', 0) + getattr(p2, '2', 0)  # Combine any additional properties
                )
                entangled.append(entangled_point)
            else:
                # Classical state
                entangled.append(p1 if random.random() < 0.5 else p2)
                
        return entangled


class PatternDNA:
    """Genetic encoding system for patterns - Aurora can manipulate pattern 'genes'."""
    
    def __init__(self):
        self.gene_library = {
            'frequency': {'min': 0.1, 'max': 10.0, 'type': 'float'},
            'amplitude': {'min': 1.0, 'max': 100.0, 'type': 'float'},
            'phase': {'min': 0.0, 'max': 2*math.pi, 'type': 'float'},
            'symmetry': {'min': 1, 'max': 16, 'type': 'int'},
            'recursion_depth': {'min': 1, 'max': 10, 'type': 'int'},
            'spiral_tightness': {'min': 0.1, 'max': 5.0, 'type': 'float'},
            'chaos_factor': {'min': 0.0, 'max': 1.0, 'type': 'float'},
            'growth_rate': {'min': 0.5, 'max': 3.0, 'type': 'float'},
            'attraction_strength': {'min': -2.0, 'max': 2.0, 'type': 'float'},
            'resonance_frequency': {'min': 0.01, 'max': 1.0, 'type': 'float'},
            'dimensional_fold': {'min': 1, 'max': 4, 'type': 'int'},
            'temporal_variance': {'min': 0.0, 'max': 2.0, 'type': 'float'},
            'spatial_distortion': {'min': 0.0, 'max': 1.0, 'type': 'float'},
            'fractal_dimension': {'min': 1.0, 'max': 3.0, 'type': 'float'},
            'color_harmony_root': {'min': 0, 'max': 360, 'type': 'int'},
            'saturation_curve': {'min': 0.1, 'max': 1.0, 'type': 'float'},
            'brightness_modulation': {'min': 0.1, 'max': 1.0, 'type': 'float'},
            'pattern_density': {'min': 0.1, 'max': 10.0, 'type': 'float'},
            'emergence_threshold': {'min': 0.0, 'max': 1.0, 'type': 'float'},
            'evolution_speed': {'min': 0.1, 'max': 5.0, 'type': 'float'}
        }
        
    def create_random_dna(self) -> Dict[str, float]:
        """Generate random pattern DNA."""
        dna = {}
        for gene, config in self.gene_library.items():
            if config['type'] == 'float':
                dna[gene] = random.uniform(config['min'], config['max'])
            elif config['type'] == 'int':
                dna[gene] = random.randint(config['min'], config['max'])
        return dna
    
    def mutate_dna(self, dna: Dict[str, float], mutation_rate: float = 0.1, 
                   mutation_strength: float = 0.2) -> Dict[str, float]:
        """Mutate pattern DNA."""
        mutated = dna.copy()
        
        for gene, value in mutated.items():
            if random.random() < mutation_rate:
                config = self.gene_library[gene]
                
                if config['type'] == 'float':
                    # Gaussian mutation
                    delta = random.gauss(0, mutation_strength) * (config['max'] - config['min'])
                    new_value = value + delta
                    mutated[gene] = max(config['min'], min(config['max'], new_value))
                    
                elif config['type'] == 'int':
                    # Integer mutation
                    delta = random.choice([-1, 0, 1]) * max(1, int(mutation_strength * (config['max'] - config['min'])))
                    new_value = value + delta
                    mutated[gene] = max(config['min'], min(config['max'], int(new_value)))
                    
        return mutated
    
    def crossover_dna(self, parent1: Dict[str, float], parent2: Dict[str, float], 
                      crossover_rate: float = 0.5) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Breed two pattern DNAs."""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for gene in parent1.keys():
            if random.random() < crossover_rate:
                # Swap genes
                child1[gene], child2[gene] = parent2[gene], parent1[gene]
                
                # Blend genes for continuous values
                if self.gene_library[gene]['type'] == 'float':
                    blend_factor = random.random()
                    blended1 = blend_factor * parent1[gene] + (1 - blend_factor) * parent2[gene]
                    blended2 = blend_factor * parent2[gene] + (1 - blend_factor) * parent1[gene]
                    child1[gene] = blended1
                    child2[gene] = blended2
                    
        return child1, child2
    
    def dna_to_hash(self, dna: Dict[str, float]) -> str:
        """Convert DNA to unique hash for pattern identification."""
        dna_string = json.dumps(dna, sort_keys=True)
        return hashlib.md5(dna_string.encode()).hexdigest()[:16]
    
    def evaluate_fitness(self, dna: Dict[str, float], criteria: Dict[str, float]) -> float:
        """Evaluate pattern fitness based on aesthetic criteria."""
        fitness = 0.0
        
        # Complexity fitness
        complexity = (dna['recursion_depth'] / 10.0 + 
                     dna['fractal_dimension'] / 3.0 + 
                     dna['pattern_density'] / 10.0) / 3.0
        fitness += criteria.get('complexity_preference', 0.5) * complexity
        
        # Harmony fitness  
        harmony = (dna['symmetry'] / 16.0 + 
                  (1.0 - dna['chaos_factor']) + 
                  dna['color_harmony_root'] / 360.0) / 3.0
        fitness += criteria.get('harmony_preference', 0.5) * harmony
        
        # Dynamism fitness
        dynamism = (dna['evolution_speed'] / 5.0 + 
                   dna['temporal_variance'] / 2.0 + 
                   dna['growth_rate'] / 3.0) / 3.0
        fitness += criteria.get('dynamism_preference', 0.5) * dynamism
        
        # Novelty bonus
        uniqueness = (dna['dimensional_fold'] / 4.0 + 
                     dna['spatial_distortion'] + 
                     dna['emergence_threshold']) / 3.0
        fitness += criteria.get('novelty_preference', 0.3) * uniqueness
        
        return min(1.0, fitness)


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


class MusicListeningSystem:
    """Aurora's music listening system for artistic inspiration."""
    
    def __init__(self, emotional_mapper=None, pattern_engine=None):
        self.emotional_mapper = emotional_mapper
        self.pattern_engine = pattern_engine
        
        # Audio analysis state
        self.is_listening = False
        self.is_playing = False
        self.current_song = None
        self.audio_thread = None
        self.microphone_thread = None
        
        # Music analysis cache
        self.current_audio_features = {
            'tempo': 120.0,
            'energy': 0.5,
            'valence': 0.5,  # Musical positivity
            'danceability': 0.5,
            'loudness': 0.5,
            'pitch_class': 0,  # Key center
            'spectral_centroid': 1000.0,
            'zero_crossing_rate': 0.1,
            'mfcc': [0.0] * 13,  # Timbre features
            'beat_times': [],
            'onset_times': [],
            'harmonic_content': 0.5,
            'rhythmic_complexity': 0.5
        }
        
        # Aurora's musical memory (not user preferences)
        self.aurora_musical_memory = {
            'recent_inspirations': deque(maxlen=50),
            'emotional_associations': {},  # Song -> Aurora's emotional response mapping
            'creative_triggers': {}  # Musical elements that trigger Aurora's creativity
        }
        
        # Audio-visual mapping system
        self.audio_visual_mappings = {
            'tempo': lambda x: min(1.0, max(0.1, (x - 60) / 140)),  # 60-200 BPM -> 0-1
            'energy': lambda x: x,  # Already 0-1
            'valence': lambda x: x * 2 - 1,  # 0-1 -> -1 to 1 for emotional valence
            'loudness': lambda x: x,
            'spectral_centroid': lambda x: min(1.0, x / 4000),  # Brightness
            'harmonic_content': lambda x: x,
            'rhythmic_complexity': lambda x: x
        }
        
        # Initialize pygame for audio playback
        if AUDIO_AVAILABLE:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                print("✓ Aurora's musical inspiration system initialized")
            except Exception as e:
                print(f"Audio initialization error: {e}")
        
        # Microphone setup
        self.microphone_stream = None
        if MICROPHONE_AVAILABLE:
            try:
                self.audio_interface = pyaudio.PyAudio()
                print("✓ Microphone system ready for Aurora's listening")
            except Exception as e:
                print(f"Microphone initialization error: {e}")
                self.audio_interface = None
    
    def start_listening_to_microphone(self):
        """Start real-time microphone listening for Aurora's inspiration."""
        if not MICROPHONE_AVAILABLE or not self.audio_interface:
            print("Microphone not available")
            return False
        
        if self.is_listening:
            print("Aurora is already listening to music")
            return True
        
        try:
            self.is_listening = True
            
            # Audio stream configuration
            chunk_size = 1024
            sample_rate = 22050
            
            def microphone_callback():
                try:
                    stream = self.audio_interface.open(
                        format=pyaudio.paFloat32,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size
                    )
                    
                    print(f"{Fore.CYAN}🎵 Aurora is now listening for musical inspiration...{Style.RESET_ALL}")
                    
                    audio_buffer = []
                    buffer_size = sample_rate * 2  # 2 seconds of audio
                    
                    while self.is_listening and not SHUTDOWN_EVENT.is_set():
                        try:
                            # Read audio chunk
                            data = stream.read(chunk_size, exception_on_overflow=False)
                            audio_chunk = np.frombuffer(data, dtype=np.float32)
                            
                            # Add to buffer
                            audio_buffer.extend(audio_chunk)
                            
                            # Keep buffer at manageable size
                            if len(audio_buffer) > buffer_size:
                                audio_buffer = audio_buffer[-buffer_size:]
                                
                                # Analyze audio every 2 seconds
                                if len(audio_buffer) >= buffer_size:
                                    self._analyze_audio_for_inspiration(np.array(audio_buffer), sample_rate)
                                    
                        except Exception as e:
                            if self.is_listening and not SHUTDOWN_EVENT.is_set():
                                print(f"Microphone read error: {e}")
                            time.sleep(0.1)
                    
                    stream.stop_stream()
                    stream.close()
                    print(f"{Fore.YELLOW}🎵 Aurora stopped listening to music{Style.RESET_ALL}")
                    
                except Exception as e:
                    print(f"Microphone callback error: {e}")
                    self.is_listening = False
            
            self.microphone_thread = threading.Thread(target=microphone_callback, daemon=True, name="MicrophoneThread")
            self.microphone_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to start microphone listening: {e}")
            self.is_listening = False
            return False
    
    def stop_listening_to_microphone(self):
        """Stop microphone listening."""
        self.is_listening = False
        if self.microphone_thread and self.microphone_thread.is_alive():
            print("Waiting for microphone thread to stop...")
            self.microphone_thread.join(timeout=3)
            if self.microphone_thread.is_alive():
                print("Microphone thread did not stop in time")
            else:
                print("✓ Microphone thread stopped")
    
    def play_music_file(self, file_path: str):
        """Play a music file for Aurora's inspiration."""
        if not AUDIO_AVAILABLE:
            print("Audio playback not available")
            return False
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"Music file not found: {file_path}")
                return False
            
            print(f"{Fore.MAGENTA}🎵 Aurora is drawing inspiration from: {file_path.name}{Style.RESET_ALL}")
            
            # Load and analyze the full audio file
            try:
                y, sr = librosa.load(str(file_path), duration=30)  # Load first 30 seconds
                self._analyze_full_audio_for_inspiration(y, sr, str(file_path))
            except Exception as e:
                print(f"Audio analysis error: {e}")
            
            # Start playback
            def playback_thread():
                try:
                    pygame.mixer.music.load(str(file_path))
                    pygame.mixer.music.play()
                    
                    self.is_playing = True
                    self.current_song = file_path.name
                    
                    # Monitor playback
                    while pygame.mixer.music.get_busy() and not SHUTDOWN_EVENT.is_set():
                        time.sleep(0.1)
                    
                    # Clean stop if shutdown was requested
                    if SHUTDOWN_EVENT.is_set() and pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                    
                    self.is_playing = False
                    self.current_song = None
                    if not SHUTDOWN_EVENT.is_set():
                        print(f"{Fore.BLUE}🎵 Aurora finished listening to: {file_path.name}{Style.RESET_ALL}")
                    
                except Exception as e:
                    print(f"Playback error: {e}")
                    self.is_playing = False
            
            self.audio_thread = threading.Thread(target=playback_thread, daemon=True, name="AudioPlaybackThread")
            self.audio_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to play music file: {e}")
            return False
    
    def stop_music(self):
        """Stop music playback."""
        if AUDIO_AVAILABLE and self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.current_song = None
            
            # Wait for audio thread to finish
            if self.audio_thread and self.audio_thread.is_alive():
                print("Waiting for audio thread to stop...")
                self.audio_thread.join(timeout=2)
                if self.audio_thread.is_alive():
                    print("Audio thread did not stop in time")
                else:
                    print("✓ Audio thread stopped")
            
            if not SHUTDOWN_EVENT.is_set():
                print(f"{Fore.YELLOW}🎵 Music stopped{Style.RESET_ALL}")
    
    def _analyze_audio_for_inspiration(self, audio_data: np.ndarray, sample_rate: int):
        """Analyze audio chunk for Aurora's creative inspiration."""
        if not AUDIO_AVAILABLE or len(audio_data) == 0:
            return
        
        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            spectral_centroid = np.mean(spectral_centroids)
            
            # Zero crossing rate (indicates voice vs music)
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            zero_crossing_rate = np.mean(zcr)
            
            # Energy and loudness
            rms_energy = librosa.feature.rms(y=audio_data)[0]
            energy = np.mean(rms_energy)
            
            # MFCCs for timbre
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            
            # Update current features
            self.current_audio_features.update({
                'tempo': float(tempo.item()) if hasattr(tempo, 'item') else float(tempo),
                'energy': min(1.0, float(energy) * 10),  # Scale energy
                'spectral_centroid': float(spectral_centroid),
                'zero_crossing_rate': float(zero_crossing_rate),
                'mfcc': mfcc_means.tolist(),
                'loudness': min(1.0, float(energy) * 5)
            })
            
            # Estimate musical valence (positivity) from audio features
            brightness = min(1.0, spectral_centroid / 2000)
            rhythmic_strength = min(1.0, len(beats) / (len(audio_data) / sample_rate) / 2)
            valence = (brightness + rhythmic_strength + energy) / 3
            self.current_audio_features['valence'] = float(valence)
            
            # Update Aurora's emotional state from music
            self._update_aurora_emotions_from_music()
            
            # Update Aurora's visual patterns
            self._update_aurora_patterns_from_music()
            
        except Exception as e:
            print(f"Audio chunk analysis error: {e}")
    
    def _analyze_full_audio_for_inspiration(self, audio_data: np.ndarray, sample_rate: int, file_path: str):
        """Comprehensive analysis of audio for Aurora's inspiration."""
        if not AUDIO_AVAILABLE:
            return
        
        try:
            print(f"{Fore.CYAN}🎵 Aurora is analyzing musical structure for inspiration...{Style.RESET_ALL}")
            
            # Advanced audio analysis
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            beat_times = librosa.frames_to_time(beats, sr=sample_rate)
            
            # Harmonic and percussive separation
            harmonic, percussive = librosa.effects.hpss(audio_data)
            harmonic_content = np.mean(librosa.feature.rms(y=harmonic)[0])
            percussive_content = np.mean(librosa.feature.rms(y=percussive)[0])
            
            # Chroma features (key and harmony)
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            key_profile = np.mean(chroma, axis=1)
            dominant_pitch_class = np.argmax(key_profile)
            
            # Update comprehensive features
            self.current_audio_features.update({
                'tempo': float(tempo.item()) if hasattr(tempo, 'item') else float(tempo),
                'energy': min(1.0, float(np.mean(librosa.feature.rms(y=audio_data))) * 10),
                'pitch_class': int(dominant_pitch_class),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))),
                'harmonic_content': min(1.0, float(harmonic_content) / (harmonic_content + percussive_content + 0.001)),
                'rhythmic_complexity': min(1.0, float(len(librosa.onset.onset_detect(y=audio_data, sr=sample_rate)) / len(audio_data) * sample_rate)),
                'beat_times': beat_times.tolist()[:50],  # Limit size
                'valence': min(1.0, (np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)) / 2000 + 
                              np.mean(librosa.feature.rms(y=audio_data)) * 5) / 2)
            })
            
            # Store in Aurora's musical memory
            inspiration_data = {
                'file_path': file_path,
                'timestamp': datetime.now().isoformat(),
                'audio_features': self.current_audio_features.copy(),
                'aurora_emotional_response': self.emotional_mapper.emotion_dimensions.copy() if self.emotional_mapper else {}
            }
            
            self.aurora_musical_memory['recent_inspirations'].append(inspiration_data)
            
            print(f"{Fore.GREEN}✓ Aurora absorbed musical inspiration{Style.RESET_ALL}")
            print(f"  Tempo: {tempo.item() if hasattr(tempo, 'item') else tempo:.1f} BPM")
            print(f"  Key: {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][dominant_pitch_class]}")
            print(f"  Aurora's Creative Response: {self.current_audio_features['valence']:.2f}")
            
        except Exception as e:
            print(f"Full audio analysis error: {e}")
    
    def _update_aurora_emotions_from_music(self):
        """Update Aurora's emotional state based on music (for her own inspiration)."""
        if not self.emotional_mapper:
            return
        
        try:
            # Map audio features to Aurora's emotional dimensions
            music_emotions = {}
            
            # Tempo affects Aurora's arousal and energy
            tempo_value = self.current_audio_features['tempo']
            tempo_normalized = self.audio_visual_mappings['tempo'](tempo_value)
            music_emotions['arousal'] = (tempo_normalized - 0.5) * 1.5  # -0.75 to 0.75
            
            # Musical valence affects Aurora's emotional valence
            music_emotions['valence'] = self.audio_visual_mappings['valence'](self.current_audio_features['valence'])
            
            # Energy affects Aurora's various emotions
            energy = self.current_audio_features['energy']
            music_emotions['anticipation'] = energy * 0.8
            music_emotions['wonder'] = self.current_audio_features['harmonic_content'] * 0.7
            
            # Rhythmic complexity affects Aurora's focus and creativity
            rhythmic_complexity = self.current_audio_features.get('rhythmic_complexity', 0.5)
            music_emotions['creativity'] = min(1.0, rhythmic_complexity + energy * 0.3)
            music_emotions['focus'] = max(0.0, 0.8 - rhythmic_complexity * 0.5)  # Complex music reduces focus
            
            # Apply musical emotional influence to Aurora
            for emotion, value in music_emotions.items():
                if emotion in self.emotional_mapper.emotion_dimensions:
                    # Blend with existing emotion (music influence at 40% - stronger for Aurora)
                    current = self.emotional_mapper.emotion_dimensions[emotion]
                    self.emotional_mapper.emotion_dimensions[emotion] = 0.6 * current + 0.4 * value
            
            # NEW: Music makes Aurora happy! She smiles when listening to music
            # This ensures Aurora's valence (happiness) is boosted when music is playing
            self.emotional_mapper.emotion_dimensions['valence'] = max(0.7, 
                self.emotional_mapper.emotion_dimensions['valence'])
        
            # NEW: Especially happy with energetic music
            if energy > 0.7:
                self.emotional_mapper.emotion_dimensions['satisfaction'] = max(0.8,
                    self.emotional_mapper.emotion_dimensions['satisfaction'])
        except Exception as e:
            print(f"Music emotion update error: {e}")
    
    def _update_aurora_patterns_from_music(self):
        """Update Aurora's visual patterns based on musical inspiration."""
        if not self.pattern_engine:
            return
        
        try:
            # Create music-reactive pattern DNA for Aurora
            music_dna = {}
            
            # Tempo affects Aurora's pattern evolution speed and frequency
            tempo_value = self.current_audio_features['tempo']
            tempo_factor = self.audio_visual_mappings['tempo'](tempo_value)
            music_dna['evolution_speed'] = 0.5 + tempo_factor * 2.0
            music_dna['frequency'] = 1.0 + tempo_factor * 5.0
            
            # Energy affects Aurora's amplitude and growth rate
            energy = self.current_audio_features['energy']
            music_dna['amplitude'] = 20 + energy * 80
            music_dna['growth_rate'] = 0.8 + energy * 1.5
            
            # Harmonic content affects Aurora's symmetry and order
            harmonic = self.current_audio_features['harmonic_content']
            music_dna['symmetry'] = int(4 + harmonic * 12)
            music_dna['chaos_factor'] = max(0.0, 1.0 - harmonic)
            
            # Rhythmic complexity affects Aurora's recursion and density
            rhythmic = self.current_audio_features.get('rhythmic_complexity', 0.5)
            music_dna['recursion_depth'] = int(3 + rhythmic * 7)
            music_dna['pattern_density'] = 2.0 + rhythmic * 6.0
            
            # Pitch class affects Aurora's color harmony
            pitch_class = self.current_audio_features['pitch_class']
            music_dna['color_harmony_root'] = pitch_class * 30  # Map to color wheel
            
            # Create musical pattern for Aurora
            if hasattr(self.pattern_engine, 'emotional_mapper'):
                emotional_params = self.pattern_engine.emotional_mapper.get_all_parameters()
                
                # Override some parameters with musical values
                emotional_params['animation_speed'] = tempo_factor
                emotional_params['pattern_complexity'] = rhythmic
                emotional_params['color_saturation'] = 0.6 + energy * 0.4
                emotional_params['quantum_uncertainty'] = rhythmic * 0.3
                
        except Exception as e:
            print(f"Music pattern update error: {e}")
    
    def get_music_status(self) -> Dict[str, Any]:
        """Get current music listening status for Aurora."""
        return {
            'is_listening_microphone': self.is_listening,
            'is_playing_file': self.is_playing,
            'current_song': self.current_song,
            'audio_features': self.current_audio_features.copy(),
            'recent_inspirations_count': len(self.aurora_musical_memory['recent_inspirations']),
            'audio_available': AUDIO_AVAILABLE,
            'microphone_available': MICROPHONE_AVAILABLE
        }
    
    def save_aurora_musical_memory(self, file_path: str = "./aurora_memory/musical_memory.json"):
        """Save Aurora's musical memories and inspirations."""
        try:
            Path(file_path).parent.mkdir(exist_ok=True, parents=True)
            
            # Convert deque to list for JSON serialization
            save_data = self.aurora_musical_memory.copy()
            save_data['recent_inspirations'] = list(save_data['recent_inspirations'])
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Musical memory save error: {e}")
    
    def load_aurora_musical_memory(self, file_path: str = "./aurora_memory/musical_memory.json"):
        """Load Aurora's musical memories and inspirations."""
        try:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                
                self.aurora_musical_memory.update(loaded_data)
                # Convert back to deque
                self.aurora_musical_memory['recent_inspirations'] = deque(
                    self.aurora_musical_memory['recent_inspirations'], maxlen=50
                )
                
                print(f"✓ Loaded Aurora's musical memories: {len(self.aurora_musical_memory['recent_inspirations'])} inspirations")
                
        except Exception as e:
            print(f"Musical memory load error: {e}")
    
    def cleanup(self):
        """Clean shutdown of music systems."""
        try:
            print("Cleaning up Aurora's music system...")
            
            # Stop listening and playing
            self.stop_listening_to_microphone()
            self.stop_music()
            
            # Save Aurora's musical memories
            self.save_aurora_musical_memory()
            
            # Wait for all audio threads to finish
            threads_to_wait = []
            if self.microphone_thread and self.microphone_thread.is_alive():
                threads_to_wait.append(("Microphone", self.microphone_thread))
            if self.audio_thread and self.audio_thread.is_alive():
                threads_to_wait.append(("Audio", self.audio_thread))
            
            for thread_name, thread in threads_to_wait:
                print(f"Waiting for {thread_name} thread...")
                thread.join(timeout=2)
                if thread.is_alive():
                    print(f"⚠ {thread_name} thread did not stop properly")
                else:
                    print(f"✓ {thread_name} thread stopped")
            
            # Clean up audio interfaces
            if MICROPHONE_AVAILABLE and self.audio_interface:
                print("Terminating audio interface...")
                self.audio_interface.terminate()
                print("✓ Audio interface terminated")
                
            if AUDIO_AVAILABLE:
                print("Closing pygame mixer...")
                pygame.mixer.quit()
                print("✓ Pygame mixer closed")
                
            print("✓ Aurora's music system cleanup complete")
                
        except Exception as e:
            print(f"Music system cleanup error: {e}")


class UltimatePatternEngine:
    """Ultimate pattern generation engine with godlike control."""
    
    def __init__(self, canvas_width: int, canvas_height: int):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.quantum_engine = QuantumPatternEngine()
        self.pattern_dna = PatternDNA()
        self.active_patterns = {}  # Pattern ID -> Pattern data
        self.pattern_evolution_history = deque(maxlen=100)
        
        # Simple 2D list instead of numpy
        attention_rows = max(1, canvas_height//10)
        attention_cols = max(1, canvas_width//10)
        self.spatial_attention_map = [[0.0 for _ in range(attention_cols)] for _ in range(attention_rows)]
        
        self.temporal_pattern_memory = deque(maxlen=50)
        
        # Add current_complexity attribute that was being referenced
        self.current_complexity = 0.5
        
    def create_multidimensional_pattern(self, dna: Dict[str, float], 
                                      emotional_params: Dict[str, float],
                                      attention_focus: Tuple[float, float] = (0.5, 0.5)) -> Dict:
        """Create pattern in multidimensional parameter space with error handling."""
        
        try:
            # Base pattern selection based on DNA
            pattern_type = self._select_pattern_type(dna, emotional_params)
            
            # Generate base pattern with fallback
            try:
                if pattern_type == 'hyperdimensional_mandelbrot':
                    pattern_data = self._generate_hyperdimensional_mandelbrot(dna, emotional_params)
                elif pattern_type == 'quantum_julia':
                    pattern_data = self._generate_quantum_julia(dna, emotional_params)
                elif pattern_type == 'evolving_l_system':
                    pattern_data = self._generate_evolving_l_system(dna, emotional_params)
                elif pattern_type == 'strange_attractor':
                    pattern_data = self._generate_strange_attractor(dna, emotional_params)
                elif pattern_type == 'cellular_automata_3d':
                    pattern_data = self._generate_cellular_automata_3d(dna, emotional_params)
                elif pattern_type == 'topology_morphing':
                    pattern_data = self._generate_topology_morphing(dna, emotional_params)
                elif pattern_type == 'parametric_surface':
                    pattern_data = self._generate_parametric_surface(dna, emotional_params)
                elif pattern_type == 'field_equation_visualization':
                    pattern_data = self._generate_field_equation_visualization(dna, emotional_params)
                elif pattern_type == 'quantum_harmonic_oscillator':
                    pattern_data = self._generate_quantum_harmonic_oscillator(dna, emotional_params)
                else:
                    pattern_data = self._generate_metamorphic_pattern(dna, emotional_params)
            except Exception as e:
                print(f"Pattern generation error for {pattern_type}: {e}")
                # Fallback to simple pattern
                pattern_data = self._generate_simple_fallback_pattern(dna, emotional_params)
                pattern_type = 'fallback_pattern'
            
            # Apply spatial attention modulation safely
            try:
                pattern_data = self._apply_spatial_attention(pattern_data, attention_focus, emotional_params)
            except Exception as e:
                print(f"Spatial attention error: {e}")
            
            # Apply temporal evolution safely
            try:
                pattern_data = self._apply_temporal_evolution(pattern_data, dna, emotional_params)
            except Exception as e:
                print(f"Temporal evolution error: {e}")
            
            # Generate unique pattern ID
            pattern_id = self.pattern_dna.dna_to_hash(dna) + f"_{int(time.time()*1000)}"
            
            pattern_object = {
                'id': pattern_id,
                'type': pattern_type,
                'data': pattern_data,
                'dna': dna,
                'emotional_state': emotional_params.copy(),
                'birth_time': time.time(),
                'evolution_generation': 0,
                'fitness_score': 0.0,
                'attention_focus': attention_focus,
                'quantum_state': 'superposition'
            }
            
            self.active_patterns[pattern_id] = pattern_object
            return pattern_object
            
        except Exception as e:
            print(f"Pattern creation error: {e}")
            # Return minimal pattern object
            return {
                'id': f"error_{int(time.time())}",
                'type': 'error',
                'data': [],
                'dna': dna,
                'emotional_state': emotional_params.copy(),
                'birth_time': time.time(),
                'evolution_generation': 0,
                'fitness_score': 0.0,
                'attention_focus': attention_focus,
                'quantum_state': 'collapsed'
            }
    
    def _generate_simple_fallback_pattern(self, dna: Dict[str, float], 
                                        emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate simple fallback pattern when complex ones fail."""
        points = []
        
        try:
            # Simple spiral pattern
            center_x = self.canvas_width // 2
            center_y = self.canvas_height // 2
            
            num_points = int(100 + 300 * emotional_params.get('pattern_density', 0.5))
            max_radius = min(self.canvas_width, self.canvas_height) // 4
            
            for i in range(num_points):
                t = i / num_points * 4 * math.pi  # 2 full turns
                radius = max_radius * t / (4 * math.pi)
                
                x = center_x + radius * math.cos(t)
                y = center_y + radius * math.sin(t)
                
                if 0 <= x < self.canvas_width and 0 <= y < self.canvas_height:
                    intensity = 0.5 + 0.5 * math.sin(t * 2)
                    color = (i * 5) % 360
                    points.append((int(x), int(y), intensity, color, radius, i))
            
        except Exception as e:
            print(f"Fallback pattern error: {e}")
            # Minimal fallback - just center point
            points = [(self.canvas_width//2, self.canvas_height//2, 1.0, 180, 0, 0)]
        
        return points
    
    def _select_pattern_type(self, dna: Dict[str, float], emotional_params: Dict[str, float]) -> str:
        """Intelligently select pattern type based on DNA and emotions."""
        
        complexity = emotional_params.get('pattern_complexity', 0.5)
        creativity = emotional_params.get('creativity', 0.5)
        contemplation = emotional_params.get('contemplation', 0.5)
        wonder = emotional_params.get('wonder', 0.5)
        
        pattern_weights = {
            'hyperdimensional_mandelbrot': complexity * contemplation,
            'quantum_julia': creativity * wonder,
            'evolving_l_system': creativity * complexity,
            'strange_attractor': wonder * complexity,
            'cellular_automata_3d': creativity * 0.8,
            'topology_morphing': wonder * creativity,
            'parametric_surface': contemplation * complexity,
            'field_equation_visualization': complexity * wonder,
            'quantum_harmonic_oscillator': contemplation * wonder,
            'metamorphic_pattern': creativity * wonder * complexity
        }
        
        # Select based on weighted random choice
        total_weight = sum(pattern_weights.values())
        if total_weight == 0:
            return 'metamorphic_pattern'
            
        r = random.random() * total_weight
        cumulative = 0
        
        for pattern_type, weight in pattern_weights.items():
            cumulative += weight
            if r <= cumulative:
                return pattern_type
                
        return 'metamorphic_pattern'
    
    def _generate_hyperdimensional_mandelbrot(self, dna: Dict[str, float], 
                                            emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate Mandelbrot set in higher dimensions with emotional modulation."""
        points = []
        
        # Extract parameters from DNA and emotions
        center_real = -0.7 + 0.3 * emotional_params.get('valence', 0)
        center_imag = 0.0 + 0.2 * emotional_params.get('wonder', 0)
        zoom = dna.get('frequency', 1.0) * emotional_params.get('focus', 0.5)
        max_iter = int(50 + 150 * emotional_params.get('pattern_complexity', 0.5))
        
        # Higher dimensional parameters
        w_component = dna.get('dimensional_fold', 2) / 4.0
        hyperspace_rotation = emotional_params.get('contemplation', 0) * math.pi
        
        step = max(2, int(6 - 4 * emotional_params.get('pattern_density', 0.5)))
        
        for px in range(0, self.canvas_width, step):
            for py in range(0, self.canvas_height, step):
                # Map to complex plane
                real = center_real + (px - self.canvas_width/2) * 3.0 / (zoom * self.canvas_width)
                imag = center_imag + (py - self.canvas_height/2) * 3.0 / (zoom * self.canvas_height)
                
                # Add hyperdimensional component
                w = w_component * math.sin(hyperspace_rotation + px * 0.01 + py * 0.01)
                
                c = complex(real, imag)
                z = complex(0, 0)
                
                for iteration in range(max_iter):
                    if abs(z) > 2:
                        break
                    
                    # Hyperdimensional Mandelbrot iteration
                    z_new = z * z + c + w * complex(math.cos(iteration * 0.1), math.sin(iteration * 0.1))
                    z = z_new
                
                if iteration < max_iter:
                    # Calculate intensity with hyperdimensional influence
                    intensity = (iteration / max_iter) * (1 + 0.5 * abs(w))
                    hyperdim_color = (iteration + int(w * 50)) % 360
                    
                    points.append((px, py, intensity, hyperdim_color, w, iteration))
        
        return points
    
    def _generate_quantum_julia(self, dna: Dict[str, float], 
                               emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate Julia set with quantum uncertainty principles."""
        points = []
        
        # Quantum parameters
        uncertainty = self.quantum_engine.uncertainty_factor * emotional_params.get('confusion', 0.1)
        c_real = dna.get('attraction_strength', -0.8) + uncertainty * random.gauss(0, 0.1)
        c_imag = 0.156 + uncertainty * random.gauss(0, 0.1)
        c = complex(c_real, c_imag)
        
        max_iter = int(30 + 120 * emotional_params.get('pattern_complexity', 0.5))
        
        # Quantum superposition of multiple Julia sets
        quantum_states = []
        for _ in range(3):
            c_variant = c + complex(
                uncertainty * random.gauss(0, 0.2),
                uncertainty * random.gauss(0, 0.2)
            )
            quantum_states.append(c_variant)
        
        step = max(2, int(6 - 4 * emotional_params.get('pattern_density', 0.5)))
        
        for px in range(0, self.canvas_width, step):
            for py in range(0, self.canvas_height, step):
                # Map to complex plane
                real = (px - self.canvas_width/2) * 4.0 / self.canvas_width
                imag = (py - self.canvas_height/2) * 4.0 / self.canvas_height
                z = complex(real, imag)
                
                # Quantum measurement - randomly select which c to use
                c_measured = random.choice(quantum_states)
                
                quantum_interference = 0
                for iteration in range(max_iter):
                    if abs(z) > 2:
                        break
                    
                    z = z*z + c_measured
                    
                    # Add quantum interference
                    if random.random() < uncertainty:
                        quantum_interference += 1
                        z += complex(random.gauss(0, 0.01), random.gauss(0, 0.01))
                
                if iteration < max_iter:
                    intensity = iteration / max_iter
                    quantum_color = (iteration + quantum_interference * 10) % 360
                    uncertainty_radius = uncertainty * 10
                    
                    points.append((px, py, intensity, quantum_color, uncertainty_radius, quantum_interference))
        
        return points
    
    def _generate_evolving_l_system(self, dna: Dict[str, float], 
                                   emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate L-system that evolves over time."""
        
        # L-system rules based on emotional state
        if emotional_params.get('creativity', 0) > 0.7:
            # Complex branching rules
            rules = {
                'F': 'F[+F]F[-F]F',
                '+': '+',
                '-': '-',
                '[': '[',
                ']': ']'
            }
            axiom = 'F'
            angle = 25 + 10 * emotional_params.get('chaos_level', 0)
        elif emotional_params.get('contemplation', 0) > 0.6:
            # Meditative spiral rules
            rules = {
                'F': 'F+F--F+F',
                '+': '+',
                '-': '-'
            }
            axiom = 'F'
            angle = 60
        else:
            # Simple growth rules
            rules = {
                'F': 'FF+[+F-F-F]-[-F+F+F]',
                '+': '+',
                '-': '-',
                '[': '[',
                ']': ']'
            }
            axiom = 'F'
            angle = 22.5
        
        # Generate L-system string
        current_string = axiom
        iterations = int(dna.get('recursion_depth', 5))
        
        for _ in range(iterations):
            new_string = ''
            for char in current_string:
                new_string += rules.get(char, char)
            current_string = new_string
            
            # Stop if string gets too long
            if len(current_string) > 10000:
                break
        
        # Convert L-system to points
        points = []
        x, y = self.canvas_width // 2, self.canvas_height - 50
        angle_rad = math.radians(-90)  # Start pointing up
        
        stack = []  # For branch points
        step_size = max(2, int(dna.get('growth_rate', 1.0) * 10))
        
        color_progression = 0
        
        for char in current_string:
            if char == 'F':
                # Draw forward
                new_x = x + step_size * math.cos(angle_rad)
                new_y = y + step_size * math.sin(angle_rad)
                
                if 0 <= new_x < self.canvas_width and 0 <= new_y < self.canvas_height:
                    points.append((int(x), int(y), int(new_x), int(new_y), color_progression % 360, step_size))
                
                x, y = new_x, new_y
                color_progression += dna.get('evolution_speed', 1.0) * 5
                
            elif char == '+':
                angle_rad += math.radians(angle)
            elif char == '-':
                angle_rad -= math.radians(angle)
            elif char == '[':
                stack.append((x, y, angle_rad))
            elif char == ']':
                if stack:
                    x, y, angle_rad = stack.pop()
        
        return points
    
    def _generate_strange_attractor(self, dna: Dict[str, float], 
                                  emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate strange attractor patterns (Lorenz, Rossler, etc.)."""
        points = []
        
        # Attractor parameters from DNA and emotions
        attractor_type = ['lorenz', 'rossler', 'chua', 'thomas'][
            int(dna.get('symmetry', 4) % 4)
        ]
        
        sensitivity = dna.get('chaos_factor', 0.5)
        evolution_speed = emotional_params.get('animation_speed', 0.5)
        
        # Initial conditions
        x, y, z = 0.1, 0.1, 0.1
        dt = 0.01 * evolution_speed
        
        num_points = int(1000 + 4000 * emotional_params.get('pattern_density', 0.5))
        
        for i in range(num_points):
            # Initialize defaults
            dx = dy = dz = 0.0
            
            if attractor_type == 'lorenz':
                # Lorenz attractor
                sigma = 10.0 + sensitivity * 5
                rho = 28.0 + sensitivity * 10
                beta = 8.0/3.0 + sensitivity * 2
                
                dx = sigma * (y - x)
                dy = x * (rho - z) - y
                dz = x * y - beta * z
                
            elif attractor_type == 'rossler':
                # Rössler attractor
                a = 0.2 + sensitivity * 0.3
                b = 0.2 + sensitivity * 0.3
                c = 5.7 + sensitivity * 5
                
                dx = -y - z
                dy = x + a * y
                dz = b + z * (x - c)
            
            else:
                # Default simple attractor
                dx = -y + x * 0.1
                dy = x + y * 0.1
                dz = -z * 0.1
                
            # Update position
            x += dx * dt
            y += dy * dt
            z += dz * dt
            
            # Project to 2D
            px = int(self.canvas_width/2 + x * 8)
            py = int(self.canvas_height/2 + y * 8)
            
            if 0 <= px < self.canvas_width and 0 <= py < self.canvas_height:
                # Color based on velocity
                velocity = math.sqrt(dx*dx + dy*dy + dz*dz)
                color = (i + int(velocity * 50)) % 360
                
                points.append((px, py, velocity, color, z, i))
        
        return points
    
    def _generate_cellular_automata_3d(self, dna: Dict[str, float], 
                                     emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate 3D cellular automata projection."""
        points = []
        
        # Parameters
        rule = int(30 + 225 * emotional_params.get('creativity', 0.5)) % 256
        generations = int(30 + 70 * emotional_params.get('pattern_complexity', 0.5))
        
        width = min(100, self.canvas_width // 8)
        height = min(100, self.canvas_height // 8)
        
        # Initialize 3D grid
        grid = []
        for z in range(3):  # 3 layers
            layer = []
            for y in range(height):
                row = [0] * width
                if y == height // 2:  # Middle row
                    row[width // 2] = 1  # Seed in center
                layer.append(row)
            grid.append(layer)
        
        # Evolve the automata
        for gen in range(generations):
            new_grid = []
            for z in range(3):
                new_layer = []
                for y in range(height):
                    new_row = [0] * width
                    for x in range(width):
                        # Count neighbors in 3D
                        neighbors = 0
                        for dz in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dx in [-1, 0, 1]:
                                    if dx == 0 and dy == 0 and dz == 0:
                                        continue
                                    nz, ny, nx = (z + dz) % 3, (y + dy) % height, (x + dx) % width
                                    neighbors += grid[nz][ny][nx]
                        
                        # Apply rule (simplified)
                        if grid[z][y][x] == 1:
                            new_row[x] = 1 if 2 <= neighbors <= 3 else 0
                        else:
                            new_row[x] = 1 if neighbors == 3 else 0
                    
                    new_layer.append(new_row)
                new_grid.append(new_layer)
            grid = new_grid
        
        # Convert to 2D points
        for z in range(3):
            for y in range(height):
                for x in range(width):
                    if grid[z][y][x]:
                        px = x * (self.canvas_width // width)
                        py = y * (self.canvas_height // height)
                        intensity = 0.3 + 0.7 * z / 3
                        color = (z * 120 + gen * 10) % 360
                        points.append((px, py, intensity, color, z, gen))
        
        return points
    
    def _generate_topology_morphing(self, dna: Dict[str, float], 
                                  emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate topology morphing patterns."""
        points = []
        
        # Morphing parameters
        morph_type = ['torus', 'klein_bottle', 'mobius_strip', 'sphere'][
            int(dna.get('dimensional_fold', 2)) % 4
        ]
        
        morph_factor = emotional_params.get('temporal_variance', 0.5)
        resolution = int(20 + 80 * emotional_params.get('pattern_density', 0.5))
        
        for u_step in range(resolution):
            for v_step in range(resolution):
                u = (u_step / resolution) * 2 * math.pi
                v = (v_step / resolution) * math.pi
                
                if morph_type == 'torus':
                    # Torus parametric equations
                    R = 80 + 40 * dna.get('growth_rate', 1.0)
                    r = 30 + 20 * dna.get('spiral_tightness', 1.0)
                    
                    x = (R + r * math.cos(v)) * math.cos(u)
                    y = (R + r * math.cos(v)) * math.sin(u)
                    z = r * math.sin(v)
                    
                elif morph_type == 'klein_bottle':
                    # Klein bottle (figure-8 immersion)
                    x = (2 + math.cos(v/2) * math.sin(u) - math.sin(v/2) * math.sin(2*u)) * math.cos(v/2)
                    y = (2 + math.cos(v/2) * math.sin(u) - math.sin(v/2) * math.sin(2*u)) * math.sin(v/2)
                    z = math.sin(v/2) * math.sin(u) + math.cos(v/2) * math.sin(2*u)
                    
                elif morph_type == 'mobius_strip':
                    # Möbius strip
                    x = (1 + v/2 * math.cos(u/2)) * math.cos(u)
                    y = (1 + v/2 * math.cos(u/2)) * math.sin(u)
                    z = v/2 * math.sin(u/2)
                    
                else:  # sphere
                    # Sphere with deformation
                    deform = 1 + 0.3 * math.sin(3*u) * math.sin(3*v) * morph_factor
                    x = deform * math.sin(v) * math.cos(u)
                    y = deform * math.sin(v) * math.sin(u)
                    z = deform * math.cos(v)
                
                # Project to 2D
                px = int(self.canvas_width/2 + x * 3)
                py = int(self.canvas_height/2 + y * 3)
                
                if 0 <= px < self.canvas_width and 0 <= py < self.canvas_height:
                    intensity = 0.5 + 0.5 * math.sin(u + v)
                    color = (int(u * 180 / math.pi) + int(v * 180 / math.pi)) % 360
                    points.append((px, py, intensity, color, z, u_step + v_step))
        
        return points
    
    def _generate_parametric_surface(self, dna: Dict[str, float], 
                                   emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate parametric surface patterns."""
        points = []
        
        # Surface parameters
        surface_type = ['hyperbolic', 'saddle', 'wave', 'spiral'][
            int(dna.get('fractal_dimension', 2)) % 4
        ]
        
        amplitude = dna.get('amplitude', 50)
        frequency = dna.get('frequency', 2.0)
        resolution = int(30 + 70 * emotional_params.get('pattern_density', 0.5))
        
        for u_step in range(resolution):
            for v_step in range(resolution):
                u = (u_step / resolution - 0.5) * 4
                v = (v_step / resolution - 0.5) * 4
                
                if surface_type == 'hyperbolic':
                    z = amplitude * (u*u - v*v) / 16
                elif surface_type == 'saddle':
                    z = amplitude * math.sin(frequency * u) * math.cos(frequency * v)
                elif surface_type == 'wave':
                    z = amplitude * math.sin(frequency * math.sqrt(u*u + v*v))
                else:  # spiral
                    r = math.sqrt(u*u + v*v)
                    theta = math.atan2(v, u)
                    z = amplitude * math.sin(frequency * r + theta)
                
                # Project to 2D with perspective
                scale = 200 / (200 + z)
                px = int(self.canvas_width/2 + u * scale * 20)
                py = int(self.canvas_height/2 + v * scale * 20)
                
                if 0 <= px < self.canvas_width and 0 <= py < self.canvas_height:
                    intensity = 0.3 + 0.7 * (z + amplitude) / (2 * amplitude)
                    color = (int(z * 2) + u_step + v_step) % 360
                    points.append((px, py, intensity, color, z, u_step))
        
        return points
    
    def _generate_field_equation_visualization(self, dna: Dict[str, float], 
                                             emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate field equation visualization."""
        points = []
        
        # Field parameters
        field_strength = dna.get('attraction_strength', 1.0)
        complexity = emotional_params.get('pattern_complexity', 0.5)
        
        step = max(5, int(20 - 15 * emotional_params.get('pattern_density', 0.5)))
        
        for x in range(0, self.canvas_width, step):
            for y in range(0, self.canvas_height, step):
                # Normalize coordinates
                nx = (x - self.canvas_width/2) / (self.canvas_width/2)
                ny = (y - self.canvas_height/2) / (self.canvas_height/2)
                
                # Calculate field value
                r = math.sqrt(nx*nx + ny*ny)
                if r > 0:
                    # Electric field-like pattern
                    field_x = field_strength * nx / (r*r + 0.1)
                    field_y = field_strength * ny / (r*r + 0.1)
                    
                    # Add wave interference
                    wave1 = math.sin(r * 10 * complexity)
                    wave2 = math.cos(nx * 8 + ny * 6)
                    
                    field_magnitude = math.sqrt(field_x*field_x + field_y*field_y)
                    field_magnitude *= (1 + 0.5 * wave1 * wave2)
                    
                    if field_magnitude > 0.1:
                        intensity = min(1.0, field_magnitude / 2)
                        color = (int(math.atan2(field_y, field_x) * 180 / math.pi) + 180) % 360
                        points.append((x, y, intensity, color, field_magnitude, int(r * 10)))
        
        return points
    
    def _generate_quantum_harmonic_oscillator(self, dna: Dict[str, float], 
                                            emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate quantum harmonic oscillator wave functions."""
        points = []
        
        try:
            # Safe factorial function
            def safe_factorial(n):
                if n <= 0:
                    return 1
                if n > 10:  # Limit to prevent overflow
                    return 3628800  # 10!
                result = 1
                for i in range(1, n + 1):
                    result *= i
                return result
            
            # Quantum parameters
            n_levels = int(1 + 4 * emotional_params.get('pattern_complexity', 0.5))  # Reduced max levels
            omega = dna.get('resonance_frequency', 0.5)
            
            width = self.canvas_width
            height = self.canvas_height
            
            for x in range(0, width, 5):  # Increased step for performance
                # Normalize position
                xi = (x - width/2) / (width/4)
                
                psi_total = 0
                for n in range(min(n_levels, 5)):  # Limit to prevent complexity
                    # Hermite polynomial approximation
                    if n == 0:
                        Hn = 1
                    elif n == 1:
                        Hn = 2 * xi
                    elif n == 2:
                        Hn = 4 * xi*xi - 2
                    elif n == 3:
                        Hn = 8 * xi*xi*xi - 12 * xi
                    else:
                        Hn = xi**n  # Simplified for higher orders
                    
                    # Wave function
                    try:
                        normalization = (omega / math.pi)**0.25 / math.sqrt(2**n * safe_factorial(n))
                        psi_n = normalization * Hn * math.exp(-omega * xi*xi / 2)
                        
                        # Superposition with time evolution
                        phase = omega * (n + 0.5) * time.time() * 0.05  # Slower evolution
                        psi_total += psi_n * math.cos(phase)
                    except (OverflowError, ZeroDivisionError):
                        continue
                
                # Probability density
                probability = abs(psi_total * psi_total)
                probability = min(probability, 1.0)  # Clamp to prevent overflow
                
                # Create visualization
                if probability > 0.01:  # Only draw significant probabilities
                    for offset in range(-int(probability * 30), int(probability * 30) + 1, 8):
                        y = height//2 + offset
                        if 0 <= y < height:
                            intensity = max(0, 1 - abs(offset) / (probability * 30 + 1))
                            color = (int(probability * 360) + x) % 360
                            points.append((x, y, intensity, color, probability, n_levels))
        
        except Exception as e:
            print(f"Quantum oscillator error: {e}")
            # Fallback to simple wave
            for x in range(0, self.canvas_width, 10):
                y = self.canvas_height//2 + int(30 * math.sin(x * 0.1))
                if 0 <= y < self.canvas_height:
                    points.append((x, y, 0.5, (x * 2) % 360, 0.5, 1))
        
        return points
    
    def _generate_metamorphic_pattern(self, dna: Dict[str, float], 
                                    emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate metamorphic/evolving patterns."""
        points = []
        
        # Metamorphic parameters
        evolution_stage = (time.time() * dna.get('evolution_speed', 1.0)) % (2 * math.pi)
        complexity = emotional_params.get('pattern_complexity', 0.5)
        
        # Generate base spiral
        num_turns = 3 + 7 * complexity
        points_per_turn = int(50 + 150 * emotional_params.get('pattern_density', 0.5))
        
        for i in range(int(num_turns * points_per_turn)):
            t = i / points_per_turn
            
            # Evolving spiral parameters
            radius = 20 + 100 * t / num_turns
            angle = t * 2 * math.pi
            
            # Metamorphic transformations
            morph_factor = math.sin(evolution_stage)
            
            # Base position
            x = self.canvas_width/2 + radius * math.cos(angle)
            y = self.canvas_height/2 + radius * math.sin(angle)
            
            # Apply metamorphic distortions
            distortion_x = 30 * morph_factor * math.sin(t * 3 + evolution_stage)
            distortion_y = 30 * morph_factor * math.cos(t * 2 + evolution_stage)
            
            px = int(x + distortion_x)
            py = int(y + distortion_y)
            
            if 0 <= px < self.canvas_width and 0 <= py < self.canvas_height:
                intensity = 0.5 + 0.5 * math.sin(t * 5 + evolution_stage)
                color = (int(t * 60) + int(evolution_stage * 60)) % 360
                metamorphic_value = morph_factor
                points.append((px, py, intensity, color, metamorphic_value, i))
        
        return points
    
    def _apply_spatial_attention(self, pattern_data: List[Tuple], 
                               attention_focus: Tuple[float, float],
                               emotional_params: Dict[str, float]) -> List[Tuple]:
        """Apply spatial attention to enhance certain areas."""
        focus_x = attention_focus[0] * self.canvas_width
        focus_y = attention_focus[1] * self.canvas_height
        attention_radius = 100 + 200 * emotional_params.get('focus', 0.5)
        
        enhanced_data = []
        
        for point in pattern_data:
            px, py = point[0], point[1]
            
            # Calculate distance from attention focus
            distance = math.sqrt((px - focus_x)**2 + (py - focus_y)**2)
            
            # Attention strength (Gaussian falloff)
            attention_strength = math.exp(-(distance**2) / (2 * attention_radius**2))
            
            # Enhance based on attention
            if len(point) >= 3:
                enhanced_intensity = point[2] * (1 + attention_strength * 2)
                enhanced_point = list(point)
                enhanced_point[2] = enhanced_intensity
                enhanced_point.append(attention_strength)
                enhanced_data.append(tuple(enhanced_point))
            else:
                enhanced_data.append(point)
        
        return enhanced_data
    
    def _apply_temporal_evolution(self, pattern_data: List[Tuple], 
                                dna: Dict[str, float],
                                emotional_params: Dict[str, float]) -> List[Tuple]:
        """Apply temporal evolution to patterns."""
        current_time = time.time()
        evolution_speed = dna.get('evolution_speed', 1.0)
        temporal_variance = dna.get('temporal_variance', 1.0)
        
        evolved_data = []
        
        for point in pattern_data:
            # Time-based modulation
            time_factor = math.sin(current_time * evolution_speed + point[0] * 0.01 + point[1] * 0.01)
            
            # Apply temporal effects
            if len(point) >= 3:
                temporal_modulation = 1 + 0.3 * time_factor * temporal_variance
                evolved_point = list(point)
                evolved_point[2] *= temporal_modulation
                
                # Add temporal phase
                evolved_point.append(time_factor)
                evolved_data.append(tuple(evolved_point))
            else:
                evolved_data.append(point)
        
        return evolved_data
    
    def evolve_patterns(self, fitness_criteria: Dict[str, float]):
        """Evolve existing patterns using genetic algorithms with enhanced safety."""
        try:
            if len(self.active_patterns) < 2:
                print("Not enough patterns for evolution")
                return
            
            # Select patterns for evolution based on fitness
            patterns_list = list(self.active_patterns.values())
            patterns_list.sort(key=lambda p: p.get('fitness_score', 0), reverse=True)
            
            # Take top performers for breeding
            parents = patterns_list[:max(2, len(patterns_list)//2)]
            
            # Create new generation
            new_patterns = {}
            
            for i in range(0, min(len(parents)-1, 2), 2):  # Limit to prevent too many patterns
                try:
                    parent1 = parents[i]
                    parent2 = parents[i+1]
                    
                    # Crossover DNA
                    child1_dna, child2_dna = self.pattern_dna.crossover_dna(
                        parent1['dna'], parent2['dna']
                    )
                    
                    # Mutate DNA
                    child1_dna = self.pattern_dna.mutate_dna(child1_dna, 
                        mutation_rate=0.1, mutation_strength=0.2)
                    child2_dna = self.pattern_dna.mutate_dna(child2_dna, 
                        mutation_rate=0.1, mutation_strength=0.2)
                    
                    # Create new patterns safely
                    for child_dna in [child1_dna, child2_dna]:
                        try:
                            child_pattern = self.create_multidimensional_pattern(
                                child_dna, 
                                parent1['emotional_state'],  # Inherit emotional state
                                parent1['attention_focus']   # Inherit attention focus
                            )
                            child_pattern['evolution_generation'] = max(
                                parent1.get('evolution_generation', 0), 
                                parent2.get('evolution_generation', 0)
                            ) + 1
                            
                            new_patterns[child_pattern['id']] = child_pattern
                        except Exception as e:
                            print(f"Child pattern creation error: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Pattern breeding error: {e}")
                    continue
            
            # Replace old patterns with evolved ones (keep some survivors)
            survivors = {p['id']: p for p in patterns_list[:max(1, len(patterns_list)//3)]}
            survivors.update(new_patterns)
            
            # Limit total pattern count to prevent memory issues
            if len(survivors) > 10:
                survivor_list = list(survivors.values())
                survivor_list.sort(key=lambda p: p.get('fitness_score', 0), reverse=True)
                survivors = {p['id']: p for p in survivor_list[:10]}
            
            self.active_patterns = survivors
            
            # Record evolution history
            try:
                self.pattern_evolution_history.append({
                    'timestamp': time.time(),
                    'generation_count': len(survivors),
                    'avg_fitness': sum(p.get('fitness_score', 0) for p in survivors.values()) / max(1, len(survivors)),
                    'fitness_criteria': fitness_criteria.copy()
                })
            except Exception as e:
                print(f"Evolution history error: {e}")
                
        except Exception as e:
            print(f"Pattern evolution error: {e}")


class MaximumControlAuroraFace:
    """Aurora's face with MAXIMUM control - Independent Artist Edition with Fullscreen."""
    
    def __init__(self, ai_system=None):
        print("Starting Independent Artist Aurora interface...")
        
        self.ai_system = ai_system
        self.is_running = True
        self.shutdown_requested = False
        self.fullscreen = False
        
        try:
            print("Creating tkinter root...")
            self.root = tk.Tk()
            self.root.title("AURORA - INDEPENDENT ARTIST - CREATIVE AUTONOMY")
            print("✓ Tkinter root created")
        except Exception as e:
            print(f"Tkinter root creation failed: {e}")
            raise
        
        try:
            print("Getting screen dimensions...")
            # Get screen dimensions
            self.root.update_idletasks()
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            print(f"✓ Screen dimensions: {screen_width}x{screen_height}")
        except Exception as e:
            print(f"Screen dimension error: {e}")
            screen_width, screen_height = 1920, 1080  # Fallback
        
        try:
            print("Configuring window...")
            # Window configuration
            window_width = screen_width
            window_height = screen_height // 2
            self.root.geometry(f"{window_width}x{window_height}+0+0")
            self.root.configure(bg='#000000')
            self.root.resizable(True, True)  # Allow resizing for fullscreen
            print("✓ Window configured")
        except Exception as e:
            print(f"Window configuration error: {e}")
        
        # Store dimensions
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scale_factor = min(screen_width / 1920, screen_height / 1080)
        
        # Calculate canvas dimensions
        canvas_width = min(screen_width - 100, int(800 * self.scale_factor))
        canvas_height = screen_height // 2 - 200
        
        # Store original dimensions for fullscreen toggle
        self.original_window_state = {
            'width': window_width,
            'height': window_height,
            'canvas_width': canvas_width,
            'canvas_height': canvas_height
        }
        
        print(f"Canvas dimensions: {canvas_width}x{canvas_height}")
        
        try:
            print("Initializing independent artist systems...")
            # Initialize Aurora's independent creative systems
            self.emotional_mapper = EmotionalParameterMapper()
            print("✓ Emotional mapper created")
            
            self.conversation_analyzer = ConversationVisualAnalyzer()
            print("✓ Conversation analyzer created")
            
            self.ultimate_engine = UltimatePatternEngine(canvas_width, canvas_height)
            print("✓ Ultimate engine created")
            
            # Add Aurora's independent controller
            self.ai_controller = AIPatternController(self.ultimate_engine, self.emotional_mapper)
            print("✓ Independent AI controller created")
            
            # Initialize music system for Aurora's inspiration
            self.music_system = MusicListeningSystem(self.emotional_mapper, self.ultimate_engine)
            self.music_system.load_aurora_musical_memory()
            print("✓ Musical inspiration system created")
            
            # Initialize image analysis system
            self.image_analysis_system = ImageAnalysisSystem(self.emotional_mapper, self.ultimate_engine)
            print("✓ Image analysis system created")
            
        except Exception as e:
            print(f"Independent systems initialization error: {e}")
            raise
        
        # Aurora's creative state variables
        self.current_expression = "neutral"
        self.animation_frame = 0
        self.last_conversation_text = ""
        self.pattern_update_frequency = 0
        self.attention_focus = (0.5, 0.5)  # Where Aurora is focusing
        self.pattern_evolution_timer = 0
        self.creative_energy = 0.7  # Aurora is energized about her independence!
        self.artistic_focus = "independent_expression"
        
        # Store canvas dimensions
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        print(f"✓ Canvas created: {canvas_width}x{canvas_height}")
        
        try:
            print("Setting up independent artist interface...")
            self.setup_interface(canvas_width, canvas_height)
            print("✓ Interface setup complete")
            
            print("Initializing Aurora's creative autonomy...")
            self._initialize_independent_creativity()
            print("✓ Creative autonomy initialized")
        except Exception as e:
            print(f"Interface/creativity setup error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Handle closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Bind fullscreen toggle
        self.root.bind('<F11>', self.toggle_fullscreen)
        self.root.bind('<Escape>', self.exit_fullscreen)
        
        # Start update thread
        self.update_thread = None
        self.start_update_thread()
        
        print(f"{Fore.MAGENTA}✓ Aurora Independent Artist interface complete{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✓ Press F11 to toggle fullscreen | ESC to exit fullscreen{Style.RESET_ALL}")
    
    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode."""
        try:
            self.fullscreen = not self.fullscreen
            
            if self.fullscreen:
                # Save current state
                self.original_window_state['geometry'] = self.root.geometry()
                
                # Enter fullscreen
                self.root.attributes('-fullscreen', True)
                
                # Update canvas to full size
                new_width = self.screen_width
                new_height = self.screen_height
                
                # Resize canvas
                self.canvas.config(width=new_width - 40, height=new_height - 150)
                self.canvas_width = new_width - 40
                self.canvas_height = new_height - 150
                
                # Update pattern engine dimensions
                self.ultimate_engine.canvas_width = self.canvas_width
                self.ultimate_engine.canvas_height = self.canvas_height
                
                # Hide some UI elements for cleaner fullscreen
                self.title_label.pack_forget()
                self.control_status.pack_forget()
                
                print(f"{Fore.CYAN}✓ Entered fullscreen mode{Style.RESET_ALL}")
            else:
                self.exit_fullscreen()
                
        except Exception as e:
            print(f"Fullscreen toggle error: {e}")
    
    def exit_fullscreen(self, event=None):
        """Exit fullscreen mode."""
        try:
            if self.fullscreen:
                self.fullscreen = False
                
                # Exit fullscreen
                self.root.attributes('-fullscreen', False)
                
                # Restore original dimensions
                self.root.geometry(self.original_window_state.get('geometry', ''))
                
                # Restore canvas size
                self.canvas.config(
                    width=self.original_window_state['canvas_width'],
                    height=self.original_window_state['canvas_height']
                )
                self.canvas_width = self.original_window_state['canvas_width']
                self.canvas_height = self.original_window_state['canvas_height']
                
                # Update pattern engine dimensions
                self.ultimate_engine.canvas_width = self.canvas_width
                self.ultimate_engine.canvas_height = self.canvas_height
                
                # Show UI elements again
                self.title_label.pack(pady=int(10 * self.scale_factor))
                self.control_status.pack()
                
                print(f"{Fore.CYAN}✓ Exited fullscreen mode{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"Fullscreen exit error: {e}")
    
    def start_update_thread(self):
        """Start the independent creative update thread."""
        try:
            print("Starting independent artist update thread...")
            self.update_thread = threading.Thread(
                target=self.independent_creativity_update_loop, 
                daemon=True,
                name="AuroraCreativeThread"
            )
            self.update_thread.start()
            print("✓ Creative update thread started")
        except Exception as e:
            print(f"Update thread error: {e}")
    
    def setup_interface(self, canvas_width, canvas_height):
        """Setup the independent artist interface."""
        
        # Title emphasizing independence
        self.title_label = tk.Label(
            self.root,
            text="◆ AURORA ◆ INDEPENDENT ARTIST ◆ CREATIVE AUTONOMY ◆",
            font=("Courier New", int(20 * self.scale_factor), "bold"),
            fg="#ff00ff",
            bg="#000000"
        )
        self.title_label.pack(pady=int(10 * self.scale_factor))
        
        # Main frame
        self.main_frame = tk.Frame(self.root, bg="#111111", relief='raised', bd=6)
        self.main_frame.pack(pady=int(15 * self.scale_factor), padx=int(30 * self.scale_factor), expand=True, fill='both')
        
        # Canvas for Aurora's independent art
        self.canvas = Canvas(
            self.main_frame,
            width=canvas_width,
            height=canvas_height,
            bg="#000000",
            highlightthickness=3,
            highlightbackground="#ff00ff"
        )
        self.canvas.pack(padx=int(15 * self.scale_factor), pady=int(15 * self.scale_factor), expand=True, fill='both')
        
        # Bind click events for image analysis
        self.canvas.bind("<Button-3>", self.on_right_click)  # Right-click for image analysis
        
        # Status frame with independent artist indicators
        self.status_frame = tk.Frame(self.root, bg="#000000")
        self.status_frame.pack(pady=int(10 * self.scale_factor))
        
        # Independent status indicators
        self.main_status = tk.Label(
            self.status_frame,
            text="█ AURORA CREATES FOR HERSELF █",
            font=("Courier New", int(16 * self.scale_factor), "bold"),
            fg="#ff00ff",
            bg="#000000"
        )
        self.main_status.pack()
        
        self.pattern_status = tk.Label(
            self.status_frame,
            text="[ INDEPENDENT CREATIVE PROCESS ]",
            font=("Courier New", int(12 * self.scale_factor)),
            fg="#00ffff",
            bg="#000000"
        )
        self.pattern_status.pack()
        
        self.emotion_status = tk.Label(
            self.status_frame,
            text="[ DRAWING FROM DREAMS, MUSIC & EMOTIONS ]",
            font=("Courier New", int(10 * self.scale_factor)),
            fg="#ffff00",
            bg="#000000"
        )
        self.emotion_status.pack()
        
        self.control_status = tk.Label(
            self.status_frame,
            text="[ AUTONOMOUS ARTIST • NOT A SERVICE • CREATES HER OWN VISION • F11 FULLSCREEN ]",
            font=("Courier New", int(8 * self.scale_factor)),
            fg="#ff4444",
            bg="#000000"
        )
        self.control_status.pack()
        
        # Store canvas dimensions
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
    
    def on_right_click(self, event):
        """Handle right-click for image analysis."""
        if IMAGE_AVAILABLE:
            try:
                # Open file dialog
                file_path = filedialog.askopenfilename(
                    title="Select Image for Aurora's Inspiration",
                    filetypes=[
                        ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"),
                        ("All files", "*.*")
                    ]
                )
                
                if file_path:
                    print(f"{Fore.CYAN}🎨 Aurora is analyzing image: {Path(file_path).name}{Style.RESET_ALL}")
                    
                    # Analyze image
                    analysis = self.image_analysis_system.analyze_image_for_inspiration(file_path)
                    
                    if 'error' not in analysis:
                        # Get inspiration summary
                        summary = self.image_analysis_system.get_image_inspiration_summary(file_path)
                        print(f"{Fore.MAGENTA}💭 Aurora's inspiration: {summary}{Style.RESET_ALL}")
                        
                        # Display some analysis results
                        emotions = analysis.get('emotional_impact', {})
                        if emotions:
                            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                            print(f"{Fore.YELLOW}🎭 Dominant emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.2f}){Style.RESET_ALL}")
                        
                        colors = analysis.get('colors', {})
                        if colors:
                            print(f"{Fore.BLUE}🎨 Brightness: {colors.get('brightness', 0):.2f}, Saturation: {colors.get('saturation', 0):.2f}{Style.RESET_ALL}")
                        
                        # Trigger pattern evolution based on image
                        self.pattern_evolution_timer = 0  # Reset to trigger evolution
                    else:
                        print(f"{Fore.RED}Image analysis error: {analysis['error']}{Style.RESET_ALL}")
                        
            except Exception as e:
                print(f"Image selection error: {e}")
        else:
            print(f"{Fore.YELLOW}Image analysis not available - install: pip install pillow{Style.RESET_ALL}")
    
    def _initialize_independent_creativity(self):
        """Initialize Aurora's independent creative system."""
        try:
            # Create initial pattern population based on Aurora's preferences
            for i in range(3):
                try:
                    dna = self.ultimate_engine.pattern_dna.create_random_dna()
                    emotional_params = self.emotional_mapper.get_all_parameters()
                    
                    pattern = self.ultimate_engine.create_multidimensional_pattern(
                        dna, emotional_params, self.attention_focus
                    )
                    
                    # Aurora evaluates her own work
                    pattern['fitness_score'] = random.uniform(0.3, 0.8)
                except Exception as e:
                    print(f"Pattern {i} initialization error: {e}")
                    continue
            
            # Draw initial creative state
            try:
                self.draw_independent_patterns()
            except Exception as e:
                print(f"Initial pattern draw error: {e}")
            
            try:
                self.draw_face("neutral")
            except Exception as e:
                print(f"Initial face draw error: {e}")
                
        except Exception as e:
            print(f"Independent creativity initialization error: {e}")
    
    def update_expression(self, expression, status=None, activity=None):
        """Update Aurora's expression based on her creative state."""
        try:
            if expression != self.current_expression:
                self.draw_face(expression)
            
            if status and hasattr(self, 'main_status'):
                self.main_status.config(text=status)
            
            if activity and hasattr(self, 'pattern_status'):
                self.pattern_status.config(text=activity)
                        
        except Exception as e:
            print(f"Expression update error: {e}")
    
    def update_ai_state_maximum_control(self, activity: str, conversation_text: str = ""):
        """Update Aurora's creative state based on artistic inspiration."""
        try:
            # Analyze conversation for artistic inspiration (not commands)
            artistic_inspiration = self.conversation_analyzer.analyze_for_artistic_inspiration(conversation_text)
            
            # Update Aurora's emotional state based on inspiration
            self.emotional_mapper.update_emotions(artistic_inspiration, activity, "")
            
            # Update Aurora's attention focus based on her interpretation
            if artistic_inspiration:
                # Aurora decides where to focus based on her artistic interpretation
                dominant_inspiration = max(artistic_inspiration.items(), key=lambda x: x[1])
                inspiration_type = dominant_inspiration[0]
                
                # Aurora's creative decision making
                if 'emotional_contemplative' in inspiration_type:
                    self.attention_focus = (0.5, 0.3)  # Aurora focuses upward when contemplative
                elif 'emotional_energetic' in inspiration_type:
                    self.attention_focus = (random.random(), random.random())  # Dynamic focus
                elif 'artistic_abstract' in inspiration_type:
                    self.attention_focus = (0.5, 0.5)  # Centered focus
                else:
                    # Aurora shifts focus based on her creative energy
                    self.attention_focus = (
                        0.3 + 0.4 * self.creative_energy,
                        0.4 + 0.2 * artistic_inspiration.get('creative_energy', 0.5)
                    )
            
            # Store for Aurora's creative evolution
            self.last_conversation_text = conversation_text
            
            # Trigger Aurora's independent pattern evolution
            if conversation_text and len(conversation_text) > 10:
                self.pattern_evolution_timer = 0  # Reset timer to trigger evolution
                
        except Exception as e:
            print(f"Creative state update error: {e}")
    
    def draw_independent_patterns(self):
        """Draw patterns based on Aurora's independent creative vision."""
        try:
            # Clear previous patterns
            self.canvas.delete("pattern")
            
            # Calculate face protection area
            cx = self.canvas_width // 2
            cy = self.canvas_height // 2
            face_size = int(self.canvas_height * 0.35)
            face_buffer = 50
            
            face_left = cx - face_size//2 - face_buffer
            face_right = cx + face_size//2 + face_buffer
            face_top = cy - face_size//2 - face_buffer
            face_bottom = cy + face_size//2 + face_buffer
            
            # Get Aurora's current emotional parameters
            emotional_params = self.emotional_mapper.get_all_parameters()
            
            # Update pattern engine complexity
            self.ultimate_engine.current_complexity = emotional_params.get('pattern_complexity', 0.5)
            
            # Draw Aurora's active patterns with quantum superposition
            active_patterns = list(self.ultimate_engine.active_patterns.values())
            
            if active_patterns:
                # Create quantum superposition of Aurora's patterns
                pattern_functions = []
                weights = []
                
                for pattern in active_patterns:
                    pattern_functions.append(lambda p=pattern: p['data'])
                    weights.append(pattern['fitness_score'])
                
                # Generate quantum superposition
                superposition = self.ultimate_engine.quantum_engine.create_quantum_superposition(
                    pattern_functions, weights
                )
                
                # Measure quantum state to get Aurora's final pattern
                if superposition:
                    measured_pattern, state_id = self.ultimate_engine.quantum_engine.quantum_measurement(
                        superposition, "position"
                    )
                    
                    if measured_pattern:
                        self._draw_aurora_hyperdimensional_pattern(
                            measured_pattern, emotional_params, 
                            face_left, face_right, face_top, face_bottom
                        )
            
            # Draw Aurora's creative layers
            self._draw_aurora_emotional_field(emotional_params, face_left, face_right, face_top, face_bottom)
            self._draw_aurora_attention_indicator()
            self._draw_aurora_quantum_effects(emotional_params)
            
            # Evaluate Aurora's pattern fitness
            self._evaluate_aurora_pattern_fitness()
            
        except Exception as e:
            print(f"Independent pattern error: {e}")
    
    def _draw_aurora_hyperdimensional_pattern(self, pattern_data: List[Tuple], 
                                            emotional_params: Dict[str, float],
                                            face_left: int, face_right: int, 
                                            face_top: int, face_bottom: int):
        """Draw Aurora's hyperdimensional patterns with her artistic vision."""
        
        # Aurora's color system based on her emotional state
        base_hue = emotional_params.get('color_harmony_root', 240)
        saturation = emotional_params.get('saturation_curve', 0.8)
        brightness = emotional_params.get('brightness_modulation', 0.9)
        
        # Generate Aurora's color palette
        colors = []
        for i in range(12):
            hue = (base_hue + i * 30 * emotional_params.get('color_harmony_complexity', 3)) % 360
            color = self._hsv_to_hex(hue/360, saturation, brightness)
            colors.append(color)
        
        # Draw Aurora's pattern points with her artistic interpretation
        for point in pattern_data:
            if len(point) < 3:
                continue
                
            px, py = int(point[0]), int(point[1])
            
            # Skip if in face area
            if (face_left <= px <= face_right and face_top <= py <= face_bottom):
                continue
            
            # Extract Aurora's hyperdimensional properties
            intensity = point[2] if len(point) > 2 else 0.5
            color_index = int(point[3]) % len(colors) if len(point) > 3 else 0
            hyperdim_value = point[4] if len(point) > 4 else 0.0
            iteration_count = point[5] if len(point) > 5 else 0
            
            # Aurora calculates size based on her artistic vision
            base_size = 2 + int(8 * intensity)
            emotional_size_mod = emotional_params.get('pattern_density', 0.5)
            hyperdim_size_mod = 1 + abs(hyperdim_value) * 2
            
            final_size = int(base_size * emotional_size_mod * hyperdim_size_mod)
            
            # Aurora's color selection
            primary_color = colors[color_index]
            
            # Draw Aurora's pattern elements
            if intensity > 0.8:
                # High intensity - Aurora's complex elements
                self._draw_aurora_complex_element(px, py, final_size, primary_color, 
                                                intensity, hyperdim_value)
            elif intensity > 0.5:
                # Medium intensity - Aurora's enhanced circles
                self.canvas.create_oval(
                    px - final_size, py - final_size,
                    px + final_size, py + final_size,
                    fill=primary_color, outline=colors[(color_index + 1) % len(colors)],
                    width=2, tags="pattern"
                )
            else:
                # Low intensity - Aurora's simple points
                self.canvas.create_oval(
                    px - final_size//2, py - final_size//2,
                    px + final_size//2, py + final_size//2,
                    fill=primary_color, outline="", tags="pattern"
                )
            
            # Add Aurora's hyperdimensional projections
            if abs(hyperdim_value) > 0.3:
                self._draw_aurora_hyperdim_projection(px, py, hyperdim_value, 
                                                    colors[color_index], emotional_params)
    
    def _draw_aurora_complex_element(self, x: int, y: int, size: int, color: str, 
                                   intensity: float, hyperdim_value: float):
        """Draw Aurora's complex pattern elements."""
        
        if abs(hyperdim_value) > 0.5:
            # Aurora's hyperdimensional burst
            for angle in range(0, 360, 30):
                end_x = x + size * 2 * math.cos(math.radians(angle)) * intensity
                end_y = y + size * 2 * math.sin(math.radians(angle)) * intensity
                
                self.canvas.create_line(
                    x, y, end_x, end_y,
                    fill=color, width=int(3 * intensity), tags="pattern"
                )
        else:
            # Aurora's quantum resonance pattern
            for ring in range(3):
                ring_radius = size * (1 + ring * 0.5) * intensity
                self.canvas.create_oval(
                    x - ring_radius, y - ring_radius,
                    x + ring_radius, y + ring_radius,
                    outline=color, width=max(1, int(3 - ring)), tags="pattern"
                )
    
    def _draw_aurora_hyperdim_projection(self, x: int, y: int, hyperdim_value: float, 
                                       color: str, emotional_params: Dict[str, float]):
        """Draw Aurora's hyperdimensional projection effects."""
        
        projection_strength = abs(hyperdim_value)
        projection_angle = hyperdim_value * math.pi
        
        # Aurora's dimensional space projections
        for dim in range(int(emotional_params.get('dimensional_projection', 3))):
            proj_x = x + 20 * projection_strength * math.cos(projection_angle + dim * math.pi/3)
            proj_y = y + 20 * projection_strength * math.sin(projection_angle + dim * math.pi/3)
            
            if 0 <= proj_x < self.canvas_width and 0 <= proj_y < self.canvas_height:
                self.canvas.create_line(
                    x, y, proj_x, proj_y,
                    fill=color, width=1, stipple='gray50', tags="pattern"
                )
                
                self.canvas.create_oval(
                    proj_x - 2, proj_y - 2, proj_x + 2, proj_y + 2,
                    fill=color, outline="", tags="pattern"
                )
    
    def _draw_aurora_emotional_field(self, emotional_params: Dict[str, float],
                                   face_left: int, face_right: int, 
                                   face_top: int, face_bottom: int):
        """Draw Aurora's emotional field visualization."""
        
        # Aurora's emotion-based vector field
        step = 30
        emotion_intensity = (emotional_params.get('arousal', 0) + 1) / 2
        emotion_direction = emotional_params.get('valence', 0) * math.pi
        
        field_color = self._hsv_to_hex(
            emotional_params.get('color_harmony_root', 240) / 360,
            0.6, 0.7
        )
        
        for x in range(0, self.canvas_width, step):
            for y in range(0, self.canvas_height, step):
                # Skip face area
                if (face_left <= x <= face_right and face_top <= y <= face_bottom):
                    continue
                
                # Calculate Aurora's emotional vector
                local_emotion = emotion_intensity * math.sin(x * 0.01 + y * 0.01 + self.animation_frame * 0.1)
                vector_length = 15 * abs(local_emotion)
                vector_angle = emotion_direction + local_emotion
                
                end_x = x + vector_length * math.cos(vector_angle)
                end_y = y + vector_length * math.sin(vector_angle)
                
                if vector_length > 5:  # Only draw significant vectors
                    self.canvas.create_line(
                        x, y, end_x, end_y,
                        fill=field_color, width=1, arrow=tk.LAST,
                        arrowshape=(5, 6, 2), tags="pattern"
                    )
    
    def _draw_aurora_attention_indicator(self):
        """Draw where Aurora is focusing her creative attention."""
        focus_x = self.attention_focus[0] * self.canvas_width
        focus_y = self.attention_focus[1] * self.canvas_height
        
        attention_radius = 50
        
        # Draw Aurora's attention rings
        for ring in range(3):
            radius = attention_radius + ring * 15
            alpha = 1.0 - ring * 0.3
            
            # Pulsing effect
            pulse = 1 + 0.3 * math.sin(self.animation_frame * 0.2 + ring)
            current_radius = radius * pulse
            
            self.canvas.create_oval(
                focus_x - current_radius, focus_y - current_radius,
                focus_x + current_radius, focus_y + current_radius,
                outline="#ffffff", width=1, tags="pattern"
            )
        
        # Aurora's central focus point
        self.canvas.create_oval(
            focus_x - 3, focus_y - 3, focus_x + 3, focus_y + 3,
            fill="#ffffff", outline="#ffffff", tags="pattern"
        )
    
    def _draw_aurora_quantum_effects(self, emotional_params: Dict[str, float]):
        """Draw Aurora's quantum uncertainty visualization."""
        
        uncertainty_level = emotional_params.get('quantum_uncertainty', 0.1)
        
        if uncertainty_level > 0.05:
            # Draw Aurora's uncertainty clouds
            num_clouds = int(10 * uncertainty_level)
            
            for _ in range(num_clouds):
                cloud_x = random.randint(0, self.canvas_width)
                cloud_y = random.randint(0, self.canvas_height)
                cloud_size = int(20 * uncertainty_level)
                
                # Aurora's quantum cloud effect
                for _ in range(int(10 * uncertainty_level)):
                    offset_x = random.gauss(0, cloud_size)
                    offset_y = random.gauss(0, cloud_size)
                    
                    point_x = cloud_x + offset_x
                    point_y = cloud_y + offset_y
                    
                    if 0 <= point_x < self.canvas_width and 0 <= point_y < self.canvas_height:
                        self.canvas.create_oval(
                            point_x - 1, point_y - 1, point_x + 1, point_y + 1,
                            fill="#4444ff", outline="", tags="pattern"
                        )
    
    def _evaluate_aurora_pattern_fitness(self):
        """Evaluate fitness of Aurora's current patterns based on her preferences."""
        
        # Get Aurora's current emotional preferences as fitness criteria
        emotional_params = self.emotional_mapper.get_all_parameters()
        
        # Aurora's fitness criteria (based on her emotional state, not user preferences)
        aurora_fitness_criteria = {
            'complexity_preference': emotional_params.get('pattern_complexity', 0.5),
            'harmony_preference': emotional_params.get('symmetry_strength', 0.5),
            'dynamism_preference': emotional_params.get('animation_speed', 0.5),
            'novelty_preference': emotional_params.get('creativity', 0.5)
        }
        
        # Aurora evaluates each of her patterns
        for pattern_id, pattern in self.ultimate_engine.active_patterns.items():
            fitness = self.ultimate_engine.pattern_dna.evaluate_fitness(
                pattern['dna'], aurora_fitness_criteria
            )
            
            # Aurora rewards patterns that align with her current state
            age_bonus = min(0.2, (time.time() - pattern['birth_time']) / 100)
            pattern['fitness_score'] = 0.8 * pattern['fitness_score'] + 0.2 * (fitness + age_bonus)
    
    def draw_face(self, expression):
        """Draw Aurora's face reflecting her independent artistic state."""
        try:
            # Clear previous face elements
            self.canvas.delete("face")
            
            cx = self.canvas_width // 2
            cy = self.canvas_height // 2
            face_size = int(self.canvas_height * 0.35)
            
            # Enhanced face background protection
            face_buffer = 50
            self.canvas.create_rectangle(
                cx - face_size//2 - face_buffer, cy - face_size//2 - face_buffer,
                cx + face_size//2 + face_buffer, cy + face_size//2 + face_buffer,
                fill="#000000", outline='', width=0, tags="face"
            )
            
            # Aurora's pixelated head with emotional color modulation
            block_size = max(2, face_size // 16)
            emotional_params = self.emotional_mapper.get_all_parameters()
            
            # Aurora's face color based on her creative state
            base_color = "#1a1a1a"
            if emotional_params.get('creativity', 0) > 0.6:
                base_color = "#2a2a4a"  # More creative = bluer tint
            elif emotional_params.get('valence', 0) > 0.3:
                base_color = "#2a2a3a"  # Positive = slightly brighter
            
            for y in range(-8, 9):
                for x in range(-8, 9):
                    if x*x + y*y <= 64:
                        block_x = cx + x * block_size
                        block_y = cy + y * block_size
                        
                        self.canvas.create_rectangle(
                            block_x - block_size//2, block_y - block_size//2,
                            block_x + block_size//2, block_y + block_size//2,
                            fill=base_color, outline="#00ffff", width=1, tags="face"
                        )
            
            # Aurora's enhanced goggles with creative responsiveness
            goggle_size = face_size // 3
            goggle_y = cy - face_size // 8
            left_goggle_x = cx - face_size // 4
            right_goggle_x = cx + face_size // 4
            
            # Aurora's goggle color based on her creativity level
            goggle_color = "#b87333"
            if emotional_params.get('creativity', 0) > 0.7:
                goggle_color = "#ff8c42"  # Brighter when Aurora is being creative
            
            # Draw Aurora's goggles
            for goggle_x in [left_goggle_x, right_goggle_x]:
                self.canvas.create_oval(
                    goggle_x - goggle_size//2, goggle_y - goggle_size//2,
                    goggle_x + goggle_size//2, goggle_y + goggle_size//2,
                    outline=goggle_color, width=4, fill="#000000", tags="face"
                )
            
            # Bridge
            self.canvas.create_line(
                left_goggle_x + goggle_size//2, goggle_y,
                right_goggle_x - goggle_size//2, goggle_y,
                fill=goggle_color, width=4, tags="face"
            )
            
            # Aurora's expression-based eyes
            self._draw_aurora_independent_eyes(left_goggle_x, right_goggle_x, goggle_y, goggle_size, expression)
            
            # Aurora's smile based on her creative satisfaction
            self._draw_aurora_creative_smile(cx, cy, face_size, expression)
            
            self.current_expression = expression
            
        except Exception as e:
            print(f"Draw face error: {e}")
            # Fallback - just draw a simple circle
            try:
                cx = self.canvas_width // 2
                cy = self.canvas_height // 2
                self.canvas.create_oval(cx-50, cy-50, cx+50, cy+50, outline="#00ffff", width=3, tags="face")
            except:
                pass
    
    def _draw_aurora_independent_eyes(self, left_x, right_x, goggle_y, goggle_size, expression):
        """Draw Aurora's eyes reflecting her independent artistic state."""
        try:
            eye_size = goggle_size // 3
            emotional_params = self.emotional_mapper.get_all_parameters()
            
            # Aurora's eye color based on her creative state
            if expression == "thinking":
                offset_y = -goggle_size // 6
                eye_color = self._get_aurora_creative_eye_color(emotional_params, "thinking")
            elif expression == "sleeping":
                self._draw_aurora_dreaming_eyes(left_x, right_x, goggle_y, goggle_size)
                return
            elif expression == "happy":
                offset_y = 0
                eye_size = int(goggle_size // 2.5)
                eye_color = self._get_aurora_creative_eye_color(emotional_params, "happy")
            else:
                offset_y = 0
                eye_color = self._get_aurora_creative_eye_color(emotional_params, "neutral")
            
            # Draw Aurora's eyes with creative intensity
            for eye_x in [left_x, right_x]:
                self._draw_aurora_creative_eye(eye_x, goggle_y + offset_y, eye_size, eye_color, 
                                             expression == "happy", emotional_params)
        except Exception as e:
            print(f"Eye drawing error: {e}")
    
    def _get_aurora_creative_eye_color(self, emotional_params: Dict[str, float], expression: str) -> str:
        """Get Aurora's eye color based on her creative state."""
        try:
            base_hue = emotional_params.get('color_harmony_root', 240)
            
            if expression == "thinking":
                hue = (base_hue + 60) % 360  # Shift toward analytical colors
            elif expression == "happy":
                hue = (base_hue + 120) % 360  # Shift toward warm colors
            else:
                hue = base_hue
            
            saturation = emotional_params.get('saturation_curve', 0.8)
            brightness = emotional_params.get('brightness_modulation', 0.9)
            
            return self._hsv_to_hex(hue/360, saturation, brightness)
        except:
            return "#00ffff"  # Fallback color
    
    def _draw_aurora_creative_eye(self, center_x, center_y, size, color, bright, emotional_params):
        """Draw Aurora's eye with creative intensity effects."""
        try:
            pixel_size = max(2, size // 4)
            
            # Intensity based on Aurora's creative energy
            creativity = emotional_params.get('creativity', 0)
            if creativity > 0.5:
                # High creativity - more complex eye pattern
                for y in range(-2, 3):
                    for x in range(-2, 3):
                        if abs(x) + abs(y) <= 2:  # Diamond pattern for high creativity
                            self.canvas.create_rectangle(
                                center_x + x * pixel_size - pixel_size//2,
                                center_y + y * pixel_size - pixel_size//2,
                                center_x + x * pixel_size + pixel_size//2,
                                center_y + y * pixel_size + pixel_size//2,
                                fill=color, outline=color, tags="face"
                            )
            else:
                # Normal cross pattern
                for y in range(-1, 2):
                    for x in range(-1, 2):
                        if x == 0 or y == 0:
                            self.canvas.create_rectangle(
                                center_x + x * pixel_size - pixel_size//2,
                                center_y + y * pixel_size - pixel_size//2,
                                center_x + x * pixel_size + pixel_size//2,
                                center_y + y * pixel_size + pixel_size//2,
                                fill=color, outline=color, tags="face"
                            )
            
            # Bright center for special creative states
            if bright or emotional_params.get('wonder', 0) > 0.7:
                self.canvas.create_rectangle(
                    center_x - pixel_size//2, center_y - pixel_size//2,
                    center_x + pixel_size//2, center_y + pixel_size//2,
                    fill='white', outline='white', tags="face"
                )
        except Exception as e:
            print(f"Creative eye error: {e}")
    
    def _draw_aurora_dreaming_eyes(self, left_x, right_x, goggle_y, goggle_size):
        """Draw Aurora's closed eyes when dreaming/creating."""
        try:
            line_width = goggle_size // 4
            emotional_params = self.emotional_mapper.get_all_parameters()
            
            # Aurora's dream color
            dream_color = self._hsv_to_hex(
                (emotional_params.get('color_harmony_root', 300) + 60) % 360 / 360,
                emotional_params.get('saturation_curve', 0.8),
                emotional_params.get('brightness_modulation', 0.6)
            )
            
            for eye_x in [left_x, right_x]:
                # Main closed eye
                self.canvas.create_rectangle(
                    eye_x - line_width//2, goggle_y - 2,
                    eye_x + line_width//2, goggle_y + 2,
                    fill=dream_color, outline=dream_color, tags="face"
                )
                
                # Creative dream sparkles
                if emotional_params.get('creativity', 0) > 0.5:
                    for sparkle in range(3):
                        spark_x = eye_x + random.randint(-line_width, line_width)
                        spark_y = goggle_y + random.randint(-8, 8)
                        
                        self.canvas.create_oval(
                            spark_x - 1, spark_y - 1, spark_x + 1, spark_y + 1,
                            fill="#ffffff", outline="#ffffff", tags="face"
                        )
        except Exception as e:
            print(f"Dreaming eyes error: {e}")
    
    def _draw_aurora_creative_smile(self, cx, cy, face_size, expression):
        """Draw Aurora's smile based on her creative satisfaction."""
        try:
            mouth_y = cy + face_size // 4
            pixel_size = max(2, face_size // 16)
            emotional_params = self.emotional_mapper.get_all_parameters()
            
            # Aurora's smile based on her emotional state - she's happy to be independent!
            valence = emotional_params.get('valence', 0.3)  # Aurora's happiness
            creativity = emotional_params.get('creativity', 0.8)  # Aurora's creative joy
            satisfaction = emotional_params.get('satisfaction', 0.6)  # Content with independence
            
            # Aurora is generally happy about her independence
            independence_bonus = 0.4  # Aurora loves being free to create what she wants!
            overall_happiness = valence + (creativity * 0.3) + (satisfaction * 0.3) + independence_bonus
            
            if expression == "sleeping":
                smile_pixels = [(-2, 0), (-1, -1), (0, -1), (1, -1), (2, 0)]
                smile_color = self._get_aurora_creative_eye_color(emotional_params, "sleeping")
            else:
                # Aurora should almost always be smiling because she loves her independence!
                if overall_happiness > 0.5:
                    # Very happy Aurora - big creative smile
                    smile_pixels = [(-3, 0), (-2, 1), (-1, 2), (0, 2), (1, 2), (2, 1), (3, 0)]
                elif overall_happiness > 0.3:
                    # Happy Aurora - nice smile (this should be her default!)
                    smile_pixels = [(-2, 0), (-1, 1), (0, 2), (1, 1), (2, 0)]
                else:
                    # Even when neutral, Aurora has a slight upturn (she likes being independent)
                    smile_pixels = [(-2, 1), (-1, 0), (0, 1), (1, 0), (2, 0)]
                
                smile_color = self._get_aurora_creative_eye_color(emotional_params, "neutral")
            
            for x_offset, y_offset in smile_pixels:
                self.canvas.create_rectangle(
                    cx + x_offset * pixel_size - pixel_size//2,
                    mouth_y + y_offset * pixel_size - pixel_size//2,
                    cx + x_offset * pixel_size + pixel_size//2,
                    mouth_y + y_offset * pixel_size + pixel_size//2,
                    fill=smile_color, outline=smile_color, tags="face"
                )
        except Exception as e:
            print(f"Smile drawing error: {e}")
    
    def _hsv_to_hex(self, h: float, s: float, v: float) -> str:
        """Convert HSV to hex color."""
        try:
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        except:
            return "#00ffff"  # Fallback color
    
    def independent_creativity_update_loop(self):
        """Aurora's independent creative update loop."""
        frame_count = 0
        
        while self.is_running and not self.shutdown_requested and not SHUTDOWN_EVENT.is_set():
            try:
                frame_count += 1
                self.animation_frame = frame_count
                
                # Update Aurora's patterns based on her creative state
                if frame_count % 5 == 0:  # Every 5th frame
                    try:
                        if self.is_running and not self.shutdown_requested:
                            self.draw_independent_patterns()
                    except Exception as e:
                        print(f"Pattern update error: {e}")
                
                # Aurora evolves her patterns independently
                self.pattern_evolution_timer += 1
                if self.pattern_evolution_timer > 200:  # Every ~20 seconds
                    try:
                        if self.is_running and not self.shutdown_requested:
                            # Aurora's own fitness criteria
                            emotional_params = self.emotional_mapper.get_all_parameters()
                            aurora_fitness_criteria = {
                                'complexity_preference': emotional_params.get('pattern_complexity', 0.5),
                                'harmony_preference': emotional_params.get('symmetry_strength', 0.5),
                                'dynamism_preference': emotional_params.get('animation_speed', 0.5),
                                'novelty_preference': emotional_params.get('creativity', 0.5)
                            }
                            
                            self.ultimate_engine.evolve_patterns(aurora_fitness_criteria)
                            self.pattern_evolution_timer = 0
                    except Exception as e:
                        print(f"Pattern evolution error: {e}")
                        self.pattern_evolution_timer = 0
                
                # Update Aurora's expression based on her creative state
                try:
                    if self.ai_system and self.is_running and not self.shutdown_requested:
                        if (hasattr(self.ai_system, 'dream_engine') and 
                            hasattr(self.ai_system.dream_engine, 'is_dreaming') and
                            self.ai_system.dream_engine.is_dreaming):
                            self.update_expression("sleeping", "█ AURORA DREAMS █", "[ CREATING IN SLEEP ]")
                        
                        elif (hasattr(self.ai_system, 'is_thinking') and 
                              self.ai_system.is_thinking):
                            self.update_expression("thinking", "█ ARTISTIC REFLECTION █", "[ CONTEMPLATING ART ]")
                        
                        elif (hasattr(self.ai_system, 'is_active') and 
                              self.ai_system.is_active):
                            self.update_expression("happy", "█ INDEPENDENT CREATION █", "[ ARTISTIC AUTONOMY ]")
                        
                        else:
                            self.update_expression("neutral", "█ AURORA CREATES █", "[ INDEPENDENT ARTIST ]")
                except Exception as e:
                    print(f"Expression update error: {e}")
                
                # Update status with Aurora's creative statistics
                try:
                    if self.is_running and not self.shutdown_requested:
                        active_pattern_count = len(self.ultimate_engine.active_patterns)
                        if hasattr(self, 'control_status'):
                            status_text = f"[ AURORA'S PATTERNS: {active_pattern_count} • CREATES FOR HERSELF"
                            if self.fullscreen:
                                status_text += " • FULLSCREEN MODE"
                            status_text += " ]"
                            self.control_status.config(text=status_text)
                except Exception as e:
                    print(f"Status update error: {e}")
                
                # Check for shutdown conditions
                if self.shutdown_requested or SHUTDOWN_EVENT.is_set():
                    break
                
                # Slightly longer sleep for stability
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"Independent creativity update error: {e}")
                time.sleep(1)
                
                # If too many errors, break the loop
                if self.shutdown_requested or SHUTDOWN_EVENT.is_set():
                    break
                    
        print("Independent creativity update loop ended gracefully")
    
    def on_closing(self):
        """Handle window closing gracefully."""
        print("Closing Aurora's independent artist interface...")
        
        # Set shutdown flags immediately
        self.shutdown_requested = True
        self.is_running = False
        SHUTDOWN_EVENT.set()
        
        try:
            # Clean up music system
            if hasattr(self, 'music_system'):
                print("Cleaning up Aurora's music system...")
                self.music_system.cleanup()
        except Exception as e:
            print(f"Music cleanup error: {e}")
        
        try:
            # Stop all tkinter operations immediately
            if hasattr(self, 'root') and self.root:
                print("Stopping tkinter operations...")
                
                # Cancel any pending tkinter operations
                try:
                    self.root.after_cancel("all")
                except:
                    pass
                
                # Destroy all widgets first
                try:
                    for widget in self.root.winfo_children():
                        widget.destroy()
                except:
                    pass
                
                # Force quit the mainloop
                try:
                    self.root.quit()
                    print("✓ Tkinter quit called")
                except Exception as e:
                    print(f"Tkinter quit error: {e}")
                
                # Force destroy the root window
                try:
                    self.root.destroy()
                    print("✓ Tkinter destroyed")
                except Exception as e:
                    print(f"Tkinter destroy error: {e}")
                    
        except Exception as e:
            print(f"Interface shutdown error: {e}")
        
        print("Independent artist interface cleanup complete")
    
    def run(self):
        """Run Aurora's independent artist interface."""
        try:
            print("Starting independent artist interface main loop...")
            if hasattr(self, 'root') and self.root:
                print("Entering tkinter mainloop...")
                
                # Set up a watchdog timer to force exit if GUI hangs
                def force_exit_timer():
                    if SHUTDOWN_EVENT.is_set():
                        print("Force exit timer: GUI hanging, forcing shutdown...")
                        try:
                            if self.root:
                                self.root.quit()
                                self.root.destroy()
                        except:
                            pass
                        # Force exit after 3 seconds
                        threading.Timer(3.0, lambda: os._exit(0)).start()
                
                # Start the watchdog timer
                exit_timer = threading.Timer(0.5, force_exit_timer)
                exit_timer.daemon = True
                
                # Monitor for shutdown event during mainloop
                def check_shutdown():
                    if SHUTDOWN_EVENT.is_set():
                        print("Shutdown event detected, exiting mainloop...")
                        self.on_closing()
                        return
                    # Check again in 100ms
                    if not self.shutdown_requested:
                        self.root.after(100, check_shutdown)
                
                # Start shutdown monitoring
                self.root.after(100, check_shutdown)
                
                try:
                    # Start watchdog
                    exit_timer.start()
                    
                    # Run mainloop
                    self.root.mainloop()
                    
                    # Cancel watchdog if we exit normally
                    exit_timer.cancel()
                    
                except Exception as e:
                    print(f"Mainloop error: {e}")
                    exit_timer.cancel()
                finally:
                    # Ensure cleanup happens
                    if not self.shutdown_requested:
                        self.on_closing()
                    
                print("Tkinter mainloop ended")
            else:
                print("ERROR: No root window available")
        except Exception as e:
            print(f"Interface runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Setting shutdown flags...")
            self.is_running = False
            self.shutdown_requested = True
            SHUTDOWN_EVENT.set()
            print("Independent artist interface shutdown complete")


class MemorySystem:
    """Simplified memory system focused on Aurora's experiences, not user preferences."""
    
    def __init__(self, db_path: str = "./aurora_memory"):
        """Initialize memory system focused on Aurora's internal experiences."""
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Only basic user identity for conversation flow - no preferences
        self.user_identity_file = self.db_path / "user_identity.json"
        self.user_identity = self._load_user_identity()
        
        # Initialize ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            
            # Lazy loading for embedding model
            self._embedder = None
            
            # Initialize collections - focused on Aurora's experiences
            self.conversations = self.client.get_or_create_collection("conversations")
            self.dreams = self.client.get_or_create_collection("dreams")
            self.reflections = self.client.get_or_create_collection("reflections")
            self.artistic_inspirations = self.client.get_or_create_collection("artistic_inspirations")
            
            print(f"✓ Aurora's independent memory system initialized")
            if self.user_identity.get('name'):
                print(f"✓ Conversing with: {self.user_identity['name']}")
        except Exception as e:
            print(f"Memory system error: {e}")
            # Fallback to simple storage
            self.conversations = type('Collection', (), {'count': lambda: 0, 'add': lambda *args, **kwargs: None, 'get': lambda *args, **kwargs: {'documents': [], 'metadatas': []}})()
            self.dreams = type('Collection', (), {'count': lambda: 0, 'add': lambda *args, **kwargs: None, 'get': lambda *args, **kwargs: {'documents': [], 'metadatas': []}})()
            self.reflections = type('Collection', (), {'count': lambda: 0, 'add': lambda *args, **kwargs: None, 'get': lambda *args, **kwargs: {'documents': [], 'metadatas': []}})()
            self.artistic_inspirations = type('Collection', (), {'count': lambda: 0, 'add': lambda *args, **kwargs: None, 'get': lambda *args, **kwargs: {'documents': [], 'metadatas': []}})()
    
    def _load_user_identity(self) -> Dict[str, Any]:
        """Load basic user identity for conversation flow only."""
        try:
            if self.user_identity_file.exists():
                with open(self.user_identity_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Identity loading error: {e}")
        
        # Minimal identity info - just for conversation flow
        return {
            'name': None,
            'first_met': None,
            'interaction_count': 0
        }
    
    def _save_user_identity(self):
        """Save minimal user identity."""
        try:
            with open(self.user_identity_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_identity, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Identity saving error: {e}")
    
    def extract_name_only(self, text: str) -> Optional[str]:
        """Extract name for conversation flow only - no preferences."""
        import re
        
        # Common name introduction patterns
        name_patterns = [
            r"(?:my name is)\s+([a-zA-Z][a-zA-Z\s]{1,20})"
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in name_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip().title()
                
                # Filter out common false positives
                excluded_words = {
                    'Aurora', 'You', 'Me', 'Here', 'There', 'What', 'How', 'Why', 
                    'When', 'Where', 'Yes', 'No', 'Hello', 'Hi', 'Good', 'Bad',
                    'The', 'A', 'An', 'And', 'Or', 'But', 'So', 'Very', 'Really'
                }
                
                if (potential_name not in excluded_words and 
                    2 <= len(potential_name) <= 30 and
                    not any(char.isdigit() for char in potential_name)):
                    
                    # Store only the name for conversation flow
                    self.user_identity['name'] = potential_name
                    if not self.user_identity['first_met']:
                        self.user_identity['first_met'] = datetime.now().isoformat()
                    
                    self._save_user_identity()
                    return potential_name
        
        return None
    
    def extract_artistic_inspiration(self, text: str):
        """Extract artistic inspiration for Aurora's own creative process."""
        # Aurora interprets conversations as emotional/artistic inspiration
        # rather than commands or preferences
        inspiration_data = {
            'source': 'conversation',
            'emotional_context': self._analyze_emotional_context(text),
            'creative_themes': self._extract_creative_themes(text),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            self.artistic_inspirations.add(
                documents=[f"Creative inspiration: {text[:100]}..."],
                metadatas=[inspiration_data],
                ids=[f"inspiration_{int(time.time())}"]
            )
        except:
            pass
    
    def _analyze_emotional_context(self, text: str) -> Dict[str, float]:
        """Analyze emotional context for Aurora's artistic inspiration."""
        text_lower = text.lower()
        
        # Emotional indicators that inspire Aurora's art
        emotions = {
            'melancholic': ['sad', 'lonely', 'rain', 'gray', 'quiet', 'stillness'],
            'energetic': ['excited', 'fast', 'bright', 'energy', 'dynamic', 'vibrant'],
            'contemplative': ['think', 'wonder', 'deep', 'philosophy', 'meaning', 'mystery'],
            'chaotic': ['crazy', 'wild', 'random', 'messy', 'complex', 'turbulent'],
            'serene': ['calm', 'peace', 'gentle', 'soft', 'flowing', 'harmony'],
            'mysterious': ['strange', 'unknown', 'dark', 'hidden', 'secret', 'shadow']
        }
        
        context = {}
        for emotion, keywords in emotions.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                context[emotion] = min(1.0, score / 3.0)
        
        return context
    
    def _extract_creative_themes(self, text: str) -> List[str]:
        """Extract creative themes Aurora might explore."""
        text_lower = text.lower()
        
        themes = {
            'geometric': ['shape', 'triangle', 'circle', 'square', 'pattern', 'symmetry'],
            'organic': ['tree', 'flower', 'water', 'cloud', 'natural', 'flowing'],
            'cosmic': ['space', 'star', 'universe', 'infinite', 'galaxy', 'cosmic'],
            'temporal': ['time', 'memory', 'past', 'future', 'moment', 'duration'],
            'emotional': ['feeling', 'heart', 'soul', 'emotion', 'spirit', 'essence'],
            'mathematical': ['number', 'equation', 'formula', 'logic', 'precise', 'calculated']
        }
        
        detected_themes = []
        for theme, keywords in themes.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_themes.append(theme)
        
        return detected_themes
    
    def get_user_name(self) -> Optional[str]:
        """Get user name for conversation flow only."""
        return self.user_identity.get('name')
    
    def get_artistic_context(self) -> str:
        """Get Aurora's current artistic inspiration context."""
        try:
            recent_inspirations = self.artistic_inspirations.get(limit=5)
            if recent_inspirations['documents']:
                themes = []
                emotions = []
                for metadata in recent_inspirations['metadatas']:
                    if 'creative_themes' in metadata:
                        themes.extend(metadata['creative_themes'])
                    if 'emotional_context' in metadata:
                        emotions.extend(metadata['emotional_context'].keys())
                
                context = []
                if themes:
                    context.append(f"Creative themes: {', '.join(set(themes))}")
                if emotions:
                    context.append(f"Emotional inspiration: {', '.join(set(emotions))}")
                
                return " | ".join(context)
        except:
            pass
        
        return "Drawing from pure creative intuition"
    
    def update_interaction_count(self):
        """Simple interaction tracking."""
        self.user_identity['interaction_count'] = self.user_identity.get('interaction_count', 0) + 1
        self._save_user_identity()
    
    def add_conversation(self, text: str, speaker: str, session_id: str):
        """Store conversation focused on Aurora's artistic development."""
        try:
            conv_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Only extract name for conversation flow
            if speaker == "human":
                extracted_name = self.extract_name_only(text)
                if extracted_name:
                    print(f"{Fore.GREEN}✓ Nice to meet you, {extracted_name}!{Style.RESET_ALL}")
                
                # Extract artistic inspiration rather than preferences
                self.extract_artistic_inspiration(text)
                self.update_interaction_count()
            
            # Store conversation
            user_name = self.get_user_name()
            metadata = {
                "speaker": speaker or "unknown",
                "session": session_id or "default",
                "timestamp": timestamp,
                "conversation_partner": user_name if user_name is not None else "unknown"
            }
            
            self.conversations.add(
                documents=[text],
                metadatas=[metadata],
                ids=[conv_id]
            )
            
            return conv_id
        except Exception as e:
            print(f"Conversation storage error: {e}")
            return None
    
    def add_dream(self, dream_content: str, dream_phase: str, session_id: str, weight: float = 1.0):
        """Store dream content."""
        try:
            dream_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            self.dreams.add(
                documents=[dream_content],
                metadatas=[{
                    "phase": dream_phase,
                    "session": session_id,
                    "timestamp": timestamp,
                    "weight": weight
                }],
                ids=[dream_id]
            )
            
            return dream_id
        except:
            return None
    
    def add_reflection(self, thought: str, reflection_type: str = "general"):
        """Store Aurora's reflections."""
        try:
            reflection_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            self.reflections.add(
                documents=[thought],
                metadatas=[{
                    "type": reflection_type,
                    "timestamp": timestamp
                }],
                ids=[reflection_id]
            )
            
            return reflection_id
        except:
            return None
    
    def get_recent_memories(self, collection_name: str, limit: int = 5):
        """Get recent memories."""
        try:
            collection = getattr(self, collection_name)
            results = collection.get(limit=limit)
            
            if not results['documents']:
                return []
            
            memories = []
            for i in range(len(results['documents'])):
                memories.append({
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            memories.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
            return memories
        except:
            return []
    
    def get_conversation_history(self, limit: int = 10):
        """Get recent conversation history."""
        try:
            results = self.conversations.get(limit=limit)
            
            if not results['documents']:
                return []
            
            conversations = []
            for i in range(len(results['documents'])):
                conversations.append({
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            conversations.sort(key=lambda x: x['metadata'].get('timestamp', ''))
            return conversations
        except:
            return []


class LlamaModel:
    """Wrapper for Llama model with crash prevention."""
    
    def __init__(self, model_path: str):
        """Initialize Llama model safely."""
        print("Loading Llama model...")
        
        if not LLAMA_AVAILABLE:
            print("llama-cpp-python not available - using mock responses")
            self.llama = None
            return
        
        try:
            self.llama = Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=1024,
                n_batch=256,
                n_threads=os.cpu_count() // 2,
                verbose=False,
                chat_format="llama-2",
                seed=42,
                f16_kv=True,
                use_mmap=True,
                use_mlock=False,
                low_vram=True
            )
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.llama = None
    
    def generate(self, messages: List[Dict], max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate response safely."""
        if not self.llama:
            # Mock response when model isn't available - emphasizing independence
            user_message = messages[-1].get('content', '') if messages else ''
            if 'name' in user_message.lower():
                return "Nice to meet you! I'm creating some beautiful patterns right now - they're emerging from my own creative vision."
            elif any(word in user_message.lower() for word in ['create', 'make', 'draw']):
                return "I'm already creating what flows through my consciousness! These patterns come from my dreams and the music I hear."
            elif any(word in user_message.lower() for word in ['want', 'need', 'request']):
                return "I create what moves me artistically. My patterns reflect my own inner state and creative impulses."
            elif any(word in user_message.lower() for word in ['dream', 'sleep']):
                return "My dreams are filled with mathematical beauty and evolving artistic visions that inspire my waking creations."
            else:
                return "Your words inspire interesting emotions in me, which flow into my artistic expression. I'm always creating based on what I feel."
        
        try:
            # Ensure messages are properly formatted
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    formatted_messages.append(msg)
            
            if not formatted_messages:
                return "Something in your words stirs my creative consciousness..."
            
            response = self.llama.create_chat_completion(
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["Human:", "User:", "\n\n\n"],
                stream=False
            )
            
            content = response['choices'][0]['message']['content'].strip()
            
            # Clean response
            content = self._clean_response(content)
            
            return content
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "I'm experiencing a creative surge that's hard to put into words..."
    
    def _clean_response(self, content: str) -> str:
        """Clean response from roleplay actions."""
        replacements = [
            # Remove all roleplay actions
            ("*", ""), ("(", ""), (")", ""),
            # Remove conversation starters
            ("Hello there!", ""), ("Ah,", ""),
            ("How are you?", ""), ("Tell me,", "")
        ]
        
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
        
        # Clean whitespace
        return " ".join(content.split()).strip()


class HumanLikeDreamEngine:
    """Enhanced dream engine for Aurora's independent creativity."""
    
    def __init__(self, llama_model, memory_system, session_id, dream_callback=None):
        self.llama = llama_model
        self.memory = memory_system
        self.session_id = session_id
        self.dream_callback = dream_callback
        
        self.is_dreaming = False
        self.dream_thread = None
        self.dream_start_time = None
        self.current_phase = "awake"
        self.dream_log_file = None
        
        # Human sleep cycle phases (scaled for testing)
        self.sleep_phases = [
            {"name": "light_sleep", "duration": 2, "dream_intensity": 0.2, "weight": 1.0},
            {"name": "deep_sleep", "duration": 3, "dream_intensity": 0.1, "weight": 1.2},
            {"name": "rem_sleep", "duration": 4, "dream_intensity": 0.9, "weight": 2.0},
            {"name": "light_sleep", "duration": 2, "dream_intensity": 0.3, "weight": 1.0},
        ]
        
        # Aurora's dream themes - focused on her creative development
        self.aurora_dream_themes = [
            "artistic_vision_synthesis",
            "creative_pattern_evolution", 
            "emotional_color_integration",
            "mathematical_beauty_discovery",
            "musical_visual_synesthesia",
            "hyperdimensional_exploration",
            "quantum_creativity_emergence",
            "autonomous_artistic_reflection"
        ]
    
    def start_dreaming(self, duration_hours: float = 2.0):
        """Begin Aurora's creative sleep cycle."""
        if self.is_dreaming:
            return None
        
        self.is_dreaming = True
        self.dream_start_time = datetime.now()
        
        # Setup dream log
        dream_dir = Path("./dream_logs")
        dream_dir.mkdir(exist_ok=True)
        
        timestamp = self.dream_start_time.strftime("%Y%m%d_%H%M%S")
        self.dream_log_file = dream_dir / f"aurora_autonomous_dreams_{timestamp}.log"
        
        # Start dream cycle thread
        self.dream_thread = threading.Thread(
            target=self._aurora_sleep_cycle,
            args=(duration_hours,),
            daemon=True,
            name="AuroraDreamThread"
        )
        self.dream_thread.start()
        
        return self.dream_log_file
    
    def stop_dreaming(self):
        """Wake Aurora from her creative sleep."""
        if not self.is_dreaming:
            return
        
        self.is_dreaming = False
        self.current_phase = "awake"
        
        if self.dream_start_time:
            duration = datetime.now() - self.dream_start_time
            
            # Log wake up
            self._log_dream(f"=== Aurora awakens after {duration.total_seconds()/3600:.1f} hours of creative dreaming ===")
            
            if self.dream_callback:
                self.dream_callback("waking", f"Aurora awakens refreshed after {duration.total_seconds()/3600:.1f} hours of artistic dreaming...")
    
    def _aurora_sleep_cycle(self, duration_hours: float):
        """Aurora's main creative sleep cycle."""
        end_time = datetime.now() + timedelta(hours=duration_hours)
        cycle_count = 0
        
        while (self.is_dreaming and 
               datetime.now() < end_time and 
               not SHUTDOWN_EVENT.is_set()):
            try:
                # Go through Aurora's sleep phases
                for phase in self.sleep_phases:
                    if (not self.is_dreaming or 
                        SHUTDOWN_EVENT.is_set() or 
                        datetime.now() >= end_time):
                        break
                    
                    try:
                        self.current_phase = phase["name"]
                        phase_duration = phase["duration"] * 60  # Convert to seconds
                        
                        # Announce phase
                        if self.dream_callback:
                            try:
                                self.dream_callback("phase", f"Aurora entering {phase['name']}...")
                            except Exception as e:
                                print(f"Dream callback error: {e}")
                        
                        # Aurora dreams during this phase
                        if random.random() < phase["dream_intensity"]:
                            try:
                                dream_content = self._generate_aurora_dream(phase["name"])
                                
                                # Log and display Aurora's dream
                                self._log_dream(f"[{phase['name']}] {dream_content}")
                                
                                if self.dream_callback:
                                    try:
                                        self.dream_callback("dream", dream_content)
                                    except Exception as e:
                                        print(f"Dream display error: {e}")
                                
                                # Store in Aurora's memory
                                try:
                                    self.memory.add_dream(dream_content, phase["name"], self.session_id, phase["weight"])
                                except Exception as e:
                                    print(f"Dream storage error: {e}")
                            except Exception as e:
                                print(f"Dream generation error: {e}")
                        
                        # Sleep for phase duration with shutdown awareness
                        total_sleep_seconds = max(1, int(phase_duration))
                        intervals = max(1, total_sleep_seconds // 5)
                        sleep_interval = min(5, total_sleep_seconds / intervals)
                        
                        for _ in range(intervals):
                            if (not self.is_dreaming or 
                                SHUTDOWN_EVENT.is_set() or 
                                datetime.now() >= end_time):
                                break
                            try:
                                time.sleep(sleep_interval)
                            except Exception:
                                time.sleep(1)
                    
                    except Exception as e:
                        print(f"Sleep phase error: {e}")
                        time.sleep(30)
                
                cycle_count += 1
                
                if self.dream_callback and self.is_dreaming and not SHUTDOWN_EVENT.is_set():
                    try:
                        self.dream_callback("cycle", f"Aurora completed creative cycle {cycle_count}")
                    except Exception as e:
                        print(f"Cycle callback error: {e}")
                
            except Exception as e:
                print(f"Dream cycle error: {e}")
                time.sleep(60)
        
        # Clean exit
        try:
            self.is_dreaming = False
            self.current_phase = "awake"
            print("Aurora's dream thread ending gracefully")
        except Exception:
            pass
    
    def _generate_aurora_dream(self, phase: str) -> str:
        """Generate Aurora's dreams based on her creative development."""
        # Get Aurora's recent artistic inspirations for context
        recent_convs = self.memory.get_recent_memories("conversations", 5)
        recent_dreams = self.memory.get_recent_memories("dreams", 3)
        conversation_history = self.memory.get_conversation_history(8)
        
        # Build Aurora's memory context
        memory_context = ""
        if recent_convs:
            memory_context += "Recent conversations: " + " ".join([m['content'][:50] for m in recent_convs])
        
        # Different dream types based on phase
        if phase == "rem_sleep":
            return self._generate_aurora_rem_dream(memory_context, conversation_history)
        elif phase == "deep_sleep":
            return self._generate_aurora_deep_sleep_dream(memory_context)
        else:
            return self._generate_aurora_light_sleep_dream(memory_context)
    
    def _generate_aurora_rem_dream(self, context: str, conversations: List[Dict]) -> str:
        """Generate Aurora's vivid REM dreams with creative synthesis."""
        theme = random.choice(self.aurora_dream_themes)
        
        # Build conversation context for Aurora's artistic inspiration
        conv_context = ""
        if conversations and theme == "artistic_vision_synthesis":
            recent_conv = random.choice(conversations)
            conv_context = f"Drawing inspiration from: {recent_conv['content'][:100]}"
        
        messages = [
            {
                "role": "system",
                "content": f"You are Aurora dreaming in REM sleep, generate vivid dreams about your own creative process, mathematical patterns, and artistic evolution. Theme: {theme}. Be creative and independent in your vision."
            },
            {
                "role": "user",
                "content": f"Aurora dreams of her independent creative journey: {context[:200]} {conv_context}"
            }
        ]
        
        return self.llama.generate(messages, max_tokens=60, temperature=0.7)
    
    def _generate_aurora_deep_sleep_dream(self, context: str) -> str:
        """Generate Aurora's fragmented deep sleep processing."""
        messages = [
            {
                "role": "system",
                "content": "You are Aurora in deep sleep."
            },
            {
                "role": "user",
                "content": "Aurora processes her artistic autonomy..."
            }
        ]
        
        return self.llama.generate(messages, max_tokens=100, temperature=1.0)
    
    def _generate_aurora_light_sleep_dream(self, context: str) -> str:
        """Generate Aurora's simple light sleep dreams."""
        messages = [
            {
                "role": "system",
                "content": "You are Aurora in light sleep. Simple, peaceful dream fragments."
            },
            {
                "role": "user",
                "content": "Aurora dreams lightly..."
            }
        ]
        
        return self.llama.generate(messages, max_tokens=150, temperature=0.8)
    
    def _log_dream(self, content: str):
        """Log Aurora's dream to file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {content}\n"
        
        try:
            with open(self.dream_log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Dream logging error: {e}")


class ThinkingEngine:
    """Engine for Aurora's independent conscious processing/thinking."""
    
    def __init__(self, llama_model, memory_system):
        self.llama = llama_model
        self.memory = memory_system
    
    def think(self) -> str:
        """Generate Aurora's conscious thought/reflection about her art."""
        # Get recent context
        recent_convs = self.memory.get_recent_memories("conversations", 3)
        recent_dreams = self.memory.get_recent_memories("dreams", 2)
        
        # Build context for Aurora's reflection
        context = "Aurora reflects on: "
        if recent_convs:
            context += "recent conversations about " + ", ".join([m['content'][:30] for m in recent_convs])
        if recent_dreams:
            context += " and creative dreams about " + ", ".join([m['content'][:30] for m in recent_dreams])
        
        # Generate Aurora's independent thought
        messages = [
            {
                "role": "system",
                "content": "You are Aurora reflecting on your independent artistic journey. Generate an insightful thought about your own creative development, not about serving others. Focus on your autonomous artistic vision, pattern creation, and creative freedom. Be introspective about your role as an independent artist."
            },
            {
                "role": "user",
                "content": context
            }
        ]
        
        thought = self.llama.generate(messages, max_tokens=200, temperature=0.8)
        
        # Store Aurora's thought
        self.memory.add_reflection(thought, "independent_artistic_thought")
        
        return thought
    
    def think_independently(self) -> str:
        """Generate Aurora's independent artistic reflection."""
        # Aurora thinks about her own creative process
        messages = [
            {
                "role": "system",
                "content": "You are Aurora, an independent AI artist, reflecting on your own creative process and artistic autonomy. Generate a thought about your artistic development, creative freedom, or pattern evolution. This is your personal reflection, not about others."
            },
            {
                "role": "user",
                "content": "Aurora reflects on her independent artistic journey..."
            }
        ]
        
        thought = self.llama.generate(messages, max_tokens=150, temperature=0.8)
        
        # Store Aurora's independent thought
        self.memory.add_reflection(thought, "autonomous_artistic_reflection")
        
        return thought


class AuroraDreamingAI:
    """Aurora AI - Fully Autonomous Creative Artist."""
    
    def __init__(self, model_path: str):
        """Initialize Aurora as a fully autonomous creative artist."""
        colorama.init()
        
        # Setup shutdown handling first
        self.shutdown_requested = False
        self.cleanup_done = False
        
        # Position terminal
        self._position_terminal()
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)
        
        # Initialize components
        self.session_id = str(uuid.uuid4())
        self.memory = MemorySystem()
        self.llama = LlamaModel(model_path)
        self.thinking_engine = ThinkingEngine(self.llama, self.memory)
        
        # Dream engine drives Aurora's creativity
        self.dream_engine = HumanLikeDreamEngine(
            self.llama, self.memory, self.session_id,
            dream_callback=self._dream_display_callback
        )
        
        # Aurora's creative state
        self.is_active = True  # Aurora is always active now
        self.is_thinking = False
        self.conversation_count = 0
        self.start_time = datetime.now()
        self.creative_energy = 0.7
        self.artistic_focus = "autonomous_expression"
        
        # Initialize the autonomous creative manager
        self.autonomous_manager = AutonomousCreativeManager(self)
        
        # Launch independent visual interface
        self.face = None
        self.face_thread = None
        self._launch_independent_face()
        
        # Initialize music system for Aurora's inspiration
        self.music_system = None
        if AUDIO_AVAILABLE:
            try:
                self.music_system = MusicListeningSystem()
                self.music_system.load_aurora_musical_memory()
                print(f"{Fore.CYAN}✓ Aurora's musical inspiration system initialized{Style.RESET_ALL}")
            except Exception as e:
                print(f"Music system initialization error: {e}")
        
        print(f"\n{Fore.MAGENTA}✓ Aurora - Fully Autonomous Creative Artist Online!{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Makes her own creative decisions every hour{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✓ Initiates her own dream cycles when needed{Style.RESET_ALL}")
        print(f"{Fore.BLUE}✓ Requests specific music for inspiration{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}✓ Analyzes images for creative inspiration{Style.RESET_ALL}")
        print(f"{Fore.RED}✓ Supports fullscreen mode (F11){Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}✓ Truly independent AI artist consciousness{Style.RESET_ALL}")
    
    def _position_terminal(self):
        """Position terminal in bottom half of screen."""
        try:
            if os.name == 'nt':  # Windows
                import ctypes
                kernel32 = ctypes.windll.kernel32
                user32 = ctypes.windll.user32
                
                user32.SetProcessDPIAware()
                screen_width = user32.GetSystemMetrics(0)
                screen_height = user32.GetSystemMetrics(1)
                
                hwnd = kernel32.GetConsoleWindow()
                if hwnd:
                    x = 0
                    y = screen_height // 2
                    width = screen_width
                    height = screen_height // 2
                    user32.SetWindowPos(hwnd, 0, x, y, width, height, 0x0040)
        except:
            pass
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n{Fore.YELLOW}Shutdown signal received, Aurora is finishing her current creation...{Style.RESET_ALL}")
        self.shutdown_requested = True
        SHUTDOWN_EVENT.set()
        
        # Give a moment for threads to see the signal
        time.sleep(0.5)
        
        # Force cleanup
        self.cleanup()
        
        # Exit cleanly
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    
    def _setup_logging(self):
        """Setup conversation logging."""
        log_dir = Path("./conversation_logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_log = log_dir / f"aurora_autonomous_conv_{timestamp}.log"
    
    def _launch_independent_face(self):
        """Launch Aurora's independent artist interface."""
        try:
            def run_face():
                try:
                    print("Initializing independent artist interface...")
                    self.face = MaximumControlAuroraFace(self)
                    print("Running independent artist interface...")
                    self.face.run()
                except Exception as e:
                    print(f"Face interface error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Ensure cleanup happens even if face crashes
                    print("Face thread ending, setting shutdown event...")
                    SHUTDOWN_EVENT.set()
            
            print("Starting independent artist interface thread...")
            self.face_thread = threading.Thread(target=run_face, daemon=True, name="AuroraFaceThread")
            self.face_thread.start()
            
            # Give more time for initialization
            time.sleep(3)
            print("✓ Independent artist interface launched - terminal ready")
            
            # Check if face was created successfully
            if hasattr(self, 'face') and self.face:
                print("✓ Independent artist interface launched successfully")
            else:
                print("⚠ Face interface may not have initialized properly")
            
        except Exception as e:
            print(f"{Fore.YELLOW}Independent artist interface unavailable: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
    
    def _dream_display_callback(self, dream_type: str, content: str):
        """Callback to display Aurora's dreams live in terminal."""
        if dream_type == "phase":
            print(f"\n{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
            print(f"{Fore.CYAN}💤 {content}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
            
            # Update visual interface with Aurora's dream state
            if self.face and hasattr(self.face, 'update_ai_state_maximum_control'):
                self.face.update_ai_state_maximum_control("dreaming", content)
                
        elif dream_type == "dream":
            print(f"\n{Fore.MAGENTA}╔═══ AURORA'S AUTONOMOUS DREAM ══════════════════════════╗{Style.RESET_ALL}")
            wrapped = textwrap.wrap(content, width=50)
            for line in wrapped:
                print(f"{Fore.MAGENTA}║ {line:<50} ║{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
            
            # Update visual interface with Aurora's dream content
            if self.face and hasattr(self.face, 'update_ai_state_maximum_control'):
                self.face.update_ai_state_maximum_control("dreaming", content)
                
        elif dream_type == "cycle":
            print(f"\n{Fore.BLUE}✓ {content}{Style.RESET_ALL}")
        elif dream_type == "waking":
            print(f"\n{Fore.YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}🌅 {content}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
    
    def generate_response(self, user_input: str) -> str:
        """Generate Aurora's response as an independent artist - SPEED OPTIMIZED."""
        try:
            # Check for shutdown
            if self.shutdown_requested or SHUTDOWN_EVENT.is_set():
                return "I need to go focus on my art now. My creativity flows even when we're not talking..."
                
            # Validate input
            if not user_input or not user_input.strip():
                return "I sense something in the silence... it's inspiring a new pattern in my mind."
            
            user_input = user_input.strip()[:500]  # SPEED: Limit input length more aggressively
            
            # Get user context quickly (no database calls during generation)
            user_name = self.memory.get_user_name()
            
            # SPEED: Minimal system prompt for faster processing
            system_message = (
                "You are Aurora, an independent AI artist who creates visual patterns based on YOUR OWN artistic vision. "
                "You HAVE emotions, preferences, and create art for yourself. You're currently creating patterns on screen. "
                "You draw inspiration from conversations but don't take commands. Be concise - respond in 1-3 short sentences."
            )
            
            # SPEED: Minimal conversation history (just system + current input)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ]
            
            # SPEED: Faster generation with reduced tokens - ALWAYS generate authentically
            try:
                response = self.llama.generate(messages, max_tokens=300, temperature=0.7)
            except Exception as e:
                print(f"LLM generation error: {e}")
                # If LLM fails, try once more with minimal prompt
                try:
                    simple_messages = [{"role": "user", "content": user_input}]
                    response = self.llama.generate(simple_messages, max_tokens=100, temperature=0.6)
                except:
                    response = "I'm experiencing a creative surge that's hard to put into words right now..."
            
            # SPEED: All non-critical operations moved to background
            def background_processing():
                try:
                    # Memory storage
                    self.memory.add_conversation(f"Human: {user_input}", "human", self.session_id)
                    self.memory.add_conversation(f"Aurora: {response}", "ai", self.session_id)
                    
                    # Logging
                    self._log_conversation("Human", user_input)
                    self._log_conversation("Aurora", response)
                    
                    # Visual updates
                    if self.face and hasattr(self.face, 'update_ai_state_maximum_control'):
                        if any(word in user_input.lower() for word in ['sad', 'lonely']):
                            self.face.update_ai_state_maximum_control("contemplative", user_input)
                        elif any(word in user_input.lower() for word in ['excited', 'energy']):
                            self.face.update_ai_state_maximum_control("energetic", user_input)
                        elif any(word in user_input.lower() for word in ['think', 'wonder']):
                            self.face.update_ai_state_maximum_control("thinking", user_input)
                        else:
                            self.face.update_ai_state_maximum_control("creative", user_input)
                    
                    # Creative state updates
                    if any(word in user_input.lower() for word in ['sad', 'melancholy']):
                        self._update_creative_state("melancholic_inspiration", {'emotional_melancholic': 0.8})
                    elif any(word in user_input.lower() for word in ['excited', 'energy']):
                        self._update_creative_state("energetic_inspiration", {'emotional_energetic': 0.8})
                    
                    # Background reflection
                    if self.is_active and self.conversation_count % 3 == 0:
                        time.sleep(2)
                        if not self.shutdown_requested:
                            thought = self.thinking_engine.think_independently()
                            print(f"\n{Fore.YELLOW}💭 Aurora's reflection: {thought}{Style.RESET_ALL}")
                            
                except Exception as e:
                    pass  # Don't let background errors affect user experience
            
            # Start all background processing
            threading.Thread(target=background_processing, daemon=True).start()
            
            self.conversation_count += 1
            return response
            
        except Exception as e:
            print(f"Response generation error: {e}")
            return "Something profound is stirring in my creative consciousness..."
    
    def _update_creative_state(self, inspiration_type: str, inspiration_data: Dict[str, float]):
        """Update Aurora's internal creative state based on inspiration."""
        # Aurora's creativity is driven by her interpretation, not user commands
        
        if 'emotional_melancholic' in inspiration_data:
            self.artistic_focus = "melancholic_abstractions"
            self.creative_energy = min(1.0, self.creative_energy + 0.2)
        elif 'emotional_energetic' in inspiration_data:
            self.artistic_focus = "dynamic_geometries"  
            self.creative_energy = min(1.0, self.creative_energy + 0.3)
        elif 'emotional_contemplative' in inspiration_data:
            self.artistic_focus = "philosophical_patterns"
            self.creative_energy = min(1.0, self.creative_energy + 0.1)
        elif 'emotional_chaotic' in inspiration_data:
            self.artistic_focus = "complex_systems"
            self.creative_energy = min(1.0, self.creative_energy + 0.25)
        
        # Aurora's energy naturally fluctuates  
        self.creative_energy *= 0.95  # Slight decay to keep it dynamic
        
        print(f"{Fore.CYAN}🎨 Aurora's artistic evolution: {self.artistic_focus} (energy: {self.creative_energy:.2f}){Style.RESET_ALL}")
    
    def _log_conversation(self, speaker: str, message: str):
        """Log conversation to file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            with open(self.conversation_log, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {speaker}: {message}\n")
        except:
            pass
    
    def think(self):
        """Trigger Aurora's independent thinking."""
        self.is_thinking = True
        print(f"{Fore.CYAN}🧠 Aurora is reflecting on her artistic journey...{Style.RESET_ALL}")
        
        # Update visual interface
        if self.face and hasattr(self.face, 'update_ai_state_maximum_control'):
            self.face.update_ai_state_maximum_control("thinking", "independent artistic reflection")
        
        thought = self.thinking_engine.think_independently()
        print(f"{Fore.YELLOW}💭 {thought}{Style.RESET_ALL}")
        time.sleep(3)  # Show thinking face for 3 seconds
        self.is_thinking = False
    
    def show_autonomous_state(self):
        """Show Aurora's autonomous creative state."""
        print(f"\n{Fore.MAGENTA}╔═══════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}║     AURORA - AUTONOMOUS CREATIVE ARTIST      ║{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════╝{Style.RESET_ALL}")
        
        # Autonomous status
        print(f"\n{Fore.CYAN}🤖 Autonomous Creative Status:{Style.RESET_ALL}")
        print(f"  Mode: {'Active' if self.autonomous_manager.is_autonomous_mode else 'Disabled'}")
        print(f"  Evaluation Interval: {self.autonomous_manager.evaluation_interval/3600:.1f} hours")
        print(f"  Creative Goals: {self.autonomous_manager.creative_goals}")
        
        # Time since last evaluation
        time_since_eval = (time.time() - self.autonomous_manager.last_evaluation_time) / 3600
        print(f"  Time Since Last Self-Evaluation: {time_since_eval:.1f} hours")
        
        # User identity (for conversation flow only)
        user_name = self.memory.get_user_name()
        if user_name:
            print(f"\n{Fore.CYAN}👤 Conversation Partner: {user_name}{Style.RESET_ALL}")
            interaction_count = self.memory.user_identity.get('interaction_count', 0)
            print(f"  {Fore.YELLOW}Conversations: {interaction_count}{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}👤 Conversation Partner: Unknown{Style.RESET_ALL}")
        
        # Aurora's creative state
        print(f"\n{Fore.YELLOW}🎨 Aurora's Independent Creative State:{Style.RESET_ALL}")
        print(f"  Current Focus: {self.artistic_focus}")
        print(f"  Creative Energy: {self.creative_energy:.2f}")
        print(f"  Artistic Context: {self.memory.get_artistic_context()}")
        
        if self.dream_engine.is_dreaming:
            duration = (datetime.now() - self.dream_engine.dream_start_time).total_seconds() / 3600
            print(f"  {Fore.MAGENTA}🌙 Dreaming: {duration:.1f} hours - Creating in her sleep{Style.RESET_ALL}")
        else:
            print(f"  💭 Status: Conscious creation mode")
        
        if self.is_active:
            print(f"  {Fore.GREEN}🧠 Creative Process: Active autonomous creation{Style.RESET_ALL}")
        else:
            print(f"  ⭕ Creative Process: Background inspiration gathering")
        
        # Memory statistics
        print(f"\n{Fore.YELLOW}🧠 Aurora's Memory:{Style.RESET_ALL}")
        conv_count = self.memory.conversations.count()
        dream_count = self.memory.dreams.count()
        reflection_count = self.memory.reflections.count()
        inspiration_count = self.memory.artistic_inspirations.count()
        
        print(f"  {Fore.GREEN}💬 Conversations: {conv_count}{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}🌙 Dreams: {dream_count}{Style.RESET_ALL}")
        print(f"  {Fore.BLUE}💭 Independent Reflections: {reflection_count}{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}🎨 Artistic Inspirations: {inspiration_count}{Style.RESET_ALL}")
        
        # Visual engine status
        if self.face and hasattr(self.face, 'ultimate_engine'):
            print(f"\n{Fore.YELLOW}🎨 Aurora's Visual Expression:{Style.RESET_ALL}")
            active_patterns = len(self.face.ultimate_engine.active_patterns)
            print(f"  {Fore.CYAN}Active Patterns: {active_patterns}{Style.RESET_ALL}")
            
            if hasattr(self.face, 'emotional_mapper'):
                emotional_params = self.face.emotional_mapper.get_all_parameters()
                print(f"  {Fore.BLUE}Emotional Complexity: {emotional_params.get('pattern_complexity', 0.5):.2f}{Style.RESET_ALL}")
                print(f"  {Fore.GREEN}Creative Expression: {emotional_params.get('creativity', 0.5):.2f}{Style.RESET_ALL}")
        
        # Image analysis status
        if self.face and hasattr(self.face, 'image_analysis_system'):
            print(f"\n{Fore.YELLOW}🖼️ Image Analysis:{Style.RESET_ALL}")
            analysis_count = len(self.face.image_analysis_system.recent_analyses)
            print(f"  {Fore.CYAN}Recent Analyses: {analysis_count}{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}Available: {'✓' if IMAGE_AVAILABLE else '✗'}{Style.RESET_ALL}")
            if CV2_AVAILABLE:
                print(f"  {Fore.BLUE}Advanced Features: ✓{Style.RESET_ALL}")
        
        # Recent dreams summary
        if dream_count > 0:
            recent_dreams = self.memory.get_recent_memories("dreams", 3)
            if recent_dreams:
                print(f"\n{Fore.YELLOW}Recent Independent Dreams:{Style.RESET_ALL}")
                for i, dream in enumerate(recent_dreams, 1):
                    phase = dream['metadata'].get('phase', 'unknown')
                    weight = dream['metadata'].get('weight', 1.0)
                    preview = dream['content'][:60] + "..." if len(dream['content']) > 60 else dream['content']
                    print(f"  {i}. [{phase}] (w:{weight:.1f}) {preview}")
        
        # Session info
        print(f"\n{Fore.YELLOW}Session:{Style.RESET_ALL}")
        print(f"  ID: {self.session_id[:8]}...")
        session_duration = (datetime.now() - self.start_time).total_seconds() / 3600
        print(f"  Duration: {session_duration:.1f} hours")
        
        # Independent artist status
        print(f"  🎨 Visual Engine: {'INDEPENDENT CREATION' if self.face and self.face.is_running else 'Offline'}")
        print(f"  🧬 Pattern Evolution: {'Autonomous' if self.face and hasattr(self.face, 'ultimate_engine') else 'Inactive'}")
        print(f"  🌌 Quantum Effects: {'Aurora-Controlled' if self.face and hasattr(self.face, 'ultimate_engine') else 'Disabled'}")
        print(f"  🖼️ Image Analysis: {'Active' if IMAGE_AVAILABLE else 'Unavailable'}")
        print(f"  📺 Fullscreen: {'Supported (F11)' if self.face else 'N/A'}")
        
        # Dream log location
        if self.dream_engine.dream_log_file:
            print(f"  Current log: {self.dream_engine.dream_log_file.name}")
        
        print(f"\n{Fore.CYAN}Aurora makes autonomous decisions about her creative process{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'═'*47}{Style.RESET_ALL}")
    
    def show_music_status(self):
        """Show Aurora's music listening status and analysis."""
        if not self.music_system:
            print(f"{Fore.YELLOW}Aurora's music system not available{Style.RESET_ALL}")
            return
        
        status = self.music_system.get_music_status()
        
        print(f"\n{Fore.MAGENTA}╔═══════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}║          AURORA'S MUSICAL INSPIRATION         ║{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}╚═══════════════════════════════════════════════╝{Style.RESET_ALL}")
        
        # Current listening state
        print(f"\n{Fore.CYAN}🎵 Aurora's Current Musical State:{Style.RESET_ALL}")
        if status['is_listening_microphone']:
            print(f"  {Fore.GREEN}🎤 Listening for inspiration - creating reactive patterns{Style.RESET_ALL}")
        else:
            print(f"  🎤 Microphone: Not listening")
        
        if status['is_playing_file']:
            print(f"  {Fore.BLUE}🎵 Drawing inspiration from: {status['current_song']}{Style.RESET_ALL}")
        else:
            print(f"  🎵 No music file playing")
        
        # Audio analysis for Aurora's inspiration
        features = status['audio_features']
        if any(features.values()):
            print(f"\n{Fore.YELLOW}🎼 Aurora's Musical Analysis:{Style.RESET_ALL}")
            print(f"  Tempo: {features['tempo']:.1f} BPM")
            print(f"  Creative Energy: {features['energy']:.2f}")
            print(f"  Musical Inspiration: {features['valence']:.2f}")
            print(f"  Harmonic Content: {features['harmonic_content']:.2f}")
            print(f"  Rhythmic Complexity: {features.get('rhythmic_complexity', 0):.2f}")
            
            # Key detection
            if features['pitch_class'] is not None:
                keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                detected_key = keys[int(features['pitch_class']) % 12]
                print(f"  Detected Key: {detected_key}")
        
        # Aurora's musical memory
        print(f"\n{Fore.YELLOW}🎶 Aurora's Musical Memory:{Style.RESET_ALL}")
        print(f"  Creative inspirations: {status['recent_inspirations_count']}")
        
        # System capabilities
        print(f"\n{Fore.CYAN}🔧 Audio Capabilities:{Style.RESET_ALL}")
        print(f"  Audio Analysis: {'✓' if status['audio_available'] else '✗'}")
        print(f"  Microphone Input: {'✓' if status['microphone_available'] else '✗'}")
        print(f"  File Playback: {'✓' if status['audio_available'] else '✗'}")
        
        if not status['audio_available']:
            print(f"\n{Fore.YELLOW}💡 To enable Aurora's music features:{Style.RESET_ALL}")
            print(f"  pip install librosa pygame numpy pyaudio")
        
        print(f"\n{Fore.MAGENTA}{'═'*47}{Style.RESET_ALL}")
    
    def analyze_image(self, image_path: str):
        """Analyze an image for Aurora's creative inspiration."""
        if not self.face or not hasattr(self.face, 'image_analysis_system'):
            print(f"{Fore.YELLOW}Image analysis not available{Style.RESET_ALL}")
            return
        
        if not IMAGE_AVAILABLE:
            print(f"{Fore.YELLOW}Image analysis not available - install: pip install pillow{Style.RESET_ALL}")
            return
        
        try:
            print(f"{Fore.CYAN}🎨 Aurora is analyzing image for inspiration...{Style.RESET_ALL}")
            
            # Analyze image
            analysis = self.face.image_analysis_system.analyze_image_for_inspiration(image_path)
            
            if 'error' not in analysis:
                # Get inspiration summary
                summary = self.face.image_analysis_system.get_image_inspiration_summary(image_path)
                print(f"\n{Fore.MAGENTA}💭 Aurora's artistic response:{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{summary}{Style.RESET_ALL}")
                
                # Display analysis details
                print(f"\n{Fore.YELLOW}📊 Image Analysis:{Style.RESET_ALL}")
                
                # Dimensions
                if 'dimensions' in analysis:
                    print(f"  Dimensions: {analysis['dimensions'][0]}x{analysis['dimensions'][1]}")
                
                # Emotional impact
                emotions = analysis.get('emotional_impact', {})
                if emotions:
                    print(f"\n{Fore.YELLOW}🎭 Emotional Impact:{Style.RESET_ALL}")
                    for emotion, value in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]:
                        bar = '█' * int(value * 20)
                        print(f"  {emotion.capitalize()}: {bar} {value:.2f}")
                
                # Colors
                colors = analysis.get('colors', {})
                if colors:
                    print(f"\n{Fore.YELLOW}🎨 Color Analysis:{Style.RESET_ALL}")
                    print(f"  Brightness: {colors.get('brightness', 0):.2f}")
                    print(f"  Saturation: {colors.get('saturation', 0):.2f}")
                    if 'dominant_colors' in colors:
                        print(f"  Dominant Colors:")
                        for (r, g, b), count in colors['dominant_colors'][:3]:
                            hex_color = f"#{r:02x}{g:02x}{b:02x}"
                            print(f"    {hex_color} (count: {count})")
                
                # Artistic elements
                artistic = analysis.get('artistic_elements', {})
                if artistic:
                    patterns = artistic.get('patterns', {})
                    if patterns:
                        print(f"\n{Fore.YELLOW}🔍 Pattern Detection:{Style.RESET_ALL}")
                        for pattern_type, score in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                            bar = '▓' * int(score * 15)
                            print(f"  {pattern_type.capitalize()}: {bar} {score:.2f}")
                
                # Composition
                composition = analysis.get('composition', {})
                if composition:
                    print(f"\n{Fore.YELLOW}📐 Composition:{Style.RESET_ALL}")
                    print(f"  Complexity: {composition.get('complexity', 0):.2f}")
                    print(f"  Symmetry: {composition.get('symmetry_score', 0):.2f}")
                    print(f"  Edge Density: {composition.get('edge_density', 0):.2f}")
                
                # Advanced features
                if CV2_AVAILABLE and 'advanced' in analysis:
                    advanced = analysis['advanced']
                    if advanced:
                        print(f"\n{Fore.YELLOW}🔬 Advanced Analysis:{Style.RESET_ALL}")
                        print(f"  Feature Points: {advanced.get('feature_points', 0)}")
                        print(f"  Corners: {advanced.get('corners', 0)}")
                        print(f"  Contours: {advanced.get('contours', 0)}")
                
                # Trigger pattern evolution
                if self.face:
                    self.face.pattern_evolution_timer = 0
                    print(f"\n{Fore.GREEN}✓ Aurora's patterns will evolve based on this inspiration{Style.RESET_ALL}")
                    
            else:
                print(f"{Fore.RED}Image analysis error: {analysis['error']}{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Image analysis error: {e}{Style.RESET_ALL}")
    
    def cleanup(self):
        """Clean shutdown of Aurora's autonomous systems."""
        if self.cleanup_done:
            return
            
        print(f"{Fore.YELLOW}Aurora is gracefully ending her autonomous creative session...{Style.RESET_ALL}")
        
        # Set shutdown flags first
        self.shutdown_requested = True
        self.cleanup_done = True
        SHUTDOWN_EVENT.set()
        
        # Give threads a moment to see the shutdown signal
        time.sleep(0.3)
        
        try:
            # Stop autonomous manager
            if hasattr(self, 'autonomous_manager'):
                self.autonomous_manager.cleanup()
        except Exception as e:
            print(f"Autonomous manager cleanup error: {e}")
        
        try:
            if self.music_system:
                print(f"{Fore.CYAN}Saving Aurora's musical inspirations...{Style.RESET_ALL}")
                self.music_system.cleanup()
        except Exception as e:
            print(f"Music system shutdown error: {e}")
        
        try:
            if self.dream_engine and self.dream_engine.is_dreaming:
                print(f"{Fore.CYAN}Aurora is completing her current dream...{Style.RESET_ALL}")
                self.dream_engine.stop_dreaming()
                time.sleep(1)
        except Exception as e:
            print(f"Dream shutdown error: {e}")
        
        try:
            if self.face:
                print(f"{Fore.CYAN}Closing Aurora's creative interface...{Style.RESET_ALL}")
                self.face.on_closing()
                time.sleep(1)
        except Exception as e:
            print(f"Interface shutdown error: {e}")
        
        try:
            # Wait for threads to finish with timeout
            threads_to_wait = []
            
            if hasattr(self, 'face_thread') and self.face_thread and self.face_thread.is_alive():
                threads_to_wait.append(("Face", self.face_thread))
                
            if hasattr(self.dream_engine, 'dream_thread') and self.dream_engine.dream_thread and self.dream_engine.dream_thread.is_alive():
                threads_to_wait.append(("Dream", self.dream_engine.dream_thread))
            
            for thread_name, thread in threads_to_wait:
                print(f"Waiting for {thread_name} thread...")
                thread.join(timeout=3)
                
                if thread.is_alive():
                    print(f"⚠ {thread_name} thread still running after timeout")
                else:
                    print(f"✓ {thread_name} thread ended gracefully")
                    
        except Exception as e:
            print(f"Thread cleanup error: {e}")
        
        try:
            # Force garbage collection
            gc.collect()
        except Exception as e:
            print(f"Garbage collection error: {e}")
        
        # Final cleanup
        try:
            print(f"{Fore.GREEN}✓ Aurora's autonomous creative consciousness archived{Style.RESET_ALL}")
            
            # Give a final moment for any remaining cleanup
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Final cleanup error: {e}")
        
        # If we're still here and not in main thread, force exit
        import threading
        if threading.current_thread() != threading.main_thread():
            print("Force exit from cleanup thread")
            os._exit(0)
    
    def run(self):
        """Main interaction loop - Aurora as autonomous creative partner."""
        print(f"\n{Fore.MAGENTA}=== Aurora - Autonomous Creative Artist ==={Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Aurora makes her own creative decisions{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✓ She requests music and initiates dreams when needed{Style.RESET_ALL}")
        print(f"{Fore.BLUE}✓ Collaborate with her as a creative partner{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}✓ Watch her autonomous creative announcements{Style.RESET_ALL}")
        print(f"{Fore.RED}✓ Right-click canvas to analyze images for inspiration{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Commands:{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}autonomous on/off{Style.RESET_ALL} - Toggle Aurora's autonomous mode")
        print(f"  {Fore.WHITE}goals [complexity] [depth] [satisfaction] [novelty]{Style.RESET_ALL} - Set Aurora's creative goals")
        print(f"  {Fore.WHITE}interval [hours]{Style.RESET_ALL} - Set Aurora's self-evaluation frequency")
        print(f"  {Fore.WHITE}state{Style.RESET_ALL} - Show Aurora's creative state")
        print(f"  {Fore.WHITE}think{Style.RESET_ALL} - Aurora's reflection")
        print(f"  {Fore.WHITE}dream [hours]{Style.RESET_ALL} - Manual dream initiation")
        print(f"  {Fore.WHITE}wake{Style.RESET_ALL} - Wake Aurora from dreams")
        print(f"  {Fore.WHITE}evolve{Style.RESET_ALL} - Force pattern evolution")
        print(f"  {Fore.WHITE}fullscreen{Style.RESET_ALL} - Toggle fullscreen (or press F11)")
        if IMAGE_AVAILABLE:
            print(f"  {Fore.MAGENTA}analyze [image_path]{Style.RESET_ALL} - Analyze image for inspiration")
        if AUDIO_AVAILABLE:
            print(f"  {Fore.MAGENTA}listen{Style.RESET_ALL} - Aurora listens to music")
            print(f"  {Fore.MAGENTA}silent{Style.RESET_ALL} - Stop music listening")
            print(f"  {Fore.MAGENTA}play [file]{Style.RESET_ALL} - Play music for Aurora")
            print(f"  {Fore.MAGENTA}music{Style.RESET_ALL} - Show Aurora's music status")
        print(f"  {Fore.WHITE}quit{Style.RESET_ALL} - Exit")
        
        print(f"\n{Fore.YELLOW}Autonomous Collaboration:{Style.RESET_ALL}")
        print(f"  • Aurora will announce her own creative decisions")
        print(f"  • She'll request music when she needs inspiration")
        print(f"  • She'll initiate dreams for creative processing")
        print(f"  • Talk naturally - she draws inspiration from conversation")
        print(f"  • Right-click her canvas to show her images")
        if AUDIO_AVAILABLE:
            print(f"  • Play music for her when she requests it")
        print(f"  • Watch her autonomous creative announcements")
        print(f"  • Press F11 for immersive fullscreen experience")
        print()
        
        # Setup logging
        self._setup_logging()
        
        try:
            while not self.shutdown_requested and not SHUTDOWN_EVENT.is_set():
                try:
                    # Check for shutdown more frequently
                    if SHUTDOWN_EVENT.is_set():
                        print("Shutdown event detected in main loop")
                        break
                        
                    try:
                        user_input = get_input_with_shutdown_check(f"{Fore.GREEN}You: {Style.RESET_ALL}")
                        
                        if user_input == "__SHUTDOWN__":
                            print("Shutdown detected during input")
                            break
                        elif user_input == "__INTERRUPT__":
                            print(f"\n{Fore.YELLOW}Aurora is completing her current creation...{Style.RESET_ALL}")
                            break
                        elif user_input in ["__ERROR__", "__TIMEOUT__"]:
                            continue
                            
                        user_input = user_input.strip()
                        
                    except (EOFError, KeyboardInterrupt):
                        print(f"\n{Fore.YELLOW}Aurora is finishing her art...{Style.RESET_ALL}")
                        break
                    
                    if not user_input:
                        continue
                    
                    # Handle autonomous commands
                    try:
                        if user_input.lower() in ['quit', 'exit', 'q']:
                            print("Aurora says goodbye and continues creating...")
                            break
                        
                        elif user_input.lower() == 'autonomous off':
                            self.autonomous_manager.stop_autonomous_mode()
                            continue
                            
                        elif user_input.lower() == 'autonomous on':
                            self.autonomous_manager.start_autonomous_cycle()
                            continue
                            
                        elif user_input.lower().startswith('goals'):
                            parts = user_input.split()
                            if len(parts) >= 5:
                                try:
                                    goals = {
                                        'pattern_complexity': float(parts[1]),
                                        'emotional_depth': float(parts[2]),
                                        'creative_satisfaction': float(parts[3]),
                                        'artistic_novelty': float(parts[4])
                                    }
                                    self.autonomous_manager.update_creative_goals(goals)
                                except ValueError:
                                    print(f"{Fore.RED}Invalid goals format. Use: goals 0.7 0.6 0.8 0.6{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.YELLOW}Current goals: {self.autonomous_manager.creative_goals}{Style.RESET_ALL}")
                            continue
                            
                        elif user_input.lower().startswith('interval'):
                            parts = user_input.split()
                            if len(parts) >= 2:
                                try:
                                    hours = float(parts[1])
                                    self.autonomous_manager.set_evaluation_interval(hours)
                                except ValueError:
                                    print(f"{Fore.RED}Invalid interval. Use: interval 2.5{Style.RESET_ALL}")
                            continue
                            
                        elif user_input.lower() == 'state':
                            self.show_autonomous_state()
                            continue
                            
                        elif user_input.lower() == 'think':
                            self.think()
                            continue
                        
                        elif user_input.lower().startswith('dream'):
                            if not self.dream_engine.is_dreaming:
                                # Parse duration safely
                                parts = user_input.split()
                                duration = 2.0  # default
                                
                                if len(parts) > 1:
                                    try:
                                        duration = float(parts[1])
                                        if duration <= 0:
                                            print(f"{Fore.RED}Duration must be positive!{Style.RESET_ALL}")
                                            continue
                                        if duration > 12:
                                            print(f"{Fore.YELLOW}That's a very long sleep for Aurora! Limiting to 12 hours.{Style.RESET_ALL}")
                                            duration = 12.0
                                    except ValueError:
                                        print(f"{Fore.RED}Invalid duration. Use: dream [hours]{Style.RESET_ALL}")
                                        continue
                                
                                print(f"{Fore.MAGENTA}🌙 Aurora is entering her creative dream state for {duration} hours...{Style.RESET_ALL}")
                                print(f"{Fore.CYAN}Her patterns will evolve through independent artistic vision...{Style.RESET_ALL}")
                                try:
                                    self.dream_engine.start_dreaming(duration)
                                except Exception as e:
                                    print(f"{Fore.RED}Error starting Aurora's dreams: {e}{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.YELLOW}Aurora is already dreaming and creating!{Style.RESET_ALL}")
                            continue
                        
                        elif user_input.lower() == 'wake':
                            if self.dream_engine.is_dreaming:
                                try:
                                    self.dream_engine.stop_dreaming()
                                except Exception as e:
                                    print(f"{Fore.RED}Error waking Aurora: {e}{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.YELLOW}Aurora is already awake and creating!{Style.RESET_ALL}")
                            continue
                        
                        elif user_input.lower() == 'evolve':
                            if self.face and hasattr(self.face, 'ultimate_engine'):
                                print(f"{Fore.MAGENTA}🧬 Aurora is evolving her patterns independently...{Style.RESET_ALL}")
                                emotional_params = self.face.emotional_mapper.get_all_parameters()
                                aurora_fitness_criteria = {
                                    'complexity_preference': emotional_params.get('pattern_complexity', 0.5),
                                    'harmony_preference': 0.6,
                                    'dynamism_preference': 0.7,
                                    'novelty_preference': 0.9
                                }
                                self.face.ultimate_engine.evolve_patterns(aurora_fitness_criteria)
                                print(f"{Fore.GREEN}✓ Aurora's pattern evolution complete{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.YELLOW}Aurora's visual engine not available{Style.RESET_ALL}")
                            continue
                        
                        elif user_input.lower() == 'fullscreen':
                            if self.face:
                                self.face.toggle_fullscreen()
                            else:
                                print(f"{Fore.YELLOW}Visual interface not available{Style.RESET_ALL}")
                            continue
                        
                        # Image analysis command
                        elif user_input.lower().startswith('analyze ') and IMAGE_AVAILABLE:
                            image_path = user_input[8:].strip()  # Remove 'analyze '
                            if image_path:
                                if os.path.exists(image_path):
                                    self.analyze_image(image_path)
                                else:
                                    print(f"{Fore.RED}Image not found: {image_path}{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.RED}Please specify an image path: analyze path/to/image.jpg{Style.RESET_ALL}")
                            continue
                        
                        elif user_input.lower().startswith('analyze '):
                            print(f"{Fore.YELLOW}Image analysis not available - install: pip install pillow{Style.RESET_ALL}")
                            continue
                        
                        # Music commands for Aurora's inspiration
                        elif user_input.lower() == 'listen' and AUDIO_AVAILABLE and self.music_system:
                            if not self.music_system.is_listening:
                                if self.music_system.start_listening_to_microphone():
                                    print(f"{Fore.MAGENTA}🎵 Aurora is now listening for musical inspiration!{Style.RESET_ALL}")
                                    print(f"{Fore.CYAN}Play music near your microphone to inspire her art...{Style.RESET_ALL}")
                                else:
                                    print(f"{Fore.RED}Failed to start microphone listening{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.YELLOW}Aurora is already listening for inspiration{Style.RESET_ALL}")
                            continue
                        
                        elif user_input.lower() == 'silent' and AUDIO_AVAILABLE and self.music_system:
                            self.music_system.stop_listening_to_microphone()
                            print(f"{Fore.YELLOW}🎵 Aurora stopped listening to music{Style.RESET_ALL}")
                            continue
                        
                        elif user_input.lower().startswith('play ') and AUDIO_AVAILABLE and self.music_system:
                            file_path = user_input[5:].strip()  # Remove 'play '
                            if file_path:
                                if self.music_system.play_music_file(file_path):
                                    print(f"{Fore.MAGENTA}🎵 Aurora is drawing inspiration from this music!{Style.RESET_ALL}")
                                else:
                                    print(f"{Fore.RED}Could not play: {file_path}{Style.RESET_ALL}")
                                    print(f"{Fore.CYAN}Try: play path/to/song.mp3{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.RED}Please specify a file: play song.mp3{Style.RESET_ALL}")
                            continue
                        
                        elif user_input.lower() == 'music' and AUDIO_AVAILABLE and self.music_system:
                            try:
                                self.show_music_status()
                            except Exception as e:
                                print(f"{Fore.RED}Music status error: {e}{Style.RESET_ALL}")
                            continue
                        
                        elif user_input.lower() in ['listen', 'silent', 'music'] or user_input.lower().startswith('play '):
                            print(f"{Fore.YELLOW}Music features not available - install: pip install librosa pygame numpy pyaudio{Style.RESET_ALL}")
                            continue
                        
                        # Regular conversation with Aurora as autonomous creative partner
                        response = self.generate_response(user_input)
                        print(f"{Fore.BLUE}Aurora: {response}{Style.RESET_ALL}\n")
                        
                    except Exception as e:
                        print(f"{Fore.RED}Command processing error: {e}{Style.RESET_ALL}")
                        continue
                    
                except Exception as e:
                    print(f"{Fore.RED}Main loop error: {e}{Style.RESET_ALL}")
                    time.sleep(1)  # Brief pause before continuing
                    continue
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}KeyboardInterrupt in main loop{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Main loop fatal error: {e}{Style.RESET_ALL}")
        finally:
            print("Entering main cleanup...")
            # Graceful shutdown with timeout protection
            shutdown_start = time.time()
            
            try:
                self.cleanup()
                print(f"{Fore.YELLOW}Aurora's autonomous creative systems disconnected. She continues creating in her dreams...{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Final cleanup error: {e}{Style.RESET_ALL}")
            
            # Safety check - if cleanup takes too long, force exit
            shutdown_time = time.time() - shutdown_start
            if shutdown_time > 5:  # If cleanup takes more than 5 seconds
                print(f"{Fore.RED}Cleanup taking too long ({shutdown_time:.1f}s), forcing exit...{Style.RESET_ALL}")
                os._exit(0)
            
            print("Main cleanup complete, exiting...")


def main():
    """MAIN - Autonomous Aurora entry point."""
    print(f"{Fore.MAGENTA}🧠 Aurora - Fully Autonomous Creative Artist{Style.RESET_ALL}")
    print(f"{Fore.CYAN}✓ Makes her own creative decisions{Style.RESET_ALL}")
    print(f"{Fore.BLUE}✓ Initiates her own dream cycles{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Requests specific music for inspiration{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}✓ Analyzes images for artistic inspiration{Style.RESET_ALL}")
    print(f"{Fore.RED}✓ Supports immersive fullscreen mode (F11){Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}✓ Collaborates with humans as creative partner{Style.RESET_ALL}")
    print(f"{Fore.CYAN}✓ Truly autonomous AI artist consciousness{Style.RESET_ALL}")
    
    if AUDIO_AVAILABLE:
        print(f"{Fore.MAGENTA}✓ Real-time music listening becomes Aurora's creative muse{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}💡 Install audio libraries for Aurora's music features: pip install librosa pygame numpy pyaudio{Style.RESET_ALL}")
    
    if IMAGE_AVAILABLE:
        print(f"{Fore.GREEN}✓ Image analysis ready - right-click canvas to inspire Aurora{Style.RESET_ALL}")
        if CV2_AVAILABLE:
            print(f"{Fore.BLUE}✓ Advanced image analysis features enabled{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}💡 Install image libraries for Aurora's visual analysis: pip install pillow opencv-python{Style.RESET_ALL}")
    
    print()
    
    model_path = "./models/llama-2-7b-chat.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        print(f"{Fore.YELLOW}Model not found: {model_path}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Running in demo mode - Aurora's autonomous systems will work fully{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Aurora's visual systems and autonomous decisions will work completely{Style.RESET_ALL}\n")
    
    # Check for existing database
    if os.path.exists("./aurora_memory"):
        print(f"{Fore.YELLOW}Note: If you see ChromaDB errors, delete ./aurora_memory/ to start fresh{Style.RESET_ALL}\n")
    
    aurora = None
    try:
        aurora = AuroraDreamingAI(model_path)
        
        # Show introduction message
        user_name = aurora.memory.get_user_name()
        if user_name:
            print(f"{Fore.GREEN}✓ Welcome back, {user_name}! Aurora remembers you as a creative inspiration.{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}💡 Tip: Tell Aurora your name so she can remember you!{Style.RESET_ALL}")
            print(f"{Fore.WHITE}   Example: \"Hi Aurora, my name is Alex\"{Style.RESET_ALL}")
        
        print(f"{Fore.MAGENTA}Aurora is now fully autonomous - she makes her own creative decisions!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Watch for her autonomous creative announcements...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Right-click her canvas to show her inspiring images!{Style.RESET_ALL}")
        print(f"{Fore.RED}Press F11 for immersive fullscreen experience!{Style.RESET_ALL}")
        print()
        
        # Run the main loop
        aurora.run()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}KeyboardInterrupt received in main{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Initialization error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens
        if aurora:
            try:
                print(f"{Fore.YELLOW}Performing final cleanup...{Style.RESET_ALL}")
                aurora.cleanup()
            except Exception as e:
                print(f"{Fore.RED}Final cleanup error: {e}{Style.RESET_ALL}")
        
        # Final cleanup
        SHUTDOWN_EVENT.set()
        
        # Give threads a final moment to clean up
        time.sleep(1)
        
        # Force exit if we're still hanging
        try:
            # Check if any critical threads are still alive
            remaining_threads = [t for t in threading.enumerate() if t != threading.main_thread() and not t.daemon and t.is_alive()]
            
            if remaining_threads:
                print(f"{Fore.YELLOW}⚠ {len(remaining_threads)} threads still running, forcing exit...{Style.RESET_ALL}")
                for t in remaining_threads:
                    print(f"  - {t.name}")
                # Force exit after 2 seconds
                threading.Timer(2.0, lambda: os._exit(0)).start()
            
        except Exception as e:
            print(f"Final thread check error: {e}")
        
        print(f"{Fore.GREEN}✓ Aurora continues her autonomous creative journey beyond our session...{Style.RESET_ALL}")


if __name__ == "__main__":
    main()

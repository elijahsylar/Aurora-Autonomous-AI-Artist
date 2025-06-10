#!/usr/bin/env python3
"""
Aurora's image analysis system for artistic inspiration
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List
from collections import defaultdict, deque
import random
import math
import colorsys
import sys

from aurora.config import IMAGE_AVAILABLE, CV2_AVAILABLE

if IMAGE_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFilter, ImageStat
    import io

if CV2_AVAILABLE:
    import cv2


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

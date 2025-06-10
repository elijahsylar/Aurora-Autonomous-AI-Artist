#!/usr/bin/env python3
"""
Aurora's visual interface - Independent Artist Edition with Fullscreen
"""

import os
import time
import threading
import random
import math
import colorsys
import tkinter as tk
from tkinter import Canvas, filedialog
from pathlib import Path
from typing import Dict, Tuple, List, Any
from colorama import Fore, Style

from aurora.config import SHUTDOWN_EVENT, IMAGE_AVAILABLE, CV2_AVAILABLE, AUDIO_AVAILABLE
from aurora.emotional.emotional_mapper import EmotionalParameterMapper
from aurora.emotional.conversation_analyzer import ConversationVisualAnalyzer
from aurora.patterns.ultimate_engine import UltimatePatternEngine
from aurora.patterns.ai_controller import AIPatternController
from aurora.systems.music_system import MusicListeningSystem
from aurora.systems.image_analysis import ImageAnalysisSystem

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


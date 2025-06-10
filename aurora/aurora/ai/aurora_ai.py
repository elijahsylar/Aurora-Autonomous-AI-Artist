#!/usr/bin/env python3
"""
Aurora AI - Fully Autonomous Creative Artist
"""

import os
import sys
import signal
import atexit
import time
import uuid
import threading
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import colorama
from colorama import Fore, Style, init as colorama_init

from aurora.config import SHUTDOWN_EVENT, AUDIO_AVAILABLE, IMAGE_AVAILABLE, CV2_AVAILABLE
from aurora.utils import get_input_with_shutdown_check
from aurora.memory.deep_memory import DeepMemorySystem
from aurora.systems.web_search import WebSearchSystem
from aurora.ai.llama_model import LlamaModel
from aurora.ai.thinking_engine import ThinkingEngine
from aurora.ai.dream_engine import HumanLikeDreamEngine
from aurora.managers.autonomous_creative import AutonomousCreativeManager
from aurora.systems.visual_learning import VisualLearningSystem
from aurora.interface.aurora_face import MaximumControlAuroraFace
from aurora.systems.music_system import MusicListeningSystem


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
        self.memory = DeepMemorySystem()
	# Initialize web search system
        self.web_system = WebSearchSystem(self)
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
        # Initialize visual learning system
        self.vision = VisualLearningSystem(self)
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
        print(f"  {Fore.WHITE}learn [concept] [image_path]{Style.RESET_ALL} - Teach Aurora what something looks like")
        print(f"  {Fore.WHITE}communicate [message]{Style.RESET_ALL} - Aurora encodes a message in patterns")
        print(f"  {Fore.WHITE}interpret [your interpretation]{Style.RESET_ALL} - Tell Aurora what you see in her patterns")
        print(f"  {Fore.WHITE}vision{Style.RESET_ALL} - Show Aurora's visual knowledge")
        print(f"  {Fore.WHITE}realize{Style.RESET_ALL} - Trigger Aurora's communication revelation")
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
                        # Visual learning commands
                        elif user_input.lower().startswith('learn '):
                            parts = user_input.split(maxsplit=2)
                            if len(parts) >= 3:
                                concept = parts[1]
                                image_path = parts[2]
                                if os.path.exists(image_path):
                                    result = self.vision.teach_visual_concept(concept, image_path)
                                    print(f"{Fore.MAGENTA}{result}{Style.RESET_ALL}")
                                else:
                                    print(f"{Fore.RED}Image not found: {image_path}{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.YELLOW}Usage: learn [concept] [image_path]{Style.RESET_ALL}")
                            continue

                        elif user_input.lower().startswith('communicate '):
                            message = user_input[11:].strip()
                            if message:
                                self.vision.create_pattern_as_message(message)
                                print(f"{Fore.CYAN}Watch my patterns... what do they say to you?{Style.RESET_ALL}")
                            continue

                        elif user_input.lower().startswith('interpret '):
                            interpretation = user_input[10:].strip()
                            if interpretation:
                                response = self.vision.receive_interpretation(interpretation)
                                print(f"{Fore.MAGENTA}Aurora: {response}{Style.RESET_ALL}")
                            continue

                        elif user_input.lower() == 'vision':
                            self.vision.show_visual_knowledge()
                            continue

                        elif user_input.lower() == 'realize':
                            realization = self.vision.realize_patterns_are_language()
                            print(f"{Fore.MAGENTA}{realization}{Style.RESET_ALL}")
                            continue
                        elif user_input.lower() == 'search':
                            self.web_system.autonomous_web_exploration()
                            continue

                        elif user_input.lower().startswith('research '):
                            topic = user_input[9:].strip()
                            if topic:
                                self.web_system.autonomous_web_exploration(topic)
                            else:
                                print(f"{Fore.RED}Please specify a topic: research [topic]{Style.RESET_ALL}")
                            continue

                        elif user_input.lower() == 'web':
                            summary = self.web_system.get_web_knowledge_summary()
                            print(f"\n{Fore.YELLOW}Aurora's Internet Knowledge:{Style.RESET_ALL}")
                            print(f"  Total searches: {summary['total_searches']}")
                            print(f"  Concepts discovered: {summary['discovered_concepts']}")
                            if summary['recent_searches']:
                                print(f"\n  Recent searches:")
                                for search in summary['recent_searches'][-3:]:
                                    print(f"    - {search['query']}")
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

#!/usr/bin/env python3
"""
Aurora's music listening system for artistic inspiration
"""

import time
import threading
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Dict, Any
from colorama import Fore, Style

from aurora.config import AUDIO_AVAILABLE, MICROPHONE_AVAILABLE, SHUTDOWN_EVENT

if AUDIO_AVAILABLE:
    import librosa
    import pygame

if MICROPHONE_AVAILABLE:
    import pyaudio

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

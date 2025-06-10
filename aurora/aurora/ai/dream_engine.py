#!/usr/bin/env python3
"""
Aurora's enhanced dream engine for independent creativity
"""

import time
import threading
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
from collections import deque

from aurora.config import SHUTDOWN_EVENT

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


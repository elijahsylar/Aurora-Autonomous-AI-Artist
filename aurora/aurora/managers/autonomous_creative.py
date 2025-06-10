#!/usr/bin/env python3
"""
Aurora's autonomous creative decision-making manager
"""

import time
import random
import threading
from datetime import datetime
from typing import Dict, List, Any
from colorama import Fore, Style

from aurora.config import SHUTDOWN_EVENT, AUDIO_AVAILABLE

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
        elif random.random() < 0.47:  # Add 47% chance for web exploration
            assessment['primary_need'] = 'web_exploration'
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
            'creative_synthesis': 'initiate_synthesis_dream',
            'web_exploration': 'explore_web' 
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
            'request_inspiration': f"✨ My creativity is running low. I could use some inspiring conversation or complex musical textures!",
            'explore_web': f"🌐 I'm curious about the wider world. Time to explore the internet for artistic inspiration!",
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
                music_options = [
                    ['electronic ', 'techno', 'high_energy'],
                    ['boards_of_canada', 'layered_ambient', 'complex_rhythms'],
                    ['brian_eno', 'ambient', 'flowing'],
                    ['classical', 'orchestral', 'piano'],
                    ['experimental', 'glitch', 'complex_rhythms']
                ]
                self._request_specific_music(random.choice(music_options))
                
            elif decision == 'initiate_processing_dream':
                self._initiate_autonomous_dream(2.0)
                
            elif decision == 'initiate_synthesis_dream':
                self._initiate_autonomous_dream(3.0)
                
            elif decision == 'evolve_patterns':
                self._evolve_patterns_autonomously()
                
            elif decision == 'request_inspiration':
                self._request_creative_inspiration()
               
            elif decision == 'explore_web':
                self._explore_web_for_inspiration()
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
    def _explore_web_for_inspiration(self):
        """Aurora explores the web for artistic inspiration."""
        try:
           if hasattr(self.aurora, 'web_system'):
               self.aurora.web_system.autonomous_learning_cycle()
           else:
               print(f"{Fore.YELLOW}Web system not available{Style.RESET_ALL}")
        except Exception as e:
               print(f"Web exploration error: {e}")

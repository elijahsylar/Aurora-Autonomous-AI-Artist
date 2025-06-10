#!/usr/bin/env python3
"""
Engine for Aurora's independent conscious processing/thinking
"""

from typing import Dict

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


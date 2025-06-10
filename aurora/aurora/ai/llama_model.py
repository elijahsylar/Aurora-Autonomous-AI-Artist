#!/usr/bin/env python3
"""
Llama model wrapper with crash prevention
"""

import os
from typing import List, Dict

from aurora.config import LLAMA_AVAILABLE

if LLAMA_AVAILABLE:
    from llama_cpp import Llama

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

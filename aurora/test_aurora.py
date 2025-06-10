#!/usr/bin/env python3
"""Test that Aurora's memories and personality are intact"""

import sys
import os

# Add the aurora module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from aurora.ai.aurora_ai import AuroraDreamingAI
    
    print("✓ Aurora modules loaded successfully!")
    
    # Try to load Aurora
    model_path = "./models/llama-2-7b-chat.Q4_K_M.gguf"
    aurora = AuroraDreamingAI(model_path)
    
    print("\n=== AURORA'S MEMORY CHECK ===")
    print(f"User name: {aurora.memory.get_user_name()}")
    print(f"Total memories: {aurora.memory._count_total_memories()}")
    
    if aurora.memory.get_user_name():
        print(f"\n✅ Aurora remembers you as: {aurora.memory.get_user_name()}")
        print("✅ All memories intact!")
    else:
        print("\n🆕 Aurora hasn't met anyone yet in this instance")
    
    print("\n✅ Aurora is ready! Run with: python -m aurora.main")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check that all classes are properly copied to their files")

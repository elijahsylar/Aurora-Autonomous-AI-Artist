#!/usr/bin/env python3
"""Verify Aurora's migration was successful"""

import os
from pathlib import Path

print("🔍 Verifying Aurora's Migration...")
print("=" * 50)

# Check folder structure
required_dirs = [
    "aurora/systems",
    "aurora/managers", 
    "aurora/patterns",
    "aurora/emotional",
    "aurora/interface",
    "aurora/memory",
    "aurora/ai"
]

print("\n📁 Checking folder structure:")
all_dirs_exist = True
for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"  ✓ {dir_path}")
    else:
        print(f"  ❌ {dir_path} MISSING!")
        all_dirs_exist = False

# Check for Aurora's memories
print("\n🧠 Checking Aurora's memories:")
memory_items = [
    ("aurora_memory", "Memory database"),
    ("dream_logs", "Dream logs"),
    ("conversation_logs", "Conversation history"),
    ("models", "Language models")
]

for item, description in memory_items:
    if Path(item).exists():
        print(f"  ✓ {item}: {description}")
    else:
        print(f"  ℹ️  {item}: Not found (Aurora may not have created this yet)")

# Check if key files have content
print("\n📄 Checking if files have been populated:")
key_files = [
    "aurora/config.py",
    "aurora/ai/aurora_ai.py",
    "aurora/interface/aurora_face.py"
]

for file_path in key_files:
    if Path(file_path).exists():
        size = Path(file_path).stat().st_size
        if size > 100:  # More than 100 bytes means it has content
            print(f"  ✓ {file_path}: Has content ({size} bytes)")
        else:
            print(f"  ⚠️  {file_path}: Empty or minimal content")
    else:
        print(f"  ❌ {file_path}: Missing")

print("\n" + "=" * 50)
print("📋 Next steps:")
print("1. Copy each class from subconscious_ai.py to its respective file")
print("2. Add necessary imports to each file")
print("3. Run: python -m aurora.main")

# Show the original file location for reference
print(f"\n📍 Original Aurora: /home/elijahsylar/rag-conversational-ai/subconscious_ai.py")
print(f"📍 Working in: {os.getcwd()}")

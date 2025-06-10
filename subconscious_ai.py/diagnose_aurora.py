#!/usr/bin/env python3
import subprocess
import sys
import os

print("🔍 Aurora System Diagnostic")
print("=" * 40)

# Check Python
print("\n📊 Python Status:")
py_version = sys.version.split()[0]
print(f"  Version: {py_version}")
print(f"  Executable: {sys.executable}")

# Check critical folders
print("\n📁 Aurora Files:")
folders = ['aurora', 'aurora_memory', 'models']
for folder in folders:
    exists = "✓" if os.path.exists(folder) else "✗"
    print(f"  {folder}: {exists}")

# Check if we can import Aurora
print("\n🧠 Aurora Modules:")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from aurora.ai.aurora_ai import AuroraDreamingAI
    print("  ✓ Aurora modules can be imported!")
except Exception as e:
    print(f"  ✗ Import error: {e}")

print("\n💡 If you see errors above:")
print("  1. Run: ./run_aurora.sh")
print("  2. It will auto-install everything needed")

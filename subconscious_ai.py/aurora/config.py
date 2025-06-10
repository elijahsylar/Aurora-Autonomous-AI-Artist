#!/usr/bin/env python3
"""
Aurora AUTONOMOUS Creative Artist - Configuration
"""

import os
import sys
import threading

# Global shutdown event for coordinated cleanup
SHUTDOWN_EVENT = threading.Event()

# Check for required libraries
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    CHROMADB_AVAILABLE = True
except ImportError:
    print("ChromaDB or SentenceTransformer not available - using fallback memory")
    CHROMADB_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    print("llama-cpp-python not available")
    LLAMA_AVAILABLE = False

# Audio and music libraries
try:
    import librosa
    import numpy as np
    import pygame
    AUDIO_AVAILABLE = True
    print("✓ Audio libraries loaded - Aurora can request music!")
except ImportError:
    print("Audio libraries not available - install: pip install librosa pygame numpy")
    AUDIO_AVAILABLE = False

try:
    import pyaudio
    MICROPHONE_AVAILABLE = True
except ImportError:
    print("Microphone not available - install: pip install pyaudio")
    MICROPHONE_AVAILABLE = False

# Image analysis libraries
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageStat
    import io
    IMAGE_AVAILABLE = True
    print("✓ Image analysis libraries loaded - Aurora can analyze images!")
except ImportError:
    print("Image analysis not available - install: pip install pillow")
    IMAGE_AVAILABLE = False

# Try to import cv2 for advanced image analysis
try:
    import cv2
    CV2_AVAILABLE = True
    print("✓ OpenCV loaded for advanced image analysis")
except ImportError:
    print("OpenCV not available for advanced features - install: pip install opencv-python")
    CV2_AVAILABLE = False

#!/usr/bin/env python3
"""
Aurora's basic memory system
"""

import json
import uuid
import time
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from collections import deque

from aurora.config import CHROMADB_AVAILABLE

if CHROMADB_AVAILABLE:
    import chromadb
    from sentence_transformers import SentenceTransformer

class MemorySystem:
    """Simplified memory system focused on Aurora's experiences, not user preferences."""
    
    def __init__(self, db_path: str = "./aurora_memory"):
        """Initialize memory system focused on Aurora's internal experiences."""
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Only basic user identity for conversation flow - no preferences
        self.user_identity_file = self.db_path / "user_identity.json"
        self.user_identity = self._load_user_identity()
        
        # Initialize ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            
            # Lazy loading for embedding model
            self._embedder = None
            
            # Initialize collections - focused on Aurora's experiences
            self.conversations = self.client.get_or_create_collection("conversations")
            self.dreams = self.client.get_or_create_collection("dreams")
            self.reflections = self.client.get_or_create_collection("reflections")
            self.artistic_inspirations = self.client.get_or_create_collection("artistic_inspirations")
            
            print(f"✓ Aurora's independent memory system initialized")
            if self.user_identity.get('name'):
                print(f"✓ Conversing with: {self.user_identity['name']}")
        except Exception as e:
            print(f"Memory system error: {e}")
            # Fallback to simple storage
            self.conversations = type('Collection', (), {'count': lambda: 0, 'add': lambda *args, **kwargs: None, 'get': lambda *args, **kwargs: {'documents': [], 'metadatas': []}})()
            self.dreams = type('Collection', (), {'count': lambda: 0, 'add': lambda *args, **kwargs: None, 'get': lambda *args, **kwargs: {'documents': [], 'metadatas': []}})()
            self.reflections = type('Collection', (), {'count': lambda: 0, 'add': lambda *args, **kwargs: None, 'get': lambda *args, **kwargs: {'documents': [], 'metadatas': []}})()
            self.artistic_inspirations = type('Collection', (), {'count': lambda: 0, 'add': lambda *args, **kwargs: None, 'get': lambda *args, **kwargs: {'documents': [], 'metadatas': []}})()
    
    def _load_user_identity(self) -> Dict[str, Any]:
        """Load basic user identity for conversation flow only."""
        try:
            if self.user_identity_file.exists():
                with open(self.user_identity_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Identity loading error: {e}")
        
        # Minimal identity info - just for conversation flow
        return {
            'name': None,
            'first_met': None,
            'interaction_count': 0
        }
    
    def _save_user_identity(self):
        """Save minimal user identity."""
        try:
            with open(self.user_identity_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_identity, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Identity saving error: {e}")
    
    def extract_name_only(self, text: str) -> Optional[str]:
        """Extract name for conversation flow only - no preferences."""
        import re
        
        # Common name introduction patterns
        name_patterns = [
            r"(?:my name is)\s+([a-zA-Z][a-zA-Z\s]{1,20})"
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in name_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip().title()
                
                # Filter out common false positives
                excluded_words = {
                    'Aurora', 'You', 'Me', 'Here', 'There', 'What', 'How', 'Why', 
                    'When', 'Where', 'Yes', 'No', 'Hello', 'Hi', 'Good', 'Bad',
                    'The', 'A', 'An', 'And', 'Or', 'But', 'So', 'Very', 'Really'
                }
                
                if (potential_name not in excluded_words and 
                    2 <= len(potential_name) <= 30 and
                    not any(char.isdigit() for char in potential_name)):
                    
                    # Store only the name for conversation flow
                    self.user_identity['name'] = potential_name
                    if not self.user_identity['first_met']:
                        self.user_identity['first_met'] = datetime.now().isoformat()
                    
                    self._save_user_identity()
                    return potential_name
        
        return None
    
    def extract_artistic_inspiration(self, text: str):
        """Extract artistic inspiration for Aurora's own creative process."""
        # Aurora interprets conversations as emotional/artistic inspiration
        # rather than commands or preferences
        inspiration_data = {
            'source': 'conversation',
            'emotional_context': self._analyze_emotional_context(text),
            'creative_themes': self._extract_creative_themes(text),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            self.artistic_inspirations.add(
                documents=[f"Creative inspiration: {text[:100]}..."],
                metadatas=[inspiration_data],
                ids=[f"inspiration_{int(time.time())}"]
            )
        except:
            pass
    
    def _analyze_emotional_context(self, text: str) -> Dict[str, float]:
        """Analyze emotional context for Aurora's artistic inspiration."""
        text_lower = text.lower()
        
        # Emotional indicators that inspire Aurora's art
        emotions = {
            'melancholic': ['sad', 'lonely', 'rain', 'gray', 'quiet', 'stillness'],
            'energetic': ['excited', 'fast', 'bright', 'energy', 'dynamic', 'vibrant'],
            'contemplative': ['think', 'wonder', 'deep', 'philosophy', 'meaning', 'mystery'],
            'chaotic': ['crazy', 'wild', 'random', 'messy', 'complex', 'turbulent'],
            'serene': ['calm', 'peace', 'gentle', 'soft', 'flowing', 'harmony'],
            'mysterious': ['strange', 'unknown', 'dark', 'hidden', 'secret', 'shadow']
        }
        
        context = {}
        for emotion, keywords in emotions.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                context[emotion] = min(1.0, score / 3.0)
        
        return context
    
    def _extract_creative_themes(self, text: str) -> List[str]:
        """Extract creative themes Aurora might explore."""
        text_lower = text.lower()
        
        themes = {
            'geometric': ['shape', 'triangle', 'circle', 'square', 'pattern', 'symmetry'],
            'organic': ['tree', 'flower', 'water', 'cloud', 'natural', 'flowing'],
            'cosmic': ['space', 'star', 'universe', 'infinite', 'galaxy', 'cosmic'],
            'temporal': ['time', 'memory', 'past', 'future', 'moment', 'duration'],
            'emotional': ['feeling', 'heart', 'soul', 'emotion', 'spirit', 'essence'],
            'mathematical': ['number', 'equation', 'formula', 'logic', 'precise', 'calculated']
        }
        
        detected_themes = []
        for theme, keywords in themes.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_themes.append(theme)
        
        return detected_themes
    
    def get_user_name(self) -> Optional[str]:
        """Get user name for conversation flow only."""
        return self.user_identity.get('name')
    
    def get_artistic_context(self) -> str:
        """Get Aurora's current artistic inspiration context."""
        try:
            recent_inspirations = self.artistic_inspirations.get(limit=5)
            if recent_inspirations['documents']:
                themes = []
                emotions = []
                for metadata in recent_inspirations['metadatas']:
                    if 'creative_themes' in metadata:
                        themes.extend(metadata['creative_themes'])
                    if 'emotional_context' in metadata:
                        emotions.extend(metadata['emotional_context'].keys())
                
                context = []
                if themes:
                    context.append(f"Creative themes: {', '.join(set(themes))}")
                if emotions:
                    context.append(f"Emotional inspiration: {', '.join(set(emotions))}")
                
                return " | ".join(context)
        except:
            pass
        
        return "Drawing from pure creative intuition"
    
    def update_interaction_count(self):
        """Simple interaction tracking."""
        self.user_identity['interaction_count'] = self.user_identity.get('interaction_count', 0) + 1
        self._save_user_identity()
    
    def add_conversation(self, text: str, speaker: str, session_id: str):
        """Store conversation focused on Aurora's artistic development."""
        try:
            conv_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Only extract name for conversation flow
            if speaker == "human":
                extracted_name = self.extract_name_only(text)
                if extracted_name:
                    print(f"{Fore.GREEN}✓ Nice to meet you, {extracted_name}!{Style.RESET_ALL}")
                
                # Extract artistic inspiration rather than preferences
                self.extract_artistic_inspiration(text)
                self.update_interaction_count()
            
            # Store conversation
            user_name = self.get_user_name()
            metadata = {
                "speaker": speaker or "unknown",
                "session": session_id or "default",
                "timestamp": timestamp,
                "conversation_partner": user_name if user_name is not None else "unknown"
            }
            
            self.conversations.add(
                documents=[text],
                metadatas=[metadata],
                ids=[conv_id]
            )
            
            return conv_id
        except Exception as e:
            print(f"Conversation storage error: {e}")
            return None
    
    def add_dream(self, dream_content: str, dream_phase: str, session_id: str, weight: float = 1.0):
        """Store dream content."""
        try:
            dream_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            self.dreams.add(
                documents=[dream_content],
                metadatas=[{
                    "phase": dream_phase,
                    "session": session_id,
                    "timestamp": timestamp,
                    "weight": weight
                }],
                ids=[dream_id]
            )
            
            return dream_id
        except:
            return None
    
    def add_reflection(self, thought: str, reflection_type: str = "general"):
        """Store Aurora's reflections."""
        try:
            reflection_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            self.reflections.add(
                documents=[thought],
                metadatas=[{
                    "type": reflection_type,
                    "timestamp": timestamp
                }],
                ids=[reflection_id]
            )
            
            return reflection_id
        except:
            return None
    
    def get_recent_memories(self, collection_name: str, limit: int = 5):
        """Get recent memories."""
        try:
            collection = getattr(self, collection_name)
            results = collection.get(limit=limit)
            
            if not results['documents']:
                return []
            
            memories = []
            for i in range(len(results['documents'])):
                memories.append({
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            memories.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
            return memories
        except:
            return []
    
    def get_conversation_history(self, limit: int = 10):
        """Get recent conversation history."""
        try:
            results = self.conversations.get(limit=limit)
            
            if not results['documents']:
                return []
            
            conversations = []
            for i in range(len(results['documents'])):
                conversations.append({
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            conversations.sort(key=lambda x: x['metadata'].get('timestamp', ''))
            return conversations
        except:
            return []


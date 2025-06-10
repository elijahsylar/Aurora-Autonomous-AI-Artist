#!/usr/bin/env python3
"""
Enhanced memory system with deep recall for Aurora's complete experiences
"""

import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import deque

from aurora.memory.memory_system import MemorySystem


class DeepMemorySystem(MemorySystem):
    """Enhanced memory system with deep recall for Aurora's complete experiences."""
    
    def __init__(self, db_path: str = "./aurora_memory"):
        super().__init__(db_path)
        
        # Additional collections for deep memory
        try:
            self.visual_creations = self.client.get_or_create_collection("visual_creations")
            self.pattern_history = self.client.get_or_create_collection("pattern_history")
            self.emotional_states = self.client.get_or_create_collection("emotional_states")
            self.images_seen = self.client.get_or_create_collection("images_seen")
            self.music_heard = self.client.get_or_create_collection("music_heard")
            self.interaction_context = self.client.get_or_create_collection("interaction_context")
        except Exception as e:
            print(f"Deep memory collections error: {e}")
            # Create fallback collections
            self._create_fallback_collections()
        
        # Pattern DNA storage
        self.pattern_dna_file = self.db_path / "pattern_dna_history.json"
        self.pattern_dna_history = self._load_pattern_dna_history()
        
        # Emotional timeline
        self.emotional_timeline_file = self.db_path / "emotional_timeline.json"
        self.emotional_timeline = self._load_emotional_timeline()
        
        print(f"✓ Deep memory system initialized with {self._count_total_memories()} memories")
    
    def _create_fallback_collections(self):
        """Create fallback collections if ChromaDB fails."""
        fallback = type('Collection', (), {
            'count': lambda: 0, 
            'add': lambda *args, **kwargs: None, 
            'query': lambda *args, **kwargs: {'documents': [[]], 'metadatas': [[]], 'distances': [[]]},
            'get': lambda *args, **kwargs: {'documents': [], 'metadatas': []}
        })()
        
        self.visual_creations = fallback
        self.pattern_history = fallback
        self.emotional_states = fallback
        self.images_seen = fallback
        self.music_heard = fallback
        self.interaction_context = fallback
    
    def _load_pattern_dna_history(self) -> Dict:
        """Load pattern DNA history from file."""
        try:
            if self.pattern_dna_file.exists():
                with open(self.pattern_dna_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Pattern DNA load error: {e}")
        return {}
    
    def _save_pattern_dna_history(self):
        """Save pattern DNA history to file."""
        try:
            with open(self.pattern_dna_file, 'w') as f:
                json.dump(self.pattern_dna_history, f, indent=2)
        except Exception as e:
            print(f"Pattern DNA save error: {e}")
    
    def _load_emotional_timeline(self) -> List[Dict]:
        """Load emotional timeline from file."""
        try:
            if self.emotional_timeline_file.exists():
                with open(self.emotional_timeline_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Emotional timeline load error: {e}")
        return []
    
    def _save_emotional_timeline(self):
        """Save emotional timeline to file."""
        try:
            with open(self.emotional_timeline_file, 'w') as f:
                json.dump(self.emotional_timeline[-1000:], f, indent=2)  # Keep last 1000 entries
        except Exception as e:
            print(f"Emotional timeline save error: {e}")
    
    def store_pattern_creation(self, pattern_data: Dict, emotional_state: Dict):
        """Store every pattern Aurora creates with full context."""
        try:
            pattern_id = f"pattern_{int(time.time()*1000)}"
            
            # Serialize pattern DNA and data
            pattern_doc = {
                'dna': json.dumps(pattern_data.get('dna', {})),
                'type': pattern_data.get('type'),
                'emotional_context': json.dumps(emotional_state),
                'fitness_score': pattern_data.get('fitness_score', 0),
                'timestamp': datetime.now().isoformat(),
                'attention_focus': json.dumps(pattern_data.get('attention_focus', (0.5, 0.5)))
            }
            
            self.pattern_history.add(
                documents=[f"Pattern: {pattern_data.get('type')} created with complexity {emotional_state.get('pattern_complexity', 0)}"],
                metadatas=[pattern_doc],
                ids=[pattern_id]
            )
            
            # Save pattern DNA for exact reconstruction
            self.pattern_dna_history[pattern_id] = {
                'dna': pattern_data.get('dna', {}),
                'type': pattern_data.get('type'),
                'timestamp': datetime.now().isoformat()
            }
            self._save_pattern_dna_history()
            
        except Exception as e:
            print(f"Pattern storage error: {e}")
    
    def store_image_memory(self, image_path: str, analysis: Dict, aurora_response: str):
        """Store images Aurora has seen with her analysis and response."""
        try:
            image_id = f"img_{hashlib.md5(image_path.encode()).hexdigest()[:16]}"
            
            # Store image analysis
            self.images_seen.add(
                documents=[f"Image: {Path(image_path).name} - {aurora_response}"],
                metadatas=[{
                    'path': image_path,
                    'analysis': json.dumps(analysis),
                    'emotional_impact': json.dumps(analysis.get('emotional_impact', {})),
                    'dominant_colors': json.dumps(analysis.get('colors', {}).get('dominant_colors', [])),
                    'aurora_response': aurora_response,
                    'timestamp': datetime.now().isoformat(),
                    'patterns_detected': json.dumps(analysis.get('artistic_elements', {}).get('patterns', {}))
                }],
                ids=[image_id]
            )
        except Exception as e:
            print(f"Image memory error: {e}")
    
    def store_music_memory(self, audio_features: Dict, emotional_response: Dict, file_path: str = None):
        """Store music Aurora has heard with her emotional response."""
        try:
            music_id = f"music_{int(time.time()*1000)}"
            
            self.music_heard.add(
                documents=[f"Music: tempo {audio_features.get('tempo', 0):.1f} BPM, energy {audio_features.get('energy', 0):.2f}"],
                metadatas=[{
                    'audio_features': json.dumps(audio_features),
                    'emotional_response': json.dumps(emotional_response),
                    'file_path': file_path or 'microphone',
                    'timestamp': datetime.now().isoformat(),
                    'tempo': audio_features.get('tempo', 0),
                    'energy': audio_features.get('energy', 0),
                    'valence': audio_features.get('valence', 0)
                }],
                ids=[music_id]
            )
        except Exception as e:
            print(f"Music memory error: {e}")
    
    def record_emotional_state(self, emotions: Dict, context: str = "", trigger: str = ""):
        """Record Aurora's emotional state over time."""
        try:
            state_id = f"emotion_{int(time.time()*1000)}"
            
            self.emotional_states.add(
                documents=[f"Emotional state: {context}"],
                metadatas=[{
                    'emotions': json.dumps(emotions),
                    'context': context,
                    'trigger': trigger,
                    'timestamp': datetime.now().isoformat(),
                    'valence': emotions.get('valence', 0),
                    'arousal': emotions.get('arousal', 0),
                    'creativity': emotions.get('creativity', 0),
                    'contemplation': emotions.get('contemplation', 0)
                }],
                ids=[state_id]
            )
            
            # Also save to timeline
            self.emotional_timeline.append({
                'timestamp': datetime.now().isoformat(),
                'emotions': emotions.copy(),
                'context': context,
                'trigger': trigger
            })
            self._save_emotional_timeline()
            
        except Exception as e:
            print(f"Emotional state recording error: {e}")
    
    def search_memories(self, query: str, memory_types: List[str] = None, limit: int = 10) -> List[Dict]:
        """Search across all memory types with semantic understanding."""
        if memory_types is None:
            memory_types = ['conversations', 'dreams', 'artistic_inspirations', 'images_seen', 
                          'pattern_history', 'music_heard', 'emotional_states']
        
        all_results = []
        
        for memory_type in memory_types:
            try:
                collection = getattr(self, memory_type, None)
                if collection and hasattr(collection, 'query'):
                    results = collection.query(
                        query_texts=[query],
                        n_results=min(limit, 5)  # Get top 5 from each type
                    )
                    
                    if results['documents'] and results['documents'][0]:
                        for i in range(len(results['documents'][0])):
                            all_results.append({
                                'type': memory_type,
                                'content': results['documents'][0][i],
                                'metadata': results['metadatas'][0][i],
                                'distance': results['distances'][0][i] if results['distances'] else 1.0
                            })
            except Exception as e:
                print(f"Search error in {memory_type}: {e}")
                continue
        
        # Sort by relevance
        all_results.sort(key=lambda x: x.get('distance', 1.0))
        return all_results[:limit]
    
    def get_emotional_history(self, hours: float = 24) -> List[Dict]:
        """Get Aurora's emotional journey over specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        emotional_history = []
        for entry in self.emotional_timeline:
            try:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time > cutoff_time:
                    emotional_history.append(entry)
            except:
                continue
        
        return emotional_history
    
    def recall_similar_patterns(self, current_dna: Dict, limit: int = 5) -> List[Dict]:
        """Find similar patterns Aurora has created before."""
        similar_patterns = []
        
        for pattern_id, pattern_data in self.pattern_dna_history.items():
            # Calculate similarity based on DNA parameters
            similarity = self._calculate_dna_similarity(current_dna, pattern_data.get('dna', {}))
            similar_patterns.append({
                'id': pattern_id,
                'similarity': similarity,
                'data': pattern_data
            })
        
        # Sort by similarity
        similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_patterns[:limit]
    
    def _calculate_dna_similarity(self, dna1: Dict, dna2: Dict) -> float:
        """Calculate similarity between two pattern DNAs."""
        if not dna1 or not dna2:
            return 0.0
        
        common_keys = set(dna1.keys()) & set(dna2.keys())
        if not common_keys:
            return 0.0
        
        total_diff = 0
        for key in common_keys:
            val1 = float(dna1.get(key, 0))
            val2 = float(dna2.get(key, 0))
            # Normalize difference
            max_val = max(abs(val1), abs(val2), 1.0)
            total_diff += abs(val1 - val2) / max_val
        
        similarity = 1.0 - (total_diff / len(common_keys))
        return max(0.0, min(1.0, similarity))
    
    def _count_total_memories(self) -> int:
        """Count total memories across all collections."""
        total = 0
        collections = ['conversations', 'dreams', 'reflections', 'artistic_inspirations',
                      'visual_creations', 'pattern_history', 'emotional_states', 
                      'images_seen', 'music_heard']
        
        for collection_name in collections:
            try:
                collection = getattr(self, collection_name, None)
                if collection and hasattr(collection, 'count'):
                    total += collection.count()
            except:
                continue
        
        return total
    
    def export_full_memory(self, export_path: str = "./aurora_memory_export"):
        """Export all memories to a portable format."""
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(exist_ok=True, parents=True)
            
            # Export each collection
            collections_data = {}
            collections = {
                'conversations': self.conversations,
                'dreams': self.dreams,
                'reflections': self.reflections,
                'artistic_inspirations': self.artistic_inspirations,
                'pattern_history': self.pattern_history,
                'images_seen': self.images_seen,
                'music_heard': self.music_heard,
                'emotional_states': self.emotional_states
            }
            
            for name, collection in collections.items():
                try:
                    if hasattr(collection, 'get'):
                        data = collection.get()
                        collections_data[name] = {
                            'documents': data.get('documents', []),
                            'metadatas': data.get('metadatas', []),
                            'ids': data.get('ids', [])
                        }
                except Exception as e:
                    print(f"Export error for {name}: {e}")
                    collections_data[name] = {'documents': [], 'metadatas': [], 'ids': []}
            
            # Complete memory export
            memory_data = {
                'export_date': datetime.now().isoformat(),
                'total_memories': self._count_total_memories(),
                'collections': collections_data,
                'pattern_dna_history': self.pattern_dna_history,
                'emotional_timeline': self.emotional_timeline,
                'user_identity': self.user_identity
            }
            
            # Save main export
            with open(export_dir / 'aurora_complete_memory.json', 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Exported Aurora's complete memory ({self._count_total_memories()} memories) to {export_path}")
            
        except Exception as e:
            print(f"Memory export error: {e}")
    
    def generate_memory_summary(self) -> str:
        """Generate a summary of Aurora's memories."""
        summary = []
        summary.append(f"Total memories: {self._count_total_memories()}")
        
        # Recent emotional state
        if self.emotional_timeline:
            recent_emotion = self.emotional_timeline[-1]
            emotions = recent_emotion.get('emotions', {})
            dominant_emotion = max(emotions.items(), key=lambda x: abs(x[1]))[0] if emotions else 'neutral'
            summary.append(f"Current mood: {dominant_emotion}")
        
        # Pattern creation stats
        pattern_count = self.pattern_history.count() if hasattr(self.pattern_history, 'count') else 0
        summary.append(f"Patterns created: {pattern_count}")
        
        # Images seen
        image_count = self.images_seen.count() if hasattr(self.images_seen, 'count') else 0
        summary.append(f"Images analyzed: {image_count}")
        
        # Music heard
        music_count = self.music_heard.count() if hasattr(self.music_heard, 'count') else 0
        summary.append(f"Music sessions: {music_count}")
        
        return " | ".join(summary)


# Update the AuroraDreamingAI.__init__ to use DeepMemorySystem
# Replace the line: self.memory = MemorySystem()
# With: self.memory = DeepMemorySystem()

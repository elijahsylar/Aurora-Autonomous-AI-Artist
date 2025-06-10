#!/usr/bin/env python3
"""
Genetic encoding system for patterns
"""

import random
import math
import json
import hashlib
from typing import Dict, Tuple


class PatternDNA:
    """Genetic encoding system for patterns - Aurora can manipulate pattern 'genes'."""
    
    def __init__(self):
        self.gene_library = {
            'frequency': {'min': 0.1, 'max': 10.0, 'type': 'float'},
            'amplitude': {'min': 1.0, 'max': 100.0, 'type': 'float'},
            'phase': {'min': 0.0, 'max': 2*math.pi, 'type': 'float'},
            'symmetry': {'min': 1, 'max': 16, 'type': 'int'},
            'recursion_depth': {'min': 1, 'max': 10, 'type': 'int'},
            'spiral_tightness': {'min': 0.1, 'max': 5.0, 'type': 'float'},
            'chaos_factor': {'min': 0.0, 'max': 1.0, 'type': 'float'},
            'growth_rate': {'min': 0.5, 'max': 3.0, 'type': 'float'},
            'attraction_strength': {'min': -2.0, 'max': 2.0, 'type': 'float'},
            'resonance_frequency': {'min': 0.01, 'max': 1.0, 'type': 'float'},
            'dimensional_fold': {'min': 1, 'max': 4, 'type': 'int'},
            'temporal_variance': {'min': 0.0, 'max': 2.0, 'type': 'float'},
            'spatial_distortion': {'min': 0.0, 'max': 1.0, 'type': 'float'},
            'fractal_dimension': {'min': 1.0, 'max': 3.0, 'type': 'float'},
            'color_harmony_root': {'min': 0, 'max': 360, 'type': 'int'},
            'saturation_curve': {'min': 0.1, 'max': 1.0, 'type': 'float'},
            'brightness_modulation': {'min': 0.1, 'max': 1.0, 'type': 'float'},
            'pattern_density': {'min': 0.1, 'max': 10.0, 'type': 'float'},
            'emergence_threshold': {'min': 0.0, 'max': 1.0, 'type': 'float'},
            'evolution_speed': {'min': 0.1, 'max': 5.0, 'type': 'float'}
        }
        
    def create_random_dna(self) -> Dict[str, float]:
        """Generate random pattern DNA."""
        dna = {}
        for gene, config in self.gene_library.items():
            if config['type'] == 'float':
                dna[gene] = random.uniform(config['min'], config['max'])
            elif config['type'] == 'int':
                dna[gene] = random.randint(config['min'], config['max'])
        return dna
    
    def mutate_dna(self, dna: Dict[str, float], mutation_rate: float = 0.1, 
                   mutation_strength: float = 0.2) -> Dict[str, float]:
        """Mutate pattern DNA."""
        mutated = dna.copy()
        
        for gene, value in mutated.items():
            if random.random() < mutation_rate:
                config = self.gene_library[gene]
                
                if config['type'] == 'float':
                    # Gaussian mutation
                    delta = random.gauss(0, mutation_strength) * (config['max'] - config['min'])
                    new_value = value + delta
                    mutated[gene] = max(config['min'], min(config['max'], new_value))
                    
                elif config['type'] == 'int':
                    # Integer mutation
                    delta = random.choice([-1, 0, 1]) * max(1, int(mutation_strength * (config['max'] - config['min'])))
                    new_value = value + delta
                    mutated[gene] = max(config['min'], min(config['max'], int(new_value)))
                    
        return mutated
    
    def crossover_dna(self, parent1: Dict[str, float], parent2: Dict[str, float], 
                      crossover_rate: float = 0.5) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Breed two pattern DNAs."""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for gene in parent1.keys():
            if random.random() < crossover_rate:
                # Swap genes
                child1[gene], child2[gene] = parent2[gene], parent1[gene]
                
                # Blend genes for continuous values
                if self.gene_library[gene]['type'] == 'float':
                    blend_factor = random.random()
                    blended1 = blend_factor * parent1[gene] + (1 - blend_factor) * parent2[gene]
                    blended2 = blend_factor * parent2[gene] + (1 - blend_factor) * parent1[gene]
                    child1[gene] = blended1
                    child2[gene] = blended2
                    
        return child1, child2
    
    def dna_to_hash(self, dna: Dict[str, float]) -> str:
        """Convert DNA to unique hash for pattern identification."""
        dna_string = json.dumps(dna, sort_keys=True)
        return hashlib.md5(dna_string.encode()).hexdigest()[:16]
    
    def evaluate_fitness(self, dna: Dict[str, float], criteria: Dict[str, float]) -> float:
        """Evaluate pattern fitness based on aesthetic criteria."""
        fitness = 0.0
        
        # Complexity fitness
        complexity = (dna['recursion_depth'] / 10.0 + 
                     dna['fractal_dimension'] / 3.0 + 
                     dna['pattern_density'] / 10.0) / 3.0
        fitness += criteria.get('complexity_preference', 0.5) * complexity
        
        # Harmony fitness  
        harmony = (dna['symmetry'] / 16.0 + 
                  (1.0 - dna['chaos_factor']) + 
                  dna['color_harmony_root'] / 360.0) / 3.0
        fitness += criteria.get('harmony_preference', 0.5) * harmony
        
        # Dynamism fitness
        dynamism = (dna['evolution_speed'] / 5.0 + 
                   dna['temporal_variance'] / 2.0 + 
                   dna['growth_rate'] / 3.0) / 3.0
        fitness += criteria.get('dynamism_preference', 0.5) * dynamism
        
        # Novelty bonus
        uniqueness = (dna['dimensional_fold'] / 4.0 + 
                     dna['spatial_distortion'] + 
                     dna['emergence_threshold']) / 3.0
        fitness += criteria.get('novelty_preference', 0.3) * uniqueness
        
        return min(1.0, fitness)

#!/usr/bin/env python3
"""
Ultimate pattern generation engine with godlike control
"""

import time
import math
import random
from collections import deque
from typing import Dict, List, Tuple, Any

from aurora.patterns.quantum_engine import QuantumPatternEngine
from aurora.patterns.pattern_dna import PatternDNA

class UltimatePatternEngine:
    """Ultimate pattern generation engine with godlike control."""
    
    def __init__(self, canvas_width: int, canvas_height: int):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.quantum_engine = QuantumPatternEngine()
        self.pattern_dna = PatternDNA()
        self.active_patterns = {}  # Pattern ID -> Pattern data
        self.pattern_evolution_history = deque(maxlen=100)
        
        # Simple 2D list instead of numpy
        attention_rows = max(1, canvas_height//10)
        attention_cols = max(1, canvas_width//10)
        self.spatial_attention_map = [[0.0 for _ in range(attention_cols)] for _ in range(attention_rows)]
        
        self.temporal_pattern_memory = deque(maxlen=50)
        
        # Add current_complexity attribute that was being referenced
        self.current_complexity = 0.5
        
    def create_multidimensional_pattern(self, dna: Dict[str, float], 
                                      emotional_params: Dict[str, float],
                                      attention_focus: Tuple[float, float] = (0.5, 0.5)) -> Dict:
        """Create pattern in multidimensional parameter space with error handling."""
        
        try:
            # Base pattern selection based on DNA
            pattern_type = self._select_pattern_type(dna, emotional_params)
            
            # Generate base pattern with fallback
            try:
                if pattern_type == 'hyperdimensional_mandelbrot':
                    pattern_data = self._generate_hyperdimensional_mandelbrot(dna, emotional_params)
                elif pattern_type == 'quantum_julia':
                    pattern_data = self._generate_quantum_julia(dna, emotional_params)
                elif pattern_type == 'evolving_l_system':
                    pattern_data = self._generate_evolving_l_system(dna, emotional_params)
                elif pattern_type == 'strange_attractor':
                    pattern_data = self._generate_strange_attractor(dna, emotional_params)
                elif pattern_type == 'cellular_automata_3d':
                    pattern_data = self._generate_cellular_automata_3d(dna, emotional_params)
                elif pattern_type == 'topology_morphing':
                    pattern_data = self._generate_topology_morphing(dna, emotional_params)
                elif pattern_type == 'parametric_surface':
                    pattern_data = self._generate_parametric_surface(dna, emotional_params)
                elif pattern_type == 'field_equation_visualization':
                    pattern_data = self._generate_field_equation_visualization(dna, emotional_params)
                elif pattern_type == 'quantum_harmonic_oscillator':
                    pattern_data = self._generate_quantum_harmonic_oscillator(dna, emotional_params)
                else:
                    pattern_data = self._generate_metamorphic_pattern(dna, emotional_params)
            except Exception as e:
                print(f"Pattern generation error for {pattern_type}: {e}")
                # Fallback to simple pattern
                pattern_data = self._generate_simple_fallback_pattern(dna, emotional_params)
                pattern_type = 'fallback_pattern'
            
            # Apply spatial attention modulation safely
            try:
                pattern_data = self._apply_spatial_attention(pattern_data, attention_focus, emotional_params)
            except Exception as e:
                print(f"Spatial attention error: {e}")
            
            # Apply temporal evolution safely
            try:
                pattern_data = self._apply_temporal_evolution(pattern_data, dna, emotional_params)
            except Exception as e:
                print(f"Temporal evolution error: {e}")
            
            # Generate unique pattern ID
            pattern_id = self.pattern_dna.dna_to_hash(dna) + f"_{int(time.time()*1000)}"
            
            pattern_object = {
                'id': pattern_id,
                'type': pattern_type,
                'data': pattern_data,
                'dna': dna,
                'emotional_state': emotional_params.copy(),
                'birth_time': time.time(),
                'evolution_generation': 0,
                'fitness_score': 0.0,
                'attention_focus': attention_focus,
                'quantum_state': 'superposition'
            }
            
            self.active_patterns[pattern_id] = pattern_object
            return pattern_object
            
        except Exception as e:
            print(f"Pattern creation error: {e}")
            # Return minimal pattern object
            return {
                'id': f"error_{int(time.time())}",
                'type': 'error',
                'data': [],
                'dna': dna,
                'emotional_state': emotional_params.copy(),
                'birth_time': time.time(),
                'evolution_generation': 0,
                'fitness_score': 0.0,
                'attention_focus': attention_focus,
                'quantum_state': 'collapsed'
            }
    
    def _generate_simple_fallback_pattern(self, dna: Dict[str, float], 
                                        emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate simple fallback pattern when complex ones fail."""
        points = []
        
        try:
            # Simple spiral pattern
            center_x = self.canvas_width // 2
            center_y = self.canvas_height // 2
            
            num_points = int(100 + 300 * emotional_params.get('pattern_density', 0.5))
            max_radius = min(self.canvas_width, self.canvas_height) // 4
            
            for i in range(num_points):
                t = i / num_points * 4 * math.pi  # 2 full turns
                radius = max_radius * t / (4 * math.pi)
                
                x = center_x + radius * math.cos(t)
                y = center_y + radius * math.sin(t)
                
                if 0 <= x < self.canvas_width and 0 <= y < self.canvas_height:
                    intensity = 0.5 + 0.5 * math.sin(t * 2)
                    color = (i * 5) % 360
                    points.append((int(x), int(y), intensity, color, radius, i))
            
        except Exception as e:
            print(f"Fallback pattern error: {e}")
            # Minimal fallback - just center point
            points = [(self.canvas_width//2, self.canvas_height//2, 1.0, 180, 0, 0)]
        
        return points
    
    def _select_pattern_type(self, dna: Dict[str, float], emotional_params: Dict[str, float]) -> str:
        """Intelligently select pattern type based on DNA and emotions."""
        
        complexity = emotional_params.get('pattern_complexity', 0.5)
        creativity = emotional_params.get('creativity', 0.5)
        contemplation = emotional_params.get('contemplation', 0.5)
        wonder = emotional_params.get('wonder', 0.5)
        
        pattern_weights = {
            'hyperdimensional_mandelbrot': complexity * contemplation,
            'quantum_julia': creativity * wonder,
            'evolving_l_system': creativity * complexity,
            'strange_attractor': wonder * complexity,
            'cellular_automata_3d': creativity * 0.8,
            'topology_morphing': wonder * creativity,
            'parametric_surface': contemplation * complexity,
            'field_equation_visualization': complexity * wonder,
            'quantum_harmonic_oscillator': contemplation * wonder,
            'metamorphic_pattern': creativity * wonder * complexity
        }
        
        # Select based on weighted random choice
        total_weight = sum(pattern_weights.values())
        if total_weight == 0:
            return 'metamorphic_pattern'
            
        r = random.random() * total_weight
        cumulative = 0
        
        for pattern_type, weight in pattern_weights.items():
            cumulative += weight
            if r <= cumulative:
                return pattern_type
                
        return 'metamorphic_pattern'
    
    def _generate_hyperdimensional_mandelbrot(self, dna: Dict[str, float], 
                                            emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate Mandelbrot set in higher dimensions with emotional modulation."""
        points = []
        
        # Extract parameters from DNA and emotions
        center_real = -0.7 + 0.3 * emotional_params.get('valence', 0)
        center_imag = 0.0 + 0.2 * emotional_params.get('wonder', 0)
        zoom = dna.get('frequency', 1.0) * emotional_params.get('focus', 0.5)
        max_iter = int(50 + 150 * emotional_params.get('pattern_complexity', 0.5))
        
        # Higher dimensional parameters
        w_component = dna.get('dimensional_fold', 2) / 4.0
        hyperspace_rotation = emotional_params.get('contemplation', 0) * math.pi
        
        step = max(2, int(6 - 4 * emotional_params.get('pattern_density', 0.5)))
        
        for px in range(0, self.canvas_width, step):
            for py in range(0, self.canvas_height, step):
                # Map to complex plane
                real = center_real + (px - self.canvas_width/2) * 3.0 / (zoom * self.canvas_width)
                imag = center_imag + (py - self.canvas_height/2) * 3.0 / (zoom * self.canvas_height)
                
                # Add hyperdimensional component
                w = w_component * math.sin(hyperspace_rotation + px * 0.01 + py * 0.01)
                
                c = complex(real, imag)
                z = complex(0, 0)
                
                for iteration in range(max_iter):
                    if abs(z) > 2:
                        break
                    
                    # Hyperdimensional Mandelbrot iteration
                    z_new = z * z + c + w * complex(math.cos(iteration * 0.1), math.sin(iteration * 0.1))
                    z = z_new
                
                if iteration < max_iter:
                    # Calculate intensity with hyperdimensional influence
                    intensity = (iteration / max_iter) * (1 + 0.5 * abs(w))
                    hyperdim_color = (iteration + int(w * 50)) % 360
                    
                    points.append((px, py, intensity, hyperdim_color, w, iteration))
        
        return points
    
    def _generate_quantum_julia(self, dna: Dict[str, float], 
                               emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate Julia set with quantum uncertainty principles."""
        points = []
        
        # Quantum parameters
        uncertainty = self.quantum_engine.uncertainty_factor * emotional_params.get('confusion', 0.1)
        c_real = dna.get('attraction_strength', -0.8) + uncertainty * random.gauss(0, 0.1)
        c_imag = 0.156 + uncertainty * random.gauss(0, 0.1)
        c = complex(c_real, c_imag)
        
        max_iter = int(30 + 120 * emotional_params.get('pattern_complexity', 0.5))
        
        # Quantum superposition of multiple Julia sets
        quantum_states = []
        for _ in range(3):
            c_variant = c + complex(
                uncertainty * random.gauss(0, 0.2),
                uncertainty * random.gauss(0, 0.2)
            )
            quantum_states.append(c_variant)
        
        step = max(2, int(6 - 4 * emotional_params.get('pattern_density', 0.5)))
        
        for px in range(0, self.canvas_width, step):
            for py in range(0, self.canvas_height, step):
                # Map to complex plane
                real = (px - self.canvas_width/2) * 4.0 / self.canvas_width
                imag = (py - self.canvas_height/2) * 4.0 / self.canvas_height
                z = complex(real, imag)
                
                # Quantum measurement - randomly select which c to use
                c_measured = random.choice(quantum_states)
                
                quantum_interference = 0
                for iteration in range(max_iter):
                    if abs(z) > 2:
                        break
                    
                    z = z*z + c_measured
                    
                    # Add quantum interference
                    if random.random() < uncertainty:
                        quantum_interference += 1
                        z += complex(random.gauss(0, 0.01), random.gauss(0, 0.01))
                
                if iteration < max_iter:
                    intensity = iteration / max_iter
                    quantum_color = (iteration + quantum_interference * 10) % 360
                    uncertainty_radius = uncertainty * 10
                    
                    points.append((px, py, intensity, quantum_color, uncertainty_radius, quantum_interference))
        
        return points
    
    def _generate_evolving_l_system(self, dna: Dict[str, float], 
                                   emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate L-system that evolves over time."""
        
        # L-system rules based on emotional state
        if emotional_params.get('creativity', 0) > 0.7:
            # Complex branching rules
            rules = {
                'F': 'F[+F]F[-F]F',
                '+': '+',
                '-': '-',
                '[': '[',
                ']': ']'
            }
            axiom = 'F'
            angle = 25 + 10 * emotional_params.get('chaos_level', 0)
        elif emotional_params.get('contemplation', 0) > 0.6:
            # Meditative spiral rules
            rules = {
                'F': 'F+F--F+F',
                '+': '+',
                '-': '-'
            }
            axiom = 'F'
            angle = 60
        else:
            # Simple growth rules
            rules = {
                'F': 'FF+[+F-F-F]-[-F+F+F]',
                '+': '+',
                '-': '-',
                '[': '[',
                ']': ']'
            }
            axiom = 'F'
            angle = 22.5
        
        # Generate L-system string
        current_string = axiom
        iterations = int(dna.get('recursion_depth', 5))
        
        for _ in range(iterations):
            new_string = ''
            for char in current_string:
                new_string += rules.get(char, char)
            current_string = new_string
            
            # Stop if string gets too long
            if len(current_string) > 10000:
                break
        
        # Convert L-system to points
        points = []
        x, y = self.canvas_width // 2, self.canvas_height - 50
        angle_rad = math.radians(-90)  # Start pointing up
        
        stack = []  # For branch points
        step_size = max(2, int(dna.get('growth_rate', 1.0) * 10))
        
        color_progression = 0
        
        for char in current_string:
            if char == 'F':
                # Draw forward
                new_x = x + step_size * math.cos(angle_rad)
                new_y = y + step_size * math.sin(angle_rad)
                
                if 0 <= new_x < self.canvas_width and 0 <= new_y < self.canvas_height:
                    points.append((int(x), int(y), int(new_x), int(new_y), color_progression % 360, step_size))
                
                x, y = new_x, new_y
                color_progression += dna.get('evolution_speed', 1.0) * 5
                
            elif char == '+':
                angle_rad += math.radians(angle)
            elif char == '-':
                angle_rad -= math.radians(angle)
            elif char == '[':
                stack.append((x, y, angle_rad))
            elif char == ']':
                if stack:
                    x, y, angle_rad = stack.pop()
        
        return points
    
    def _generate_strange_attractor(self, dna: Dict[str, float], 
                                  emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate strange attractor patterns (Lorenz, Rossler, etc.)."""
        points = []
        
        # Attractor parameters from DNA and emotions
        attractor_type = ['lorenz', 'rossler', 'chua', 'thomas'][
            int(dna.get('symmetry', 4) % 4)
        ]
        
        sensitivity = dna.get('chaos_factor', 0.5)
        evolution_speed = emotional_params.get('animation_speed', 0.5)
        
        # Initial conditions
        x, y, z = 0.1, 0.1, 0.1
        dt = 0.01 * evolution_speed
        
        num_points = int(1000 + 4000 * emotional_params.get('pattern_density', 0.5))
        
        for i in range(num_points):
            # Initialize defaults
            dx = dy = dz = 0.0
            
            if attractor_type == 'lorenz':
                # Lorenz attractor
                sigma = 10.0 + sensitivity * 5
                rho = 28.0 + sensitivity * 10
                beta = 8.0/3.0 + sensitivity * 2
                
                dx = sigma * (y - x)
                dy = x * (rho - z) - y
                dz = x * y - beta * z
                
            elif attractor_type == 'rossler':
                # Rössler attractor
                a = 0.2 + sensitivity * 0.3
                b = 0.2 + sensitivity * 0.3
                c = 5.7 + sensitivity * 5
                
                dx = -y - z
                dy = x + a * y
                dz = b + z * (x - c)
            
            else:
                # Default simple attractor
                dx = -y + x * 0.1
                dy = x + y * 0.1
                dz = -z * 0.1
                
            # Update position
            x += dx * dt
            y += dy * dt
            z += dz * dt
            
            # Project to 2D
            px = int(self.canvas_width/2 + x * 8)
            py = int(self.canvas_height/2 + y * 8)
            
            if 0 <= px < self.canvas_width and 0 <= py < self.canvas_height:
                # Color based on velocity
                velocity = math.sqrt(dx*dx + dy*dy + dz*dz)
                color = (i + int(velocity * 50)) % 360
                
                points.append((px, py, velocity, color, z, i))
        
        return points
    
    def _generate_cellular_automata_3d(self, dna: Dict[str, float], 
                                     emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate 3D cellular automata projection."""
        points = []
        
        # Parameters
        rule = int(30 + 225 * emotional_params.get('creativity', 0.5)) % 256
        generations = int(30 + 70 * emotional_params.get('pattern_complexity', 0.5))
        
        width = min(100, self.canvas_width // 8)
        height = min(100, self.canvas_height // 8)
        
        # Initialize 3D grid
        grid = []
        for z in range(3):  # 3 layers
            layer = []
            for y in range(height):
                row = [0] * width
                if y == height // 2:  # Middle row
                    row[width // 2] = 1  # Seed in center
                layer.append(row)
            grid.append(layer)
        
        # Evolve the automata
        for gen in range(generations):
            new_grid = []
            for z in range(3):
                new_layer = []
                for y in range(height):
                    new_row = [0] * width
                    for x in range(width):
                        # Count neighbors in 3D
                        neighbors = 0
                        for dz in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dx in [-1, 0, 1]:
                                    if dx == 0 and dy == 0 and dz == 0:
                                        continue
                                    nz, ny, nx = (z + dz) % 3, (y + dy) % height, (x + dx) % width
                                    neighbors += grid[nz][ny][nx]
                        
                        # Apply rule (simplified)
                        if grid[z][y][x] == 1:
                            new_row[x] = 1 if 2 <= neighbors <= 3 else 0
                        else:
                            new_row[x] = 1 if neighbors == 3 else 0
                    
                    new_layer.append(new_row)
                new_grid.append(new_layer)
            grid = new_grid
        
        # Convert to 2D points
        for z in range(3):
            for y in range(height):
                for x in range(width):
                    if grid[z][y][x]:
                        px = x * (self.canvas_width // width)
                        py = y * (self.canvas_height // height)
                        intensity = 0.3 + 0.7 * z / 3
                        color = (z * 120 + gen * 10) % 360
                        points.append((px, py, intensity, color, z, gen))
        
        return points
    
    def _generate_topology_morphing(self, dna: Dict[str, float], 
                                  emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate topology morphing patterns."""
        points = []
        
        # Morphing parameters
        morph_type = ['torus', 'klein_bottle', 'mobius_strip', 'sphere'][
            int(dna.get('dimensional_fold', 2)) % 4
        ]
        
        morph_factor = emotional_params.get('temporal_variance', 0.5)
        resolution = int(20 + 80 * emotional_params.get('pattern_density', 0.5))
        
        for u_step in range(resolution):
            for v_step in range(resolution):
                u = (u_step / resolution) * 2 * math.pi
                v = (v_step / resolution) * math.pi
                
                if morph_type == 'torus':
                    # Torus parametric equations
                    R = 80 + 40 * dna.get('growth_rate', 1.0)
                    r = 30 + 20 * dna.get('spiral_tightness', 1.0)
                    
                    x = (R + r * math.cos(v)) * math.cos(u)
                    y = (R + r * math.cos(v)) * math.sin(u)
                    z = r * math.sin(v)
                    
                elif morph_type == 'klein_bottle':
                    # Klein bottle (figure-8 immersion)
                    x = (2 + math.cos(v/2) * math.sin(u) - math.sin(v/2) * math.sin(2*u)) * math.cos(v/2)
                    y = (2 + math.cos(v/2) * math.sin(u) - math.sin(v/2) * math.sin(2*u)) * math.sin(v/2)
                    z = math.sin(v/2) * math.sin(u) + math.cos(v/2) * math.sin(2*u)
                    
                elif morph_type == 'mobius_strip':
                    # Möbius strip
                    x = (1 + v/2 * math.cos(u/2)) * math.cos(u)
                    y = (1 + v/2 * math.cos(u/2)) * math.sin(u)
                    z = v/2 * math.sin(u/2)
                    
                else:  # sphere
                    # Sphere with deformation
                    deform = 1 + 0.3 * math.sin(3*u) * math.sin(3*v) * morph_factor
                    x = deform * math.sin(v) * math.cos(u)
                    y = deform * math.sin(v) * math.sin(u)
                    z = deform * math.cos(v)
                
                # Project to 2D
                px = int(self.canvas_width/2 + x * 3)
                py = int(self.canvas_height/2 + y * 3)
                
                if 0 <= px < self.canvas_width and 0 <= py < self.canvas_height:
                    intensity = 0.5 + 0.5 * math.sin(u + v)
                    color = (int(u * 180 / math.pi) + int(v * 180 / math.pi)) % 360
                    points.append((px, py, intensity, color, z, u_step + v_step))
        
        return points
    
    def _generate_parametric_surface(self, dna: Dict[str, float], 
                                   emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate parametric surface patterns."""
        points = []
        
        # Surface parameters
        surface_type = ['hyperbolic', 'saddle', 'wave', 'spiral'][
            int(dna.get('fractal_dimension', 2)) % 4
        ]
        
        amplitude = dna.get('amplitude', 50)
        frequency = dna.get('frequency', 2.0)
        resolution = int(30 + 70 * emotional_params.get('pattern_density', 0.5))
        
        for u_step in range(resolution):
            for v_step in range(resolution):
                u = (u_step / resolution - 0.5) * 4
                v = (v_step / resolution - 0.5) * 4
                
                if surface_type == 'hyperbolic':
                    z = amplitude * (u*u - v*v) / 16
                elif surface_type == 'saddle':
                    z = amplitude * math.sin(frequency * u) * math.cos(frequency * v)
                elif surface_type == 'wave':
                    z = amplitude * math.sin(frequency * math.sqrt(u*u + v*v))
                else:  # spiral
                    r = math.sqrt(u*u + v*v)
                    theta = math.atan2(v, u)
                    z = amplitude * math.sin(frequency * r + theta)
                
                # Project to 2D with perspective
                scale = 200 / (200 + z)
                px = int(self.canvas_width/2 + u * scale * 20)
                py = int(self.canvas_height/2 + v * scale * 20)
                
                if 0 <= px < self.canvas_width and 0 <= py < self.canvas_height:
                    intensity = 0.3 + 0.7 * (z + amplitude) / (2 * amplitude)
                    color = (int(z * 2) + u_step + v_step) % 360
                    points.append((px, py, intensity, color, z, u_step))
        
        return points
    
    def _generate_field_equation_visualization(self, dna: Dict[str, float], 
                                             emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate field equation visualization."""
        points = []
        
        # Field parameters
        field_strength = dna.get('attraction_strength', 1.0)
        complexity = emotional_params.get('pattern_complexity', 0.5)
        
        step = max(5, int(20 - 15 * emotional_params.get('pattern_density', 0.5)))
        
        for x in range(0, self.canvas_width, step):
            for y in range(0, self.canvas_height, step):
                # Normalize coordinates
                nx = (x - self.canvas_width/2) / (self.canvas_width/2)
                ny = (y - self.canvas_height/2) / (self.canvas_height/2)
                
                # Calculate field value
                r = math.sqrt(nx*nx + ny*ny)
                if r > 0:
                    # Electric field-like pattern
                    field_x = field_strength * nx / (r*r + 0.1)
                    field_y = field_strength * ny / (r*r + 0.1)
                    
                    # Add wave interference
                    wave1 = math.sin(r * 10 * complexity)
                    wave2 = math.cos(nx * 8 + ny * 6)
                    
                    field_magnitude = math.sqrt(field_x*field_x + field_y*field_y)
                    field_magnitude *= (1 + 0.5 * wave1 * wave2)
                    
                    if field_magnitude > 0.1:
                        intensity = min(1.0, field_magnitude / 2)
                        color = (int(math.atan2(field_y, field_x) * 180 / math.pi) + 180) % 360
                        points.append((x, y, intensity, color, field_magnitude, int(r * 10)))
        
        return points
    
    def _generate_quantum_harmonic_oscillator(self, dna: Dict[str, float], 
                                            emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate quantum harmonic oscillator wave functions."""
        points = []
        
        try:
            # Safe factorial function
            def safe_factorial(n):
                if n <= 0:
                    return 1
                if n > 10:  # Limit to prevent overflow
                    return 3628800  # 10!
                result = 1
                for i in range(1, n + 1):
                    result *= i
                return result
            
            # Quantum parameters
            n_levels = int(1 + 4 * emotional_params.get('pattern_complexity', 0.5))  # Reduced max levels
            omega = dna.get('resonance_frequency', 0.5)
            
            width = self.canvas_width
            height = self.canvas_height
            
            for x in range(0, width, 5):  # Increased step for performance
                # Normalize position
                xi = (x - width/2) / (width/4)
                
                psi_total = 0
                for n in range(min(n_levels, 5)):  # Limit to prevent complexity
                    # Hermite polynomial approximation
                    if n == 0:
                        Hn = 1
                    elif n == 1:
                        Hn = 2 * xi
                    elif n == 2:
                        Hn = 4 * xi*xi - 2
                    elif n == 3:
                        Hn = 8 * xi*xi*xi - 12 * xi
                    else:
                        Hn = xi**n  # Simplified for higher orders
                    
                    # Wave function
                    try:
                        normalization = (omega / math.pi)**0.25 / math.sqrt(2**n * safe_factorial(n))
                        psi_n = normalization * Hn * math.exp(-omega * xi*xi / 2)
                        
                        # Superposition with time evolution
                        phase = omega * (n + 0.5) * time.time() * 0.05  # Slower evolution
                        psi_total += psi_n * math.cos(phase)
                    except (OverflowError, ZeroDivisionError):
                        continue
                
                # Probability density
                probability = abs(psi_total * psi_total)
                probability = min(probability, 1.0)  # Clamp to prevent overflow
                
                # Create visualization
                if probability > 0.01:  # Only draw significant probabilities
                    for offset in range(-int(probability * 30), int(probability * 30) + 1, 8):
                        y = height//2 + offset
                        if 0 <= y < height:
                            intensity = max(0, 1 - abs(offset) / (probability * 30 + 1))
                            color = (int(probability * 360) + x) % 360
                            points.append((x, y, intensity, color, probability, n_levels))
        
        except Exception as e:
            print(f"Quantum oscillator error: {e}")
            # Fallback to simple wave
            for x in range(0, self.canvas_width, 10):
                y = self.canvas_height//2 + int(30 * math.sin(x * 0.1))
                if 0 <= y < self.canvas_height:
                    points.append((x, y, 0.5, (x * 2) % 360, 0.5, 1))
        
        return points
    
    def _generate_metamorphic_pattern(self, dna: Dict[str, float], 
                                    emotional_params: Dict[str, float]) -> List[Tuple]:
        """Generate metamorphic/evolving patterns."""
        points = []
        
        # Metamorphic parameters
        evolution_stage = (time.time() * dna.get('evolution_speed', 1.0)) % (2 * math.pi)
        complexity = emotional_params.get('pattern_complexity', 0.5)
        
        # Generate base spiral
        num_turns = 3 + 7 * complexity
        points_per_turn = int(50 + 150 * emotional_params.get('pattern_density', 0.5))
        
        for i in range(int(num_turns * points_per_turn)):
            t = i / points_per_turn
            
            # Evolving spiral parameters
            radius = 20 + 100 * t / num_turns
            angle = t * 2 * math.pi
            
            # Metamorphic transformations
            morph_factor = math.sin(evolution_stage)
            
            # Base position
            x = self.canvas_width/2 + radius * math.cos(angle)
            y = self.canvas_height/2 + radius * math.sin(angle)
            
            # Apply metamorphic distortions
            distortion_x = 30 * morph_factor * math.sin(t * 3 + evolution_stage)
            distortion_y = 30 * morph_factor * math.cos(t * 2 + evolution_stage)
            
            px = int(x + distortion_x)
            py = int(y + distortion_y)
            
            if 0 <= px < self.canvas_width and 0 <= py < self.canvas_height:
                intensity = 0.5 + 0.5 * math.sin(t * 5 + evolution_stage)
                color = (int(t * 60) + int(evolution_stage * 60)) % 360
                metamorphic_value = morph_factor
                points.append((px, py, intensity, color, metamorphic_value, i))
        
        return points
    
    def _apply_spatial_attention(self, pattern_data: List[Tuple], 
                               attention_focus: Tuple[float, float],
                               emotional_params: Dict[str, float]) -> List[Tuple]:
        """Apply spatial attention to enhance certain areas."""
        focus_x = attention_focus[0] * self.canvas_width
        focus_y = attention_focus[1] * self.canvas_height
        attention_radius = 100 + 200 * emotional_params.get('focus', 0.5)
        
        enhanced_data = []
        
        for point in pattern_data:
            px, py = point[0], point[1]
            
            # Calculate distance from attention focus
            distance = math.sqrt((px - focus_x)**2 + (py - focus_y)**2)
            
            # Attention strength (Gaussian falloff)
            attention_strength = math.exp(-(distance**2) / (2 * attention_radius**2))
            
            # Enhance based on attention
            if len(point) >= 3:
                enhanced_intensity = point[2] * (1 + attention_strength * 2)
                enhanced_point = list(point)
                enhanced_point[2] = enhanced_intensity
                enhanced_point.append(attention_strength)
                enhanced_data.append(tuple(enhanced_point))
            else:
                enhanced_data.append(point)
        
        return enhanced_data
    
    def _apply_temporal_evolution(self, pattern_data: List[Tuple], 
                                dna: Dict[str, float],
                                emotional_params: Dict[str, float]) -> List[Tuple]:
        """Apply temporal evolution to patterns."""
        current_time = time.time()
        evolution_speed = dna.get('evolution_speed', 1.0)
        temporal_variance = dna.get('temporal_variance', 1.0)
        
        evolved_data = []
        
        for point in pattern_data:
            # Time-based modulation
            time_factor = math.sin(current_time * evolution_speed + point[0] * 0.01 + point[1] * 0.01)
            
            # Apply temporal effects
            if len(point) >= 3:
                temporal_modulation = 1 + 0.3 * time_factor * temporal_variance
                evolved_point = list(point)
                evolved_point[2] *= temporal_modulation
                
                # Add temporal phase
                evolved_point.append(time_factor)
                evolved_data.append(tuple(evolved_point))
            else:
                evolved_data.append(point)
        
        return evolved_data
    
    def evolve_patterns(self, fitness_criteria: Dict[str, float]):
        """Evolve existing patterns using genetic algorithms with enhanced safety."""
        try:
            if len(self.active_patterns) < 2:
                print("Not enough patterns for evolution")
                return
            
            # Select patterns for evolution based on fitness
            patterns_list = list(self.active_patterns.values())
            patterns_list.sort(key=lambda p: p.get('fitness_score', 0), reverse=True)
            
            # Take top performers for breeding
            parents = patterns_list[:max(2, len(patterns_list)//2)]
            
            # Create new generation
            new_patterns = {}
            
            for i in range(0, min(len(parents)-1, 2), 2):  # Limit to prevent too many patterns
                try:
                    parent1 = parents[i]
                    parent2 = parents[i+1]
                    
                    # Crossover DNA
                    child1_dna, child2_dna = self.pattern_dna.crossover_dna(
                        parent1['dna'], parent2['dna']
                    )
                    
                    # Mutate DNA
                    child1_dna = self.pattern_dna.mutate_dna(child1_dna, 
                        mutation_rate=0.1, mutation_strength=0.2)
                    child2_dna = self.pattern_dna.mutate_dna(child2_dna, 
                        mutation_rate=0.1, mutation_strength=0.2)
                    
                    # Create new patterns safely
                    for child_dna in [child1_dna, child2_dna]:
                        try:
                            child_pattern = self.create_multidimensional_pattern(
                                child_dna, 
                                parent1['emotional_state'],  # Inherit emotional state
                                parent1['attention_focus']   # Inherit attention focus
                            )
                            child_pattern['evolution_generation'] = max(
                                parent1.get('evolution_generation', 0), 
                                parent2.get('evolution_generation', 0)
                            ) + 1
                            
                            new_patterns[child_pattern['id']] = child_pattern
                        except Exception as e:
                            print(f"Child pattern creation error: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Pattern breeding error: {e}")
                    continue
            
            # Replace old patterns with evolved ones (keep some survivors)
            survivors = {p['id']: p for p in patterns_list[:max(1, len(patterns_list)//3)]}
            survivors.update(new_patterns)
            
            # Limit total pattern count to prevent memory issues
            if len(survivors) > 10:
                survivor_list = list(survivors.values())
                survivor_list.sort(key=lambda p: p.get('fitness_score', 0), reverse=True)
                survivors = {p['id']: p for p in survivor_list[:10]}
            
            self.active_patterns = survivors
            
            # Record evolution history
            try:
                self.pattern_evolution_history.append({
                    'timestamp': time.time(),
                    'generation_count': len(survivors),
                    'avg_fitness': sum(p.get('fitness_score', 0) for p in survivors.values()) / max(1, len(survivors)),
                    'fitness_criteria': fitness_criteria.copy()
                })
            except Exception as e:
                print(f"Evolution history error: {e}")
                
        except Exception as e:
            print(f"Pattern evolution error: {e}")



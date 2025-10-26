#!/usr/bin/env python3
"""
Create a simple test PLY file for testing the surface offset thinning
"""

import numpy as np
import trimesh

def create_test_cube():
    """Create a simple cube PLY file for testing"""
    # Create a simple cube
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    
    # Add some color variation
    cube.visual.face_colors = [255, 100, 100, 255]  # Red
    
    # Save to input directory
    cube.export("input/test_cube.ply")
    print("Created test_cube.ply")

def create_test_cylinder():
    """Create a simple cylinder PLY file for testing"""
    # Create a cylinder
    cylinder = trimesh.creation.cylinder(radius=0.5, height=2.0)
    
    # Add some color variation
    cylinder.visual.face_colors = [100, 255, 100, 255]  # Green
    
    # Save to input directory
    cylinder.export("input/test_cylinder.ply")
    print("Created test_cylinder.ply")

def create_test_sphere():
    """Create a simple sphere PLY file for testing"""
    # Create a sphere
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.8)
    
    # Add some color variation
    sphere.visual.face_colors = [100, 100, 255, 255]  # Blue
    
    # Save to input directory
    sphere.export("input/test_sphere.ply")
    print("Created test_sphere.ply")

if __name__ == "__main__":
    import os
    os.makedirs("input", exist_ok=True)
    
    create_test_cube()
    create_test_cylinder()
    create_test_sphere()
    
    print("Test PLY files created in input/ directory")

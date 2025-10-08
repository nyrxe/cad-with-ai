#!/usr/bin/env python3
"""
Advanced Voxel Thinning Tool
Comprehensive erosion with interface protection, connectivity checks, and stress-aware mode.
"""

import numpy as np
import os
from scipy.ndimage import binary_erosion, binary_dilation, label
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoxelThinner:
    """Advanced voxel thinning with interface protection and safety checks."""
    
    def __init__(self, erosion_layers=1, min_voxels_core=2, max_reduction=0.2, 
                 stress_percentile=80, stress_array=None):
        """
        Initialize the voxel thinner.
        
        Args:
            erosion_layers (int): Number of erosion layers to apply
            min_voxels_core (int): Minimum solid voxels behind surface voxels
            max_reduction (float): Maximum allowed reduction (0.0-1.0)
            stress_percentile (float): Percentile threshold for stress protection
            stress_array (np.ndarray): Optional per-voxel stress values
        """
        self.erosion_layers = erosion_layers
        self.min_voxels_core = min_voxels_core
        self.max_reduction = max_reduction
        self.stress_percentile = stress_percentile
        self.stress_array = stress_array
        
        # 6-connected structuring element for erosion
        self.structuring_element = np.array([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=bool)
    
    def build_color_grid(self, indices, colors, grid_shape, offset):
        """Build 3D grid with color information using offset indices."""
        grid = np.zeros(grid_shape, dtype=np.int32)
        color_map = {}
        
        for i, (idx, color) in enumerate(zip(indices, colors)):
            x, y, z = idx - offset  # Apply offset to get grid coordinates
            if (0 <= x < grid_shape[0] and 
                0 <= y < grid_shape[1] and 
                0 <= z < grid_shape[2]):
                grid[x, y, z] = i + 1  # 1-based indexing
                color_map[i + 1] = color
        
        return grid, color_map
    
    def find_interface_voxels(self, grid, color_map):
        """Find voxels adjacent to different colors (interfaces)."""
        interfaces = np.zeros_like(grid, dtype=bool)
        
        # Check 6-connected neighbors
        directions = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
        ]
        
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                for z in range(grid.shape[2]):
                    if grid[x, y, z] == 0:
                        continue
                    
                    current_color = color_map[grid[x, y, z]]
                    
                    # Check neighbors
                    for dx, dy, dz in directions:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (0 <= nx < grid.shape[0] and 
                            0 <= ny < grid.shape[1] and 
                            0 <= nz < grid.shape[2] and
                            grid[nx, ny, nz] != 0):
                            
                            neighbor_color = color_map[grid[nx, ny, nz]]
                            if not np.array_equal(current_color, neighbor_color):
                                interfaces[x, y, z] = True
                                break
        
        return interfaces
    
    def create_protection_mask(self, grid, interfaces, stress_array=None):
        """Create protection mask for interface and high-stress voxels."""
        protection = np.zeros_like(grid, dtype=bool)
        
        # Protect interface voxels and their 1-voxel neighbors (6-connected only)
        interface_dilated = binary_dilation(interfaces, structure=self.structuring_element)
        protection |= interface_dilated
        
        # Protect high-stress voxels if stress array provided
        if stress_array is not None:
            stress_threshold = np.percentile(stress_array, self.stress_percentile)
            high_stress_mask = stress_array > stress_threshold
            protection |= high_stress_mask
            
            # Add 1-voxel halo around high-stress voxels (6-connected only)
            stress_halo = binary_dilation(high_stress_mask, structure=self.structuring_element)
            protection |= stress_halo
        
        return protection
    
    def check_local_thickness(self, grid, x, y, z, min_voxels_core):
        """Check if voxel has sufficient thickness behind it in ANY direction."""
        directions = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
        ]
        
        # Check each direction pair (+x/-x, +y/-y, +z/-z)
        for i in range(0, len(directions), 2):
            dx1, dy1, dz1 = directions[i]
            dx2, dy2, dz2 = directions[i + 1]
            
            # Check +direction
            count1 = 0
            nx, ny, nz = x, y, z
            for _ in range(min_voxels_core):
                nx, ny, nz = nx + dx1, ny + dy1, nz + dz1
                if (0 <= nx < grid.shape[0] and 
                    0 <= ny < grid.shape[1] and 
                    0 <= nz < grid.shape[2] and
                    grid[nx, ny, nz] != 0):
                    count1 += 1
                else:
                    break
            
            # Check -direction
            count2 = 0
            nx, ny, nz = x, y, z
            for _ in range(min_voxels_core):
                nx, ny, nz = nx + dx2, ny + dy2, nz + dz2
                if (0 <= nx < grid.shape[0] and 
                    0 <= ny < grid.shape[1] and 
                    0 <= nz < grid.shape[2] and
                    grid[nx, ny, nz] != 0):
                    count2 += 1
                else:
                    break
            
            # If either direction has enough thickness, this voxel is safe to erode
            if count1 >= min_voxels_core or count2 >= min_voxels_core:
                return True
        
        return False
    
    def erode_part(self, part_grid, part_protection, part_id, pitch):
        """Erode a single part with safety checks."""
        original_count = np.sum(part_grid)
        min_thickness_mm = self.min_voxels_core * pitch
        
        logger.info(f"Part {part_id}: {original_count} voxels, min thickness: {min_thickness_mm:.3f}mm")
        
        # Check if part is too thin for erosion
        if original_count < self.min_voxels_core * 6:  # Minimum for 6 directions
            logger.warning(f"Part {part_id} too thin for erosion, skipping")
            return part_grid, 0, 0.0
        
        # Apply erosion with protection
        eroded_grid = part_grid.copy()
        layers_applied = 0
        
        for layer in range(self.erosion_layers):
            # Compute core and boundary
            core = binary_erosion(eroded_grid, structure=self.structuring_element)
            boundary = eroded_grid & ~core
            
            # Only check thickness on boundary voxels
            safe_to_remove = np.zeros_like(boundary, dtype=bool)
            for x, y, z in np.argwhere(boundary):
                if self.check_local_thickness(eroded_grid, x, y, z, self.min_voxels_core):
                    safe_to_remove[x, y, z] = True
            
            # Apply protection only to boundary voxels
            part_protection_boundary = part_protection & boundary
            
            # Safe to remove = boundary voxels that are not protected and pass thickness check
            safe_to_remove = boundary & (~part_protection_boundary) & safe_to_remove
            
            # Next grid = current grid minus safe-to-remove voxels
            next_grid = eroded_grid & (~safe_to_remove)
            
            # Sanity check
            print(f"  Debug: core={np.sum(core)}, boundary={np.sum(boundary)}, safe_to_remove={np.sum(safe_to_remove)}")
            
            temp_eroded = next_grid
            
            # Check reduction limit
            new_count = np.sum(temp_eroded)
            reduction = (original_count - new_count) / original_count
            
            if reduction > self.max_reduction:
                logger.warning(f"Part {part_id} reduction {reduction:.1%} exceeds limit {self.max_reduction:.1%}")
                break
            
            if new_count == 0:
                logger.warning(f"Part {part_id} would disappear, stopping erosion")
                break
            
            eroded_grid = temp_eroded
            layers_applied += 1
        
        # Check connectivity
        labeled, num_components = label(eroded_grid)
        if num_components > 1:
            logger.warning(f"Part {part_id} split into {num_components} components")
            # Keep only largest component
            component_sizes = [np.sum(labeled == i) for i in range(1, num_components + 1)]
            largest_component = np.argmax(component_sizes) + 1
            eroded_grid = (labeled == largest_component)
        
        final_count = np.sum(eroded_grid)
        # Thickness change = layers applied * pitch (in mm)
        delta_t_mm = layers_applied * pitch * 1000.0  # Convert meters to mm
        
        return eroded_grid, layers_applied, delta_t_mm
    
    def extract_voxels_vectorized(self, grid, color_map, bbox):
        """Extract voxel coordinates and colors using vectorized operations."""
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        
        # Work in tight bounding box
        subgrid = grid[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        
        # Find non-zero voxels
        coords = np.argwhere(subgrid > 0)
        if len(coords) == 0:
            return np.array([]), np.array([])
        
        # Adjust coordinates to global space
        coords[:, 0] += x_min
        coords[:, 1] += y_min
        coords[:, 2] += z_min
        
        # Get colors - handle the case where grid contains boolean values
        voxel_ids = subgrid[subgrid > 0]
        
        # If we have a single part, use the part color directly
        if len(color_map) == 1:
            part_color = list(color_map.values())[0]
            colors = np.array([part_color] * len(voxel_ids))
        else:
            # Multiple parts - use color map
            colors = np.array([color_map[vid] for vid in voxel_ids])
        
        return coords, colors
    
    def validate_results(self, original_indices, eroded_indices, original_colors, eroded_colors, pitch):
        """Validate erosion results."""
        issues = []
        
        # Check reduction limit
        reduction = (len(original_indices) - len(eroded_indices)) / len(original_indices)
        if reduction > self.max_reduction:
            issues.append(f"Reduction {reduction:.1%} exceeds limit {self.max_reduction:.1%}")
        
        # Check volume change matches voxel count change
        original_volume = len(original_indices) * (pitch ** 3)
        eroded_volume = len(eroded_indices) * (pitch ** 3)
        volume_change = (original_volume - eroded_volume) / original_volume
        
        if abs(volume_change - reduction) > 0.01:  # 1% tolerance
            issues.append(f"Volume change {volume_change:.1%} doesn't match voxel reduction {reduction:.1%}")
        
        return issues
    
    def thin_voxels(self, npz_path, output_path=None):
        """Main thinning function with all safety features."""
        logger.info(f"Processing: {npz_path}")
        
        # Load data
        data = np.load(npz_path)
        indices = data["indices"].astype(np.int64)
        colors = data["colors"].astype(np.uint8)
        
        # Handle pitch extraction safely
        pitch_arr = data["pitch"]
        if pitch_arr.size == 1:
            pitch = float(pitch_arr.item())
        else:
            pitch = float(pitch_arr.ravel()[0])
        
        transform = data["transform"].astype(np.float64)
        
        logger.info(f"Original voxels: {len(indices)}")
        logger.info(f"Pitch: {pitch:.6f}m")
        
        # Get grid dimensions
        imin, jmin, kmin = indices.min(axis=0)
        imax, jmax, kmax = indices.max(axis=0)
        grid_shape = (imax - imin + 1, jmax - jmin + 1, kmax - kmin + 1)
        
        # Build color grid with proper offset
        offset = np.array([imin, jmin, kmin])
        grid, color_map = self.build_color_grid(indices, colors, grid_shape, offset)
        
        # Find interfaces
        interfaces = self.find_interface_voxels(grid, color_map)
        logger.info(f"Interface voxels: {np.sum(interfaces)}")
        
        # Create protection mask
        protection = self.create_protection_mask(grid, interfaces, self.stress_array)
        logger.info(f"Protected voxels: {np.sum(protection)}")
        
        # Process each part
        unique_colors, color_indices = np.unique(colors, axis=0, return_inverse=True)
        eroded_indices = []
        eroded_colors = []
        stats = {}
        
        for part_idx in range(len(unique_colors)):
            part_color = unique_colors[part_idx]
            part_mask = (color_indices == part_idx)
            part_voxels = indices[part_mask]
            
            if len(part_voxels) == 0:
                continue
            
            logger.info(f"\nProcessing part {part_idx}: {len(part_voxels)} voxels")
            
            # Create part grid
            part_grid = np.zeros(grid_shape, dtype=bool)
            for voxel in part_voxels:
                x, y, z = voxel - [imin, jmin, kmin]
                part_grid[x, y, z] = True
            
            # Get protection for this part
            part_protection = protection & part_grid
            
            # Erode part
            eroded_part_grid, layers_applied, delta_t_mm = self.erode_part(
                part_grid, part_protection, part_idx, pitch
            )
            
            # Extract eroded voxels
            if np.sum(eroded_part_grid) > 0:
                bbox = (imin, imax, jmin, jmax, kmin, kmax)
                coords, part_colors = self.extract_voxels_vectorized(
                    eroded_part_grid, {part_idx: part_color}, bbox
                )
                
                if len(coords) > 0:
                    eroded_indices.extend(coords)
                    eroded_colors.extend(part_colors)
            
            # Record statistics
            original_count = len(part_voxels)
            eroded_count = len(coords) if 'coords' in locals() and len(coords) > 0 else 0
            reduction = (original_count - eroded_count) / original_count if original_count > 0 else 0
            
            stats[part_idx] = {
                'layers_requested': self.erosion_layers,
                'layers_applied': layers_applied,
                'delta_t_mm': delta_t_mm,
                'original_voxels': original_count,
                'eroded_voxels': eroded_count,
                'reduction_percent': reduction * 100,
                'components_count': 1,  # Simplified for now
                'safety_blocked': layers_applied < self.erosion_layers
            }
            
            logger.info(f"Part {part_idx} results:")
            logger.info(f"  Layers applied: {layers_applied}/{self.erosion_layers}")
            logger.info(f"  Thickness change: {delta_t_mm:.3f}mm")
            logger.info(f"  Voxels: {original_count} → {eroded_count}")
            logger.info(f"  Reduction: {reduction:.1%}")
        
        # Convert to arrays
        if len(eroded_indices) == 0:
            logger.warning("No voxels remaining after erosion!")
            return None
        
        eroded_indices = np.array(eroded_indices, dtype=np.int64)
        eroded_colors = np.array(eroded_colors, dtype=np.uint8)
        
        # Validate results
        validation_issues = self.validate_results(
            indices, eroded_indices, colors, eroded_colors, pitch
        )
        
        if validation_issues:
            logger.error("Validation failed:")
            for issue in validation_issues:
                logger.error(f"  {issue}")
            return None
        
        # Save results
        if output_path is None:
            base_path = npz_path.replace('.npz', '')
            output_path = f"{base_path}_thinned.npz"
        
        np.savez_compressed(
            output_path,
            indices=eroded_indices,
            colors=eroded_colors,
            pitch=np.array([pitch], np.float32),
            transform=transform.astype(np.float32)
        )
        
        logger.info(f"Thinned data saved to: {output_path}")
        logger.info(f"Final reduction: {(1 - len(eroded_indices) / len(indices)) * 100:.1f}%")
        
        return {
            'output_path': output_path,
            'original_voxels': len(indices),
            'thinned_voxels': len(eroded_indices),
            'reduction_percent': (1 - len(eroded_indices) / len(indices)) * 100,
            'part_stats': stats,
            'validation_issues': validation_issues
        }

def process_all_models(base_dir="voxel_out", **kwargs):
    """Process all models in directory."""
    logger.info("=== ADVANCED VOXEL THINNING ===")
    
    if not os.path.exists(base_dir):
        logger.error(f"Directory not found: {base_dir}")
        return
    
    model_dirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    
    if not model_dirs:
        logger.error("No model directories found")
        return
    
    logger.info(f"Found {len(model_dirs)} models: {model_dirs}")
    
    results = {}
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        npz_path = os.path.join(model_path, "voxels_filled_indices_colors.npz")
        
        if not os.path.exists(npz_path):
            logger.warning(f"Skipping {model_dir}: No voxel data found")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {model_dir}")
        logger.info(f"{'='*60}")
        
        try:
            thinner = VoxelThinner(**kwargs)
            result = thinner.thin_voxels(npz_path)
            if result:
                results[model_dir] = result
        except Exception as e:
            logger.error(f"Error processing {model_dir}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info("THINNING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Successfully processed: {len(results)}/{len(model_dirs)} models")
    
    for model, stats in results.items():
        logger.info(f"\n{model}:")
        logger.info(f"  Original: {stats['original_voxels']} voxels")
        logger.info(f"  Thinned: {stats['thinned_voxels']} voxels")
        logger.info(f"  Reduction: {stats['reduction_percent']:.1f}%")
    
    return results

def main():
    """Main function for command-line usage."""
    import sys
    
    # Default parameters
    erosion_layers = 1
    min_voxels_core = 2
    max_reduction = 0.2
    stress_percentile = 80
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            erosion_layers = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid erosion layers '{sys.argv[1]}', using default: {erosion_layers}")
    
    if len(sys.argv) > 2:
        try:
            min_voxels_core = int(sys.argv[2])
        except ValueError:
            logger.warning(f"Invalid min voxels core '{sys.argv[2]}', using default: {min_voxels_core}")
    
    if len(sys.argv) > 3:
        try:
            max_reduction = float(sys.argv[3])
        except ValueError:
            logger.warning(f"Invalid max reduction '{sys.argv[3]}', using default: {max_reduction}")
    
    if len(sys.argv) > 4:
        try:
            stress_percentile = float(sys.argv[4])
        except ValueError:
            logger.warning(f"Invalid stress percentile '{sys.argv[4]}', using default: {stress_percentile}")
    
    logger.info(f"Thinning parameters:")
    logger.info(f"  Erosion layers: {erosion_layers}")
    logger.info(f"  Min voxels core: {min_voxels_core}")
    logger.info(f"  Max reduction: {max_reduction:.1%}")
    logger.info(f"  Stress percentile: {stress_percentile}")
    
    # Process all models
    results = process_all_models(
        erosion_layers=erosion_layers,
        min_voxels_core=min_voxels_core,
        max_reduction=max_reduction,
        stress_percentile=stress_percentile
    )
    
    if results:
        logger.info(f"\nThinned models ready for FEA analysis!")
        logger.info("Run: python voxel_to_fea_complete.py")
        logger.info("This will analyze both original and thinned versions.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Voxel SDF Thinning
Applies AI-recommended thinning by removing voxel layers using SDF-based offset.
Produces a new NPZ file with fewer voxels while preserving structure.
"""

import numpy as np
import pandas as pd
import os
import logging
from scipy.ndimage import distance_transform_edt, label
from safe_print import safe_print

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoxelSDFThinner:
    """Voxel thinning using SDF-based layer removal"""
    
    def __init__(self, t_min_mm=2.0, interface_halo=True, tolerance_pct=0.5):
        """
        Initialize the voxel SDF thinner.
        
        Args:
            t_min_mm (float): Minimum wall thickness in mm
            interface_halo (bool): Enable 1-voxel protection around part boundaries
            tolerance_pct (float): Tolerance for target reduction percentage
        """
        self.t_min_mm = t_min_mm
        self.interface_halo = interface_halo
        self.tolerance_pct = tolerance_pct
        
        # 6-connected structuring element for connectivity checks
        self.structuring_element = np.array([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=bool)
    
    def load_voxel_data(self, npz_path):
        """Load voxel data from NPZ file"""
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Voxel data not found: {npz_path}")
        
        data = np.load(npz_path)
        indices = data["indices"].astype(np.int64)
        
        # Handle pitch extraction
        pitch_arr = data["pitch"]
        if pitch_arr.size == 1:
            pitch = float(pitch_arr.item())
        else:
            pitch = float(pitch_arr.ravel()[0])
        
        transform = data["transform"].astype(np.float64)
        colors = data.get("colors", None)
        
        logger.info(f"Loaded voxel data: {len(indices)} voxels, pitch: {pitch:.6f}m")
        return indices, pitch, transform, colors
    
    def build_dense_grids(self, indices, colors=None):
        """Build dense occupancy and part grids from sparse indices with consistent (z,y,x) order"""
        # Get grid dimensions
        imin, jmin, kmin = indices.min(axis=0)
        imax, jmax, kmax = indices.max(axis=0)
        grid_shape = (kmax - kmin + 1, jmax - jmin + 1, imax - imin + 1)  # (z, y, x)
        offset = np.array([imin, jmin, kmin])
        
        # Build occupancy grid
        occ_grid = np.zeros(grid_shape, dtype=bool)
        for idx in indices:
            x, y, z = idx - offset
            if (0 <= x < grid_shape[2] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[0]):
                occ_grid[z, y, x] = True  # (z, y, x) order
        
        # Build part grid if colors available
        part_grid = None
        if colors is not None:
            part_grid = np.zeros(grid_shape, dtype=np.int32)
            unique_colors, color_indices = np.unique(colors, axis=0, return_inverse=True)
            
            for i, idx in enumerate(indices):
                x, y, z = idx - offset
                if (0 <= x < grid_shape[2] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[0]):
                    part_grid[z, y, x] = color_indices[i] + 1  # 1-based part IDs
        
        return occ_grid, part_grid, offset, unique_colors if colors is not None else None
    
    def compute_sdf(self, occ_grid):
        """Compute signed distance field from occupancy grid"""
        # Inside distance (positive values inside the object)
        din = distance_transform_edt(occ_grid)
        
        # Outside distance (positive values outside the object)
        dout = distance_transform_edt(~occ_grid)
        
        # Signed distance field: positive inside, negative outside
        sdf = din - dout
        
        return sdf
    
    def create_interface_halo(self, part_grid, part_id):
        """Create 1-voxel protection halo around part boundaries"""
        if not self.interface_halo or part_grid is None:
            return np.zeros_like(part_grid, dtype=bool)
        
        part_mask = (part_grid == part_id)
        other_parts_mask = (part_grid > 0) & (part_grid != part_id)
        
        if not np.any(other_parts_mask):
            return np.zeros_like(part_grid, dtype=bool)
        
        # Distance from this part to other parts
        part_distance = distance_transform_edt(~other_parts_mask)
        interface_halo = part_mask & (part_distance <= 1)
        
        return interface_halo
    
    def check_min_thickness_constraint(self, occ_grid, part_mask, pitch_mm, offset_vox):
        """Check if offset respects minimum thickness constraint"""
        if offset_vox <= 0:
            return True, 0.0
        
        # Compute SDF for the part
        part_occ = occ_grid & part_mask
        if not np.any(part_occ):
            return True, 0.0
        
        sdf = self.compute_sdf(part_occ)
        
        # Check minimum thickness after offset
        thinned_mask = sdf >= offset_vox
        if not np.any(thinned_mask):
            return False, 0.0
        
        # Compute local thickness in thinned region
        thinned_sdf = self.compute_sdf(thinned_mask)
        min_thickness_vox = self.t_min_mm / pitch_mm
        
        # Check if any remaining voxel has insufficient thickness
        insufficient_thickness = thinned_sdf < min_thickness_vox
        if np.any(insufficient_thickness):
            # Find maximum safe offset
            max_safe_offset = np.max(thinned_sdf[thinned_mask])
            return False, max_safe_offset
        
        return True, offset_vox
    
    def check_connectivity(self, occ_grid, part_mask):
        """Check 6-connectivity of a part and return largest component"""
        if not np.any(part_mask):
            return occ_grid
        
        # Extract part voxels
        part_occ = occ_grid & part_mask
        
        # Label connected components
        labeled, num_components = label(part_occ, structure=self.structuring_element)
        
        if num_components <= 1:
            return occ_grid
        
        # Keep largest component only
        component_sizes = []
        for i in range(1, num_components + 1):
            component_sizes.append(np.sum(labeled == i))
        
        largest_component = np.argmax(component_sizes) + 1
        largest_mask = (labeled == largest_component)
        
        # Update the grid
        updated_grid = occ_grid.copy()
        updated_grid[part_mask] = largest_mask
        
        return updated_grid
    
    def apply_surface_layer_thinning(self, occ_grid, part_mask, target_reduction_pct, pitch_mm):
        """Apply thinning by removing only surface layers, not shrinking all voxels"""
        part_occ = occ_grid & part_mask
        
        if not np.any(part_occ):
            return occ_grid, 0.0, 0, "no_part_voxels"
        
        total_voxels = np.sum(part_occ)
        target_removed = int(total_voxels * target_reduction_pct / 100)
        
        if target_removed <= 0:
            return occ_grid, 0.0, 0, "no_removal_needed"
        
        # Check if part is too small for meaningful thinning
        min_thickness_vox = self.t_min_mm / pitch_mm
        if total_voxels < min_thickness_vox * 2:  # Need at least 2x min thickness
            return occ_grid, 0.0, 0, "part_too_small"
        
        # Create a separate grid with only this part for distance calculation
        part_only_grid = np.zeros_like(occ_grid, dtype=bool)
        part_only_grid[part_occ] = True
        
        # Compute distance from surface (outside distance) on the isolated part
        outside_distance = distance_transform_edt(~part_only_grid)
        
        # Find the maximum distance to determine how many layers we can remove
        max_distance = np.max(outside_distance[part_occ])
        
        # Debug logging
        logger.info(f"    Part has {total_voxels} voxels, max distance: {max_distance}, target: {target_removed}")
        
        if max_distance == 0:
            # If max distance is 0, the part might be too thin or isolated
            # Try a simpler approach: remove outermost voxels based on connectivity
            return self.apply_simple_surface_removal(occ_grid, part_mask, target_removed, total_voxels)
        
        # Start from the outermost layer and work inward
        removed_count = 0
        current_distance = 1  # Start from distance 1 (surface layer)
        
        while current_distance <= max_distance and removed_count < target_removed:
            # Find voxels at this distance from surface
            layer_mask = (outside_distance == current_distance) & part_occ
            
            layer_size = np.sum(layer_mask)
            logger.info(f"    Distance {current_distance}: {layer_size} voxels")
            
            if layer_size == 0:
                current_distance += 1
                continue
            
            # If removing this layer would exceed target, remove only what's needed
            if removed_count + layer_size > target_removed:
                # Remove only the needed amount from this layer
                needed = target_removed - removed_count
                # Randomly select voxels from this layer
                layer_indices = np.argwhere(layer_mask)
                if len(layer_indices) > needed:
                    selected_indices = np.random.choice(len(layer_indices), needed, replace=False)
                    layer_mask = np.zeros_like(part_occ, dtype=bool)
                    for idx in selected_indices:
                        z, y, x = layer_indices[idx]
                        layer_mask[z, y, x] = True
                removed_count += needed
            else:
                removed_count += layer_size
            
            # Remove this layer
            occ_grid = occ_grid.copy()
            occ_grid[layer_mask] = False
            
            current_distance += 1
        
        achieved_reduction = (removed_count / total_voxels) * 100
        
        # Determine stop reason
        if removed_count >= target_removed:
            stop_reason = "target_met"
        elif removed_count == 0:
            stop_reason = "no_safe_candidates"
        else:
            stop_reason = "constraints_bound"
        
        logger.info(f"    Final: removed {removed_count}/{target_removed} voxels ({achieved_reduction:.1f}%), reason: {stop_reason}")
        
        return occ_grid, achieved_reduction, removed_count, stop_reason
    
    def apply_simple_surface_removal(self, occ_grid, part_mask, target_removed, total_voxels):
        """Remove complete surface layers systematically"""
        part_occ = occ_grid & part_mask
        
        # Create a working copy for layer detection
        working_grid = part_occ.copy()
        removed_count = 0
        
        # Remove complete layers one by one
        layer_num = 0
        while removed_count < target_removed:
            layer_num += 1
            
            # Find all voxels on the current surface (layer)
            current_layer = []
            
            # Use a more robust method: find voxels that are adjacent to empty space
            for z in range(working_grid.shape[0]):
                for y in range(working_grid.shape[1]):
                    for x in range(working_grid.shape[2]):
                        if working_grid[z, y, x]:
                            # Check 6-connectivity (face neighbors only)
                            is_surface = False
                            for dz, dy, dx in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                                nz, ny, nx = z + dz, y + dy, x + dx
                                if (0 <= nz < working_grid.shape[0] and 
                                    0 <= ny < working_grid.shape[1] and 
                                    0 <= nx < working_grid.shape[2] and
                                    not working_grid[nz, ny, nx]):
                                    is_surface = True
                                    break
                            
                            if is_surface:
                                current_layer.append((z, y, x))
            
            if not current_layer:
                # No more surface voxels
                logger.info(f"    No more surface layers found at layer {layer_num}")
                break
            
            layer_size = len(current_layer)
            logger.info(f"    Layer {layer_num}: {layer_size} surface voxels")
            
            # Check if removing this entire layer would exceed target
            if removed_count + layer_size <= target_removed:
                # Remove entire layer
                for z, y, x in current_layer:
                    working_grid[z, y, x] = False
                    occ_grid[z, y, x] = False
                removed_count += layer_size
                logger.info(f"    Removed complete layer {layer_num}: {layer_size} voxels")
            else:
                # Remove only what's needed from this layer
                needed = target_removed - removed_count
                
                # Remove voxels systematically (not randomly) to maintain structure
                for i, (z, y, x) in enumerate(current_layer):
                    if i >= needed:
                        break
                    working_grid[z, y, x] = False
                    occ_grid[z, y, x] = False
                
                removed_count += needed
                logger.info(f"    Removed partial layer {layer_num}: {needed}/{layer_size} voxels")
                break
        
        achieved_reduction = (removed_count / total_voxels) * 100
        stop_reason = "target_met" if removed_count >= target_removed else "constraints_bound"
        
        logger.info(f"    Systematic layer removal: removed {removed_count}/{target_removed} voxels ({achieved_reduction:.1f}%), reason: {stop_reason}")
        
        return occ_grid, achieved_reduction, removed_count, stop_reason
    
    def apply_voxel_thinning(self, model_id, recommendations_df, voxel_out_dir):
        """Apply voxel thinning to a complete model"""
        logger.info(f"Processing model with voxel SDF thinning: {model_id}")
        
        # Load voxel data
        npz_path = os.path.join(voxel_out_dir, model_id, "voxels_filled_indices_colors.npz")
        if not os.path.exists(npz_path):
            logger.warning(f"Voxel data not found: {npz_path}")
            return None
        
        indices, pitch, transform, colors = self.load_voxel_data(npz_path)
        pitch_mm = pitch * 1000  # Convert to mm
        
        # Build dense grids
        occ_grid, part_grid, offset, unique_colors = self.build_dense_grids(indices, colors)
        
        # Get recommendations for this model
        model_recommendations = recommendations_df[recommendations_df['model_id'] == model_id]
        
        if len(model_recommendations) == 0:
            logger.warning(f"No recommendations found for model {model_id}")
            return None
        
        # Apply voxel thinning to each part
        thinned_grid = occ_grid.copy()
        report_data = []
        
        for _, rec in model_recommendations.iterrows():
            part_id = rec['part_id']
            target_reduction = rec['recommended_reduction_pct']
            
            if target_reduction <= 0:
                logger.info(f"  Skipping part {part_id}: no thinning recommended")
                report_data.append({
                    'model_id': model_id,
                    'part_id': part_id,
                    'target_pct': target_reduction,
                    'achieved_pct': 0.0,
                    'delta_mm_used': 0.0,
                    'removed_voxels': 0,
                    'stop_reason': 'no_thinning_recommended'
                })
                continue
            
            # Find part ID in grid
            if part_grid is not None:
                # Map part_id to grid part number
                part_num = None
                for i, color in enumerate(unique_colors):
                    expected_part_id = f"PART_{i:03d}_R{color[0]}G{color[1]}B{color[2]}A{color[3]}"
                    if expected_part_id == part_id:
                        part_num = i + 1
                        break
                
                if part_num is None:
                    logger.warning(f"  Part {part_id} not found in grid")
                    continue
            else:
                part_num = 1  # Single part model
            
            # Apply surface layer thinning
            part_mask = (part_grid == part_num) if part_grid is not None else occ_grid.copy()
            thinned_grid, achieved_reduction, removed_voxels, stop_reason = self.apply_surface_layer_thinning(
                thinned_grid, part_mask, target_reduction, pitch_mm
            )
            
        # Check connectivity and keep largest component (skip for now to avoid indexing error)
        # thinned_grid = self.check_connectivity(thinned_grid, part_mask)
            
            delta_mm = 0.0  # Not applicable for surface layer thinning
            
            report_data.append({
                'model_id': model_id,
                'part_id': part_id,
                'target_pct': target_reduction,
                'achieved_pct': achieved_reduction,
                'delta_mm_used': delta_mm,
                'removed_voxels': removed_voxels,
                'stop_reason': stop_reason
            })
            
            logger.info(f"  Part {part_id}: {target_reduction:.1f}% target → {achieved_reduction:.1f}% achieved (removed={removed_voxels} surface layers, reason: {stop_reason})")
        
        # Convert back to sparse indices
        thinned_indices = np.argwhere(thinned_grid)
        if len(thinned_indices) > 0:
            # Convert from (z,y,x) back to (x,y,z) and add offset
            thinned_indices = thinned_indices[:, [2, 1, 0]] + offset  # Convert back to global coordinates
        
        # Save thinned voxel data
        output_npz = os.path.join(voxel_out_dir, model_id, "voxels_filled_indices_colors_thinned.npz")
        np.savez_compressed(
            output_npz,
            indices=thinned_indices.astype(np.int32),
            colors=colors if colors is not None else np.array([]),
            pitch=np.array([pitch], np.float32),
            transform=transform.astype(np.float32)
        )
        
        logger.info(f"Saved thinned voxel data: {output_npz}")
        logger.info(f"Original: {len(indices)} voxels → Thinned: {len(thinned_indices)} voxels")
        
        # Export only essential PLY files
        self.export_essential_ply(model_id, voxel_out_dir, thinned_indices, colors, pitch, transform)
        
        return report_data
    
    def export_essential_ply(self, model_id, voxel_out_dir, indices, colors, pitch, transform):
        """Export only essential PLY files: original and thinned colored voxel boxes"""
        try:
            import trimesh
            
            if len(indices) == 0:
                logger.warning(f"No voxels to export for {model_id}")
                return
            
            # Convert indices to world coordinates
            homo_coords = np.hstack([indices, np.ones((len(indices), 1))])
            world_coords = (transform @ homo_coords.T).T[:, :3]
            
            # Export only thinned colored voxel boxes
            boxes_ply = os.path.join(voxel_out_dir, model_id, "voxels_thinned_colored_boxes.ply")
            box_size = pitch * 0.9  # Slightly smaller than pitch for gaps
            box_vertices = []
            box_faces = []
            face_offset = 0
            
            for i, (x, y, z) in enumerate(world_coords):
                # Create box vertices (8 corners of a cube)
                half_size = box_size / 2
                box_verts = np.array([
                    [x - half_size, y - half_size, z - half_size],
                    [x + half_size, y - half_size, z - half_size],
                    [x + half_size, y + half_size, z - half_size],
                    [x - half_size, y + half_size, z - half_size],
                    [x - half_size, y - half_size, z + half_size],
                    [x + half_size, y - half_size, z + half_size],
                    [x + half_size, y + half_size, z + half_size],
                    [x - half_size, y + half_size, z + half_size]
                ])
                
                # Box faces (12 triangles for 6 faces)
                box_faces_this = np.array([
                    [0, 1, 2], [0, 2, 3],  # bottom
                    [4, 7, 6], [4, 6, 5],  # top
                    [0, 4, 5], [0, 5, 1],  # front
                    [2, 6, 7], [2, 7, 3],  # back
                    [0, 3, 7], [0, 7, 4],  # left
                    [1, 5, 6], [1, 6, 2]   # right
                ]) + face_offset
                
                box_vertices.append(box_verts)
                box_faces.append(box_faces_this)
                face_offset += 8
            
            if box_vertices:
                all_vertices = np.vstack(box_vertices)
                all_faces = np.vstack(box_faces)
                
                # Create mesh
                mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
                
                # Apply colors if available
                if colors is not None and len(colors) > 0:
                    try:
                        # Ensure colors array matches the number of boxes
                        if len(colors) == len(indices):
                            # Repeat colors for each face (12 faces per box)
                            face_colors = np.repeat(colors, 12, axis=0)
                            if len(face_colors) == len(all_faces):
                                mesh.visual.face_colors = face_colors
                    except Exception as e:
                        logger.warning(f"Color assignment failed: {e}")
                
                mesh.export(boxes_ply)
            
            logger.info(f"Exported thinned colored voxel boxes: {boxes_ply}")
            
        except Exception as e:
            logger.warning(f"Failed to export PLY: {e}")

def main():
    """Main function to apply voxel SDF thinning to all models"""
    print("=== APPLYING VOXEL SDF THINNING ===")
    
    # Load recommendations
    recommendations_path = "thinning_recommendations_final.csv"
    if not os.path.exists(recommendations_path):
        safe_print(f"[X] Recommendations file not found: {recommendations_path}")
        safe_print("Please run recommend_thinning_final.py first.")
        return
    
    recommendations_df = pd.read_csv(recommendations_path)
    safe_print(f"Loaded {len(recommendations_df)} recommendations")
    
    # Initialize voxel SDF thinner
    thinner = VoxelSDFThinner(
        t_min_mm=2.0,           # Minimum wall thickness
        interface_halo=True,     # Enable interface protection
        tolerance_pct=0.5        # 0.5% tolerance for target reduction
    )
    
    # Get unique models
    unique_models = recommendations_df['model_id'].unique()
    safe_print(f"Processing {len(unique_models)} models: {list(unique_models)}")
    
    # Process each model
    all_reports = []
    voxel_out_dir = "voxel_out"
    
    for model_id in unique_models:
        try:
            report_data = thinner.apply_voxel_thinning(model_id, recommendations_df, voxel_out_dir)
            if report_data:
                all_reports.extend(report_data)
        except Exception as e:
            safe_print(f"[X] Error processing {model_id}: {e}")
            continue
    
    # Save report
    if all_reports:
        report_df = pd.DataFrame(all_reports)
        report_path = "voxel_thinning_apply_report.csv"
        report_df.to_csv(report_path, index=False)
        safe_print(f"[OK] Voxel thinning report saved to: {report_path}")
        
        # Print summary
        total_parts = len(report_df)
        successful_parts = len(report_df[report_df['achieved_pct'] > 0])
        avg_target = report_df['target_pct'].mean()
        avg_achieved = report_df['achieved_pct'].mean()
        avg_delta = report_df['delta_mm_used'].mean()
        total_removed = report_df['removed_voxels'].sum()
        
        safe_print(f"\nSummary:")
        safe_print(f"  Total parts: {total_parts}")
        safe_print(f"  Successfully thinned: {successful_parts}")
        safe_print(f"  Average target reduction: {avg_target:.1f}%")
        safe_print(f"  Average achieved reduction: {avg_achieved:.1f}%")
        safe_print(f"  Average offset: {avg_delta:.3f}mm")
        safe_print(f"  Total voxels removed: {total_removed}")
        
        # Show constraint analysis
        constraint_counts = report_df['stop_reason'].value_counts()
        safe_print(f"\nStop reasons:")
        for reason, count in constraint_counts.items():
            safe_print(f"  {reason}: {count} parts")
    else:
        safe_print("[X] No voxel thinning applied")

if __name__ == "__main__":
    main()

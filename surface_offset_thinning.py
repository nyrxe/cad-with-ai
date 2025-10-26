#!/usr/bin/env python3
"""
Surface Offset Thinning
Implements SDF-based surface offset thinning that slices the outer shell inwards by a specified offset,
producing a watertight thinned surface rather than smaller cubes.
"""

import numpy as np
import pandas as pd
import os
import logging
from scipy.ndimage import distance_transform_edt, label
from collections import defaultdict
from safe_print import safe_print

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SurfaceOffsetThinner:
    """Surface offset thinning using SDF-based approach"""
    
    def __init__(self, t_min_mm=2.0, interface_halo=True, tolerance_pct=0.5):
        """
        Initialize the surface offset thinner.
        
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
    
    def check_min_thickness_constraint(self, occ_grid, part_mask, pitch_mm, offset_mm):
        """Check if offset respects minimum thickness constraint"""
        if offset_mm <= 0:
            return True, 0.0
        
        # Compute SDF for the part
        part_occ = occ_grid & part_mask
        if not np.any(part_occ):
            return True, 0.0
        
        sdf = self.compute_sdf(part_occ)
        offset_vox = offset_mm / pitch_mm
        
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
            max_safe_offset = np.max(thinned_sdf[thinned_mask]) * pitch_mm
            return False, max_safe_offset
        
        return True, offset_mm
    
    def bisection_search_offset(self, occ_grid, part_grid, part_id, target_reduction_pct, pitch_mm):
        """Find offset that achieves target reduction percentage using bisection search"""
        part_mask = (part_grid == part_id) if part_grid is not None else occ_grid.copy()
        
        if not np.any(part_mask):
            return 0.0, 0.0, "no_part_voxels"
        
        # Create interface halo if enabled
        interface_halo = self.create_interface_halo(part_grid, part_id)
        
        # Compute SDF for the part
        part_occ = occ_grid & part_mask
        sdf = self.compute_sdf(part_occ)
        
        # Check if part is too small for meaningful thinning
        total_voxels = np.sum(part_occ)
        min_thickness_vox = self.t_min_mm / pitch_mm
        
        # If part is smaller than minimum thickness, no thinning possible
        if total_voxels < min_thickness_vox * 2:  # Need at least 2x min thickness
            return 0.0, 0.0, "part_too_small"
        
        # Initial bounds
        offset_min = 0.0
        offset_max = 10.0 * pitch_mm  # Start with 10 voxels as max
        
        # Target volume reduction
        target_removed = int(total_voxels * target_reduction_pct / 100)
        
        if target_removed <= 0:
            return 0.0, 0.0, "no_removal_needed"
        
        # Bisection search
        max_iterations = 20
        tolerance = self.tolerance_pct / 100
        
        for iteration in range(max_iterations):
            offset_mid = (offset_min + offset_max) / 2
            
            # Check minimum thickness constraint
            thickness_ok, safe_offset = self.check_min_thickness_constraint(
                occ_grid, part_mask, pitch_mm, offset_mid
            )
            
            if not thickness_ok:
                offset_max = safe_offset
                if offset_max <= offset_min:
                    break
                continue
            
            # Apply offset (excluding interface halo)
            offset_vox = offset_mid / pitch_mm
            thinned_mask = (sdf >= offset_vox) & (~interface_halo)
            
            # Count removed voxels
            removed_voxels = np.sum(part_occ) - np.sum(thinned_mask)
            achieved_reduction = removed_voxels / total_voxels
            
            # Check convergence
            if abs(achieved_reduction - target_reduction_pct/100) < tolerance:
                return offset_mid, achieved_reduction * 100, "target_met"
            
            # Update bounds
            if achieved_reduction < target_reduction_pct/100:
                offset_min = offset_mid
            else:
                offset_max = offset_mid
            
            # Check if bounds converged
            if offset_max - offset_min < 0.01 * pitch_mm:
                break
        
        # Return final result
        final_offset = (offset_min + offset_max) / 2
        final_thinned = (sdf >= final_offset/pitch_mm) & (~interface_halo)
        final_removed = np.sum(part_occ) - np.sum(final_thinned)
        final_reduction = final_removed / total_voxels * 100
        
        # Validate result - if offset is 0 but reduction is high, something's wrong
        if final_offset < 0.001 * pitch_mm and final_reduction > 50:  # Less than 0.1% of pitch but >50% reduction
            return 0.0, 0.0, "invalid_result"
        
        # Determine stop reason
        if final_reduction < target_reduction_pct - self.tolerance_pct:
            if thickness_ok:
                stop_reason = "constraints_bound"
            else:
                stop_reason = "min_thickness_bound"
        else:
            stop_reason = "target_met"
        
        return final_offset, final_reduction, stop_reason
    
    def apply_surface_offset_thinning(self, model_id, recommendations_df, voxel_out_dir):
        """Apply surface offset thinning to a complete model"""
        logger.info(f"Processing model with surface offset: {model_id}")
        
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
        
        # Apply surface offset thinning to each part
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
                    'offset_mm': 0.0,
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
            
            # Apply surface offset thinning
            offset_mm, achieved_reduction, stop_reason = self.bisection_search_offset(
                thinned_grid, part_grid, part_num, target_reduction, pitch_mm
            )
            
            # Apply the final offset to the grid
            if offset_mm > 0:
                part_mask = (part_grid == part_num) if part_grid is not None else occ_grid.copy()
                part_occ = thinned_grid & part_mask
                sdf = self.compute_sdf(part_occ)
                interface_halo = self.create_interface_halo(part_grid, part_num)
                
                offset_vox = offset_mm / pitch_mm
                thinned_part = (sdf >= offset_vox) & (~interface_halo)
                
                # Update the global grid
                thinned_grid = thinned_grid.copy()
                thinned_grid[part_mask] = thinned_part[part_mask]
            
            report_data.append({
                'model_id': model_id,
                'part_id': part_id,
                'target_pct': target_reduction,
                'achieved_pct': achieved_reduction,
                'offset_mm': offset_mm,
                'stop_reason': stop_reason
            })
            
            logger.info(f"  Part {part_id}: {target_reduction:.1f}% target → {achieved_reduction:.1f}% achieved (Δ={offset_mm:.3f}mm, reason: {stop_reason})")
        
        # Export watertight surface mesh
        self.export_watertight_surface(model_id, voxel_out_dir, thinned_grid, offset, pitch, transform)
        
        return report_data
    
    def export_watertight_surface(self, model_id, voxel_out_dir, occ_grid, offset, pitch, transform):
        """Export watertight surface mesh using marching cubes"""
        try:
            import trimesh
            from skimage.measure import marching_cubes
            
            if not np.any(occ_grid):
                logger.warning(f"No voxels to export for {model_id}")
                return
            
            # Add 1-voxel zero padding border
            pad_border_voxels = 1
            padded_shape = tuple(s + 2*pad_border_voxels for s in occ_grid.shape)
            padded_grid = np.zeros(padded_shape, dtype=bool)
            
            # Copy occupancy grid to padded grid
            padded_grid[pad_border_voxels:-pad_border_voxels,
                       pad_border_voxels:-pad_border_voxels,
                       pad_border_voxels:-pad_border_voxels] = occ_grid
            
            # Generate marching cubes mesh
            try:
                vertices, faces, normals, values = marching_cubes(
                    padded_grid, level=0.5, spacing=(pitch, pitch, pitch)
                )
                
                # Remove padding offset from vertices
                vertices -= pad_border_voxels * pitch
                
                # Apply original offset to vertices
                vertices += offset * pitch
                
                # Apply transform to vertices
                homo_vertices = np.hstack([vertices, np.ones((len(vertices), 1))])
                world_vertices = (transform @ homo_vertices.T).T[:, :3]
                
                # Create trimesh object
                mesh = trimesh.Trimesh(vertices=world_vertices, faces=faces)
                
                # Postprocess mesh for watertight surface
                mesh = self.postprocess_mesh(mesh)
                
                # Export watertight surface
                surface_ply = os.path.join(voxel_out_dir, model_id, "voxels_thinned_surface.ply")
                mesh.export(surface_ply)
                
                logger.info(f"Exported watertight surface: {surface_ply}")
                logger.info(f"Mesh: {len(world_vertices)} vertices, {len(faces)} faces")
                
            except Exception as e:
                logger.warning(f"Marching cubes failed: {e}")
            
        except ImportError as e:
            logger.warning(f"Required libraries not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to export watertight surface: {e}")
    
    def postprocess_mesh(self, mesh):
        """Postprocess mesh for watertight surface"""
        try:
            # Remove degenerate faces
            mesh.remove_degenerate_faces()
            
            # Merge duplicate vertices
            mesh.merge_vertices()
            
            # Fix normals and winding
            mesh.fix_normals()
            
            # Remove non-manifold edges
            mesh.remove_duplicate_faces()
            
            # Fill small holes
            mesh.fill_holes()
            
            # Keep largest component only
            if len(mesh.split()) > 1:
                components = mesh.split()
                largest_component = max(components, key=len)
                mesh = largest_component
            
            return mesh
            
        except Exception as e:
            logger.warning(f"Mesh postprocessing failed: {e}")
            return mesh

def main():
    """Main function to apply surface offset thinning to all models"""
    print("=== APPLYING SURFACE OFFSET THINNING ===")
    
    # Load recommendations
    recommendations_path = "thinning_recommendations_final.csv"
    if not os.path.exists(recommendations_path):
        safe_print(f"[X] Recommendations file not found: {recommendations_path}")
        safe_print("Please run recommend_thinning_final.py first.")
        return
    
    recommendations_df = pd.read_csv(recommendations_path)
    safe_print(f"Loaded {len(recommendations_df)} recommendations")
    
    # Initialize surface offset thinner
    thinner = SurfaceOffsetThinner(
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
            report_data = thinner.apply_surface_offset_thinning(model_id, recommendations_df, voxel_out_dir)
            if report_data:
                all_reports.extend(report_data)
        except Exception as e:
            safe_print(f"[X] Error processing {model_id}: {e}")
            continue
    
    # Save report
    if all_reports:
        report_df = pd.DataFrame(all_reports)
        report_path = "surface_offset_thinning_report.csv"
        report_df.to_csv(report_path, index=False)
        safe_print(f"[OK] Surface offset thinning report saved to: {report_path}")
        
        # Print summary
        total_parts = len(report_df)
        successful_parts = len(report_df[report_df['achieved_pct'] > 0])
        avg_target = report_df['target_pct'].mean()
        avg_achieved = report_df['achieved_pct'].mean()
        avg_offset = report_df['offset_mm'].mean()
        
        safe_print(f"\nSummary:")
        safe_print(f"  Total parts: {total_parts}")
        safe_print(f"  Successfully thinned: {successful_parts}")
        safe_print(f"  Average target reduction: {avg_target:.1f}%")
        safe_print(f"  Average achieved reduction: {avg_achieved:.1f}%")
        safe_print(f"  Average offset: {avg_offset:.3f}mm")
        
        # Show constraint analysis
        constraint_counts = report_df['stop_reason'].value_counts()
        safe_print(f"\nStop reasons:")
        for reason, count in constraint_counts.items():
            safe_print(f"  {reason}: {count} parts")
    else:
        safe_print("[X] No surface offset thinning applied")

if __name__ == "__main__":
    main()

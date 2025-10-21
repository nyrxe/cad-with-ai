#!/usr/bin/env python3
"""
Apply AI Thinning to Voxels
Applies recommended thinning from AI to voxel models and exports modified geometry.
"""

import numpy as np
import pandas as pd
import os
import glob
from scipy.ndimage import distance_transform_edt, binary_erosion, label
from collections import defaultdict
import logging
from safe_print import safe_print

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIThinningApplier:
    """Apply AI thinning recommendations to voxel models"""
    
    def __init__(self, k_surface=1, k_interface=1, t_min_mm=2.0, stress_percentile=80, 
                 batch_size_pct=0.5, connectivity_check=True):
        """
        Initialize the AI thinning applier.
        
        Args:
            k_surface (int): Voxels to protect at exterior surfaces
            k_interface (int): Voxels to protect at part boundaries  
            t_min_mm (float): Minimum wall thickness in mm
            stress_percentile (float): Percentile for stress protection
            batch_size_pct (float): Batch size as percentage of total voxels
            connectivity_check (bool): Enforce 6-connectivity
        """
        self.k_surface = k_surface
        self.k_interface = k_interface
        self.t_min_mm = t_min_mm
        self.stress_percentile = stress_percentile
        self.batch_size_pct = batch_size_pct
        self.connectivity_check = connectivity_check
        
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
        """Build dense occupancy and part grids from sparse indices"""
        # Get grid dimensions
        imin, jmin, kmin = indices.min(axis=0)
        imax, jmax, kmax = indices.max(axis=0)
        grid_shape = (imax - imin + 1, jmax - jmin + 1, kmax - kmin + 1)
        offset = np.array([imin, jmin, kmin])
        
        # Build occupancy grid
        occ_grid = np.zeros(grid_shape, dtype=bool)
        for idx in indices:
            x, y, z = idx - offset
            if (0 <= x < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2]):
                occ_grid[x, y, z] = True
        
        # Build part grid if colors available
        part_grid = None
        if colors is not None:
            part_grid = np.zeros(grid_shape, dtype=np.int32)
            unique_colors, color_indices = np.unique(colors, axis=0, return_inverse=True)
            
            for i, idx in enumerate(indices):
                x, y, z = idx - offset
                if (0 <= x < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2]):
                    part_grid[x, y, z] = color_indices[i] + 1  # 1-based part IDs
        
        return occ_grid, part_grid, offset, unique_colors if colors is not None else None
    
    def create_protection_masks(self, occ_grid, part_grid, pitch, stress_data=None):
        """Create protection masks for various constraints"""
        protection = np.zeros_like(occ_grid, dtype=bool)
        total_voxels = np.sum(occ_grid)
        
        # 1. Surface protection (distance to air) - DISABLED by default
        if self.k_surface > 0 and np.any(occ_grid):
            # Distance transform from air (inverted occupancy)
            air_distance = distance_transform_edt(~occ_grid)
            surface_mask = air_distance <= self.k_surface
            protection |= surface_mask
            logger.info(f"Surface protection: {np.sum(surface_mask)} voxels ({np.sum(surface_mask)/total_voxels*100:.1f}%)")
        
        # 2. Interface protection (distance to different parts) - DISABLED by default
        if self.k_interface > 0 and part_grid is not None and np.any(part_grid > 0):
            interface_mask = np.zeros_like(occ_grid, dtype=bool)
            
            # For each part, find distance to other parts
            unique_parts = np.unique(part_grid[part_grid > 0])
            for part_id in unique_parts:
                part_mask = (part_grid == part_id)
                other_parts_mask = (part_grid > 0) & (part_grid != part_id)
                
                if np.any(other_parts_mask):
                    # Distance from this part to other parts
                    part_distance = distance_transform_edt(~other_parts_mask)
                    part_interface = part_mask & (part_distance <= self.k_interface)
                    interface_mask |= part_interface
            
            protection |= interface_mask
            logger.info(f"Interface protection: {np.sum(interface_mask)} voxels ({np.sum(interface_mask)/total_voxels*100:.1f}%)")
        
        # 3. Minimum thickness protection - REDUCED by default
        t_min_voxels = max(1, int(np.ceil(self.t_min_mm / (pitch * 1000))))  # Convert mm to voxels
        if t_min_voxels > 1:
            # For each voxel, check if it has sufficient thickness in all directions
            thickness_mask = np.zeros_like(occ_grid, dtype=bool)
            
            for x in range(occ_grid.shape[0]):
                for y in range(occ_grid.shape[1]):
                    for z in range(occ_grid.shape[2]):
                        if occ_grid[x, y, z]:
                            # Check thickness in 6 directions
                            directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
                            min_thickness = float('inf')
                            
                            for dx, dy, dz in directions:
                                thickness = 0
                                nx, ny, nz = x, y, z
                                for _ in range(t_min_voxels):
                                    nx, ny, nz = nx + dx, ny + dy, nz + dz
                                    if (0 <= nx < occ_grid.shape[0] and 
                                        0 <= ny < occ_grid.shape[1] and 
                                        0 <= nz < occ_grid.shape[2] and
                                        occ_grid[nx, ny, nz]):
                                        thickness += 1
                                    else:
                                        break
                                min_thickness = min(min_thickness, thickness)
                            
                            if min_thickness < t_min_voxels:
                                thickness_mask[x, y, z] = True
            
            protection |= thickness_mask
            logger.info(f"Min thickness protection: {np.sum(thickness_mask)} voxels ({np.sum(thickness_mask)/total_voxels*100:.1f}%)")
        
        # 4. Stress protection (if stress data available) - DISABLED by default
        if stress_data is not None and self.stress_percentile < 100:
            stress_threshold = np.percentile(stress_data, self.stress_percentile)
            high_stress_mask = stress_data > stress_threshold
            protection |= high_stress_mask
            logger.info(f"Stress protection: {np.sum(high_stress_mask)} voxels ({np.sum(high_stress_mask)/total_voxels*100:.1f}%)")
        
        total_protection = np.sum(protection)
        logger.info(f"Total protection: {total_protection} voxels ({total_protection/total_voxels*100:.1f}%)")
        
        # If protection is too high, warn user
        if total_protection > total_voxels * 0.8:  # More than 80% protected
            logger.warning(f"High protection ratio ({total_protection/total_voxels*100:.1f}%) - consider reducing constraints")
        
        return protection
    
    def get_candidate_voxels(self, occ_grid, part_grid, part_id, protection_mask):
        """Get candidate voxels for removal (outermost, non-protected)"""
        if part_grid is None:
            # No part information, work on all voxels
            part_mask = occ_grid.copy()
        else:
            part_mask = (part_grid == part_id) & occ_grid
        
        if not np.any(part_mask):
            return np.array([])
        
        # Find outermost voxels (largest distance to air)
        air_distance = distance_transform_edt(~occ_grid)
        part_distance = air_distance * part_mask.astype(float)
        
        # Get candidates (non-protected, part voxels)
        candidates = part_mask & (~protection_mask)
        
        if not np.any(candidates):
            return np.array([])
        
        # Sort by distance (outermost first - ascending distance)
        candidate_coords = np.argwhere(candidates)
        candidate_distances = part_distance[candidates]
        sorted_indices = np.argsort(candidate_distances)  # Ascending order (smallest distance first)
        
        return candidate_coords[sorted_indices]
    
    def check_connectivity(self, occ_grid, part_mask):
        """Check 6-connectivity of a part"""
        if not self.connectivity_check:
            return True
        
        # Extract part voxels
        part_occ = occ_grid & part_mask
        
        # Label connected components
        labeled, num_components = label(part_occ, structure=self.structuring_element)
        
        return num_components <= 1
    
    def check_local_thickness(self, occ_grid, x, y, z, pitch):
        """Check if voxel has sufficient local thickness"""
        t_min_voxels = max(1, int(np.ceil(self.t_min_mm / (pitch * 1000))))
        
        directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        
        for dx, dy, dz in directions:
            thickness = 0
            nx, ny, nz = x, y, z
            for _ in range(t_min_voxels):
                nx, ny, nz = nx + dx, ny + dy, nz + dz
                if (0 <= nx < occ_grid.shape[0] and 
                    0 <= ny < occ_grid.shape[1] and 
                    0 <= nz < occ_grid.shape[2] and
                    occ_grid[nx, ny, nz]):
                    thickness += 1
                else:
                    break
            
            if thickness >= t_min_voxels:
                return True
        
        return False
    
    def apply_thinning_to_part(self, occ_grid, part_grid, part_id, target_reduction_pct, 
                              protection_mask, pitch, stress_data=None):
        """Apply thinning to a specific part"""
        if part_grid is None:
            part_mask = occ_grid.copy()
        else:
            part_mask = (part_grid == part_id) & occ_grid
        
        if not np.any(part_mask):
            return occ_grid, 0, "no_part_voxels"
        
        # Calculate target removal
        total_voxels = np.sum(part_mask)
        target_remove = int(total_voxels * target_reduction_pct / 100)
        
        if target_remove <= 0:
            return occ_grid, 0, "no_removal_needed"
        
        # Work on a copy
        thinned_grid = occ_grid.copy()
        removed_count = 0
        batch_size = max(1, int(total_voxels * self.batch_size_pct / 100))
        
        logger.info(f"Part {part_id}: {total_voxels} voxels, target removal: {target_remove}")
        
        while removed_count < target_remove:
            # Get candidates
            candidates = self.get_candidate_voxels(thinned_grid, part_grid, part_id, protection_mask)
            
            if len(candidates) == 0:
                break
            
            # Process batch
            batch_end = min(len(candidates), batch_size)
            batch_candidates = candidates[:batch_end]
            
            # Try to remove batch
            temp_grid = thinned_grid.copy()
            batch_removed = 0
            
            for x, y, z in batch_candidates:
                # Check if removal is safe
                temp_grid[x, y, z] = False
                
                # Check connectivity
                if not self.check_connectivity(temp_grid, part_mask):
                    temp_grid[x, y, z] = True  # Rollback
                    continue
                
                # Check local thickness
                if not self.check_local_thickness(temp_grid, x, y, z, pitch):
                    temp_grid[x, y, z] = True  # Rollback
                    continue
                
                batch_removed += 1
            
            # Apply batch if successful
            if batch_removed > 0:
                thinned_grid = temp_grid
                removed_count += batch_removed
                logger.info(f"  Removed {batch_removed} voxels, total: {removed_count}/{target_remove}")
            else:
                # No more safe removals
                break
            
            # Reduce batch size if no progress
            if batch_removed == 0:
                batch_size = max(1, batch_size // 2)
                if batch_size == 1:
                    break
        
        achieved_reduction = (removed_count / total_voxels) * 100
        
        # Determine stop reason
        if removed_count >= target_remove:
            stop_reason = "target_met"
        elif removed_count == 0:
            stop_reason = "no_safe_candidates"
        else:
            stop_reason = "constraints_bound"
        
        logger.info(f"  Achieved: {achieved_reduction:.1f}% (reason: {stop_reason})")
        
        return thinned_grid, removed_count, stop_reason
    
    def apply_thinning_to_model(self, model_id, recommendations_df, voxel_out_dir):
        """Apply thinning to a complete model"""
        logger.info(f"Processing model: {model_id}")
        
        # Load voxel data
        npz_path = os.path.join(voxel_out_dir, model_id, "voxels_filled_indices_colors.npz")
        if not os.path.exists(npz_path):
            logger.warning(f"Voxel data not found: {npz_path}")
            return None
        
        indices, pitch, transform, colors = self.load_voxel_data(npz_path)
        
        # Build dense grids
        occ_grid, part_grid, offset, unique_colors = self.build_dense_grids(indices, colors)
        
        # Create protection masks
        protection_mask = self.create_protection_masks(occ_grid, part_grid, pitch)
        
        # Get recommendations for this model
        model_recommendations = recommendations_df[recommendations_df['model_id'] == model_id]
        
        if len(model_recommendations) == 0:
            logger.warning(f"No recommendations found for model {model_id}")
            return None
        
        # Apply thinning to each part
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
            
            # Apply thinning
            thinned_grid, removed_count, stop_reason = self.apply_thinning_to_part(
                thinned_grid, part_grid, part_num, target_reduction, protection_mask, pitch
            )
            
            achieved_reduction = (removed_count / np.sum((part_grid == part_num) & occ_grid)) * 100 if part_grid is not None else 0
            
            report_data.append({
                'model_id': model_id,
                'part_id': part_id,
                'target_pct': target_reduction,
                'achieved_pct': achieved_reduction,
                'removed_voxels': removed_count,
                'stop_reason': stop_reason
            })
        
        # Convert back to sparse indices
        thinned_indices = np.argwhere(thinned_grid)
        if len(thinned_indices) > 0:
            thinned_indices = thinned_indices + offset  # Convert back to global coordinates
        
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
        
        return report_data
    
    def export_visualization_files(self, model_id, voxel_out_dir):
        """Export watertight surface mesh from thinned voxel grid"""
        try:
            import trimesh
            from skimage.measure import marching_cubes
            
            # Load thinned data
            thinned_npz = os.path.join(voxel_out_dir, model_id, "voxels_filled_indices_colors_thinned.npz")
            if not os.path.exists(thinned_npz):
                return
            
            data = np.load(thinned_npz)
            indices = data["indices"]
            colors = data.get("colors", None)
            pitch = float(data["pitch"].item())
            transform = data["transform"]
            
            if len(indices) == 0:
                logger.warning(f"No voxels to export for {model_id}")
                return
            
            # Build dense grid from sparse indices
            # Axis order: (z, y, x) - document and keep consistent
            imin, jmin, kmin = indices.min(axis=0)
            imax, jmax, kmax = indices.max(axis=0)
            grid_shape = (kmax - kmin + 1, jmax - jmin + 1, imax - imin + 1)  # (z, y, x)
            offset = np.array([imin, jmin, kmin])
            
            # Create dense occupancy grid with 1-voxel padding border
            pad_border_voxels = 1
            padded_shape = (grid_shape[0] + 2*pad_border_voxels, 
                           grid_shape[1] + 2*pad_border_voxels, 
                           grid_shape[2] + 2*pad_border_voxels)
            dense_grid = np.zeros(padded_shape, dtype=bool)
            
            # Fill occupied voxels in padded grid
            for idx in indices:
                x, y, z = idx - offset
                if (0 <= x < grid_shape[2] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[0]):
                    # Add padding offset
                    z_pad = z + pad_border_voxels
                    y_pad = y + pad_border_voxels  
                    x_pad = x + pad_border_voxels
                    dense_grid[z_pad, y_pad, x_pad] = True
            
            # Generate marching cubes mesh
            try:
                vertices, faces, normals, values = marching_cubes(dense_grid, level=0.5, spacing=(pitch, pitch, pitch))
                
                # Remove padding offset from vertices
                vertices -= pad_border_voxels * pitch
                
                # Apply original offset to vertices
                vertices += offset * pitch
                
                # Apply transform to vertices (once, after marching cubes)
                homo_vertices = np.hstack([vertices, np.ones((len(vertices), 1))])
                world_vertices = (transform @ homo_vertices.T).T[:, :3]
                
                # Create trimesh object
                mesh = trimesh.Trimesh(vertices=world_vertices, faces=faces)
                
                # Postprocess mesh for watertight surface
                mesh = self.postprocess_mesh(mesh)
                
                # Apply colors if available
                if colors is not None and len(colors) > 0:
                    try:
                        # Sample colors from nearest voxels
                        face_colors = []
                        for face in faces:
                            # Get centroid of triangle
                            centroid = np.mean(world_vertices[face], axis=0)
                            
                            # Find nearest voxel
                            distances = np.linalg.norm(world_vertices - centroid, axis=1)
                            nearest_idx = np.argmin(distances)
                            
                            # Map back to original voxel index
                            if nearest_idx < len(indices):
                                color = colors[nearest_idx]
                                face_colors.append(color)
                            else:
                                face_colors.append([128, 128, 128, 255])  # Default gray
                        
                        if face_colors:
                            mesh.visual.face_colors = np.array(face_colors)
                    except Exception as e:
                        logger.warning(f"Color assignment failed: {e}")
                
                # Export watertight surface
                surface_ply = os.path.join(voxel_out_dir, model_id, "voxels_thinned_surface.ply")
                mesh.export(surface_ply)
                
                logger.info(f"Exported watertight surface: {surface_ply}")
                logger.info(f"Mesh: {len(world_vertices)} vertices, {len(faces)} faces")
                
                # Also export voxel version (point cloud and boxes)
                self.export_voxel_visualization(model_id, voxel_out_dir, indices, colors, pitch, transform)
                
                # Export solid voxel mesh (no gaps, internal faces culled)
                self.export_solid_voxels(model_id, voxel_out_dir, indices, colors, pitch, transform)
                
            except Exception as e:
                logger.warning(f"Marching cubes failed: {e}")
                # Fallback: export as point cloud
                points_ply = os.path.join(voxel_out_dir, model_id, "voxels_thinned_points.ply")
                world_coords = (transform @ np.hstack([indices, np.ones((len(indices), 1))]).T).T[:, :3]
                
                if colors is not None and len(colors) > 0:
                    pc = trimesh.points.PointCloud(vertices=world_coords, colors=colors)
                else:
                    pc = trimesh.points.PointCloud(vertices=world_coords)
                pc.export(points_ply)
                logger.info(f"Exported point cloud fallback: {points_ply}")
            
        except ImportError as e:
            logger.warning(f"Required libraries not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to export visualization files: {e}")
    
    def export_voxel_visualization(self, model_id, voxel_out_dir, indices, colors, pitch, transform):
        """Export voxel-based visualization (points and boxes) for thinned model"""
        try:
            import trimesh
            
            if len(indices) == 0:
                return
            
            # Convert indices to world coordinates
            homo_coords = np.hstack([indices, np.ones((len(indices), 1))])
            world_coords = (transform @ homo_coords.T).T[:, :3]
            
            # Export point cloud
            points_ply = os.path.join(voxel_out_dir, model_id, "voxels_thinned_points.ply")
            if colors is not None and len(colors) > 0:
                pc = trimesh.points.PointCloud(vertices=world_coords, colors=colors)
            else:
                pc = trimesh.points.PointCloud(vertices=world_coords)
            pc.export(points_ply)
            
            # Export voxel boxes
            boxes_ply = os.path.join(voxel_out_dir, model_id, "voxels_thinned_boxes.ply")
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
            
            logger.info(f"Exported voxel visualization: {points_ply}, {boxes_ply}")
            
        except Exception as e:
            logger.warning(f"Failed to export voxel visualization: {e}")
    
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
    
    def export_solid_voxels(self, model_id, voxel_out_dir, indices, colors, pitch, transform):
        """Export solid voxel mesh (no gaps, internal faces culled)"""
        try:
            import trimesh
            
            if len(indices) == 0:
                return
            
            # Convert indices to world coordinates
            homo_coords = np.hstack([indices, np.ones((len(indices), 1))])
            world_coords = (transform @ homo_coords.T).T[:, :3]
            
            # Build voxel grid for face culling
            voxel_set = set()
            for coord in world_coords:
                voxel_set.add(tuple(coord))
            
            # Export solid voxel boxes (exact pitch, no gaps)
            solid_ply = os.path.join(voxel_out_dir, model_id, "voxels_thinned_solid.ply")
            box_vertices = []
            box_faces = []
            face_offset = 0
            
            for i, (x, y, z) in enumerate(world_coords):
                # Create box vertices (8 corners of a cube) - exact pitch
                half_size = pitch / 2
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
                
                # Cull internal faces (where two voxels touch)
                faces_to_keep = []
                for face_idx, face in enumerate(box_faces_this):
                    # Check if this face is internal (touching another voxel)
                    if len(face) >= 3:  # Ensure we have at least 3 vertices
                        face_center = np.mean(box_verts[face], axis=0)
                        face_normal = np.cross(box_verts[face[1]] - box_verts[face[0]], 
                                            box_verts[face[2]] - box_verts[face[0]])
                        face_normal = face_normal / np.linalg.norm(face_normal)
                        
                        # Check if there's a voxel in the direction of the normal
                        neighbor_pos = face_center + face_normal * pitch * 0.1
                        neighbor_voxel = tuple(np.round(neighbor_pos / pitch) * pitch)
                        
                        if neighbor_voxel not in voxel_set:
                            faces_to_keep.append(face)
                
                if faces_to_keep:
                    box_vertices.append(box_verts)
                    box_faces.append(np.array(faces_to_keep))
                    face_offset += 8
            
            if box_vertices:
                all_vertices = np.vstack(box_vertices)
                all_faces = np.vstack(box_faces)
                
                # Create mesh
                mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
                
                # Apply colors if available
                if colors is not None and len(colors) > 0:
                    try:
                        if len(colors) == len(indices):
                            face_colors = np.repeat(colors, len(all_faces) // len(indices), axis=0)
                            if len(face_colors) == len(all_faces):
                                mesh.visual.face_colors = face_colors
                    except Exception as e:
                        logger.warning(f"Color assignment failed: {e}")
                
                mesh.export(solid_ply)
                logger.info(f"Exported solid voxel mesh: {solid_ply}")
            
        except Exception as e:
            logger.warning(f"Failed to export solid voxels: {e}")

def main():
    """Main function to apply AI thinning to all models"""
    print("=== APPLYING AI THINNING TO VOXELS ===")
    
    # Load recommendations
    recommendations_path = "thinning_recommendations_final.csv"
    if not os.path.exists(recommendations_path):
        safe_print(f"[X] Recommendations file not found: {recommendations_path}")
        safe_print("Please run recommend_thinning_final.py first.")
        return
    
    recommendations_df = pd.read_csv(recommendations_path)
    safe_print(f"Loaded {len(recommendations_df)} recommendations")
    
    # Initialize applier with voxel-appropriate safety constraints
    applier = AIThinningApplier(
        k_surface=0,           # No surface protection (too aggressive for small voxels)
        k_interface=0,         # No interface protection (too aggressive for small voxels)
        t_min_mm=0.5,          # Reduced minimum thickness 0.5mm (voxel-appropriate)
        stress_percentile=90,  # Only protect top 10% stress regions
        batch_size_pct=1.0,    # Larger batches for efficiency
        connectivity_check=True # Keep connectivity check for safety
    )
    
    # Get unique models
    unique_models = recommendations_df['model_id'].unique()
    safe_print(f"Processing {len(unique_models)} models: {list(unique_models)}")
    
    # Process each model
    all_reports = []
    voxel_out_dir = "voxel_out"
    
    for model_id in unique_models:
        try:
            report_data = applier.apply_thinning_to_model(model_id, recommendations_df, voxel_out_dir)
            if report_data:
                all_reports.extend(report_data)
                # Export visualization files
                applier.export_visualization_files(model_id, voxel_out_dir)
        except Exception as e:
            safe_print(f"[X] Error processing {model_id}: {e}")
            continue
    
    # Save report
    if all_reports:
        report_df = pd.DataFrame(all_reports)
        report_path = "thinning_apply_report.csv"
        report_df.to_csv(report_path, index=False)
        safe_print(f"[OK] Thinning report saved to: {report_path}")
        
        # Print summary
        total_parts = len(report_df)
        successful_parts = len(report_df[report_df['achieved_pct'] > 0])
        avg_target = report_df['target_pct'].mean()
        avg_achieved = report_df['achieved_pct'].mean()
        
        safe_print(f"\nSummary:")
        safe_print(f"  Total parts: {total_parts}")
        safe_print(f"  Successfully thinned: {successful_parts}")
        safe_print(f"  Average target reduction: {avg_target:.1f}%")
        safe_print(f"  Average achieved reduction: {avg_achieved:.1f}%")
        
        # Show constraint analysis
        constraint_counts = report_df['stop_reason'].value_counts()
        safe_print(f"\nStop reasons:")
        for reason, count in constraint_counts.items():
            safe_print(f"  {reason}: {count} parts")
    else:
        safe_print("[X] No thinning applied")

if __name__ == "__main__":
    main()

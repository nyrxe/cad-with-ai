#!/usr/bin/env python3
"""
Local PLY to Voxel Pipeline
Adapted from the working Colab code
Processes multiple PLY files from input/ folder and creates separate output folders
"""

import trimesh
import numpy as np
import os
import glob
from trimesh.proximity import ProximityQuery

def process_single_ply(ply_path, output_base_dir="voxel_out", target_res=64):
    """Process a single PLY file and create its own output folder"""
    # Get filename without extension for folder name
    base_name = os.path.splitext(os.path.basename(ply_path))[0]
    out_dir = os.path.join(output_base_dir, base_name)
    
    # print(f"\n=== Processing: {ply_path} ===")
    # print(f"Output folder: {out_dir}")
    
    # Load mesh
    mesh = trimesh.load(ply_path, process=False)  # keep face colors as-is
    
    # print("verts:", len(mesh.vertices))
    # print("faces:", len(mesh.faces))
    # print("is_watertight:", mesh.is_watertight)
    # print("extents (xyz size):", mesh.extents)
    
    # Check per-face RGBA existence
    has_face_colors = (
        hasattr(mesh, "visual") and
        getattr(mesh.visual, "face_colors", None) is not None and
        len(mesh.visual.face_colors) == len(mesh.faces)
    )
    # print("face colors per face:", has_face_colors)
    # if has_face_colors:
    #     print("face_colors dtype/shape:", mesh.visual.face_colors.dtype, mesh.visual.face_colors.shape)
    
    # Voxelization parameters
    TARGET_RES = target_res  # Use parameter instead of hardcoded value
    longest = float(max(mesh.extents))
    pitch = longest / TARGET_RES
    # print(f"target_res={TARGET_RES}, pitch={pitch:.8f} (mesh units)")
    
    # Create voxel grids
    vg_surface = mesh.voxelized(pitch=pitch)
    vg_filled = vg_surface.fill()
    
    # print("surface grid shape:", vg_surface.shape, "pitch:", vg_surface.pitch)
    # print("filled  grid shape:", vg_filled.shape)
    # print("surface voxels:", len(vg_surface.sparse_indices))
    # print("filled  voxels:", len(vg_filled.sparse_indices))
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Save as box-meshes (handy for quick preview in any mesh viewer)
    vox_mesh_surface = vg_surface.as_boxes()
    vox_mesh_filled  = vg_filled.as_boxes()
    surf_ply  = os.path.join(out_dir, "vox_surface_boxes.ply")
    fill_ply  = os.path.join(out_dir, "vox_filled_boxes.ply")
    vox_mesh_surface.export(surf_ply)
    vox_mesh_filled.export(fill_ply)
    # print("saved:", surf_ply, "and", fill_ply)
    
    # Save sparse occupancy with metadata
    np.savez_compressed(
        os.path.join(out_dir, "vox_occupancy_sparse.npz"),
        indices=vg_filled.sparse_indices,                 # (N, 3) int voxel coords (i,j,k)
        pitch=np.array([vg_filled.pitch], np.float32),    # scalar voxel size
        transform=vg_filled.transform.astype(np.float32)  # 4x4: index -> world coords
    )
    # print("saved:", os.path.join(out_dir, "vox_occupancy_sparse.npz"))
    
    # Color propagation (if mesh has face colors)
    if has_face_colors:
        # print("\n=== COLOR PROPAGATION ===")
        
        # Use the filled voxel grid
        vg = vg_filled
        
        # Occupied voxel indices (i, j, k)
        idx = vg.sparse_indices
        N = len(idx)
        # print(f"Filled voxels: {N}")
        
        # Convert voxel indices -> world coordinates (voxel centers)
        homo = np.hstack([idx, np.ones((N, 1), dtype=idx.dtype)])  # (N,4)
        centers_world = (vg.transform @ homo.T).T[:, :3]            # (N,3)
        # print("centers_world shape:", centers_world.shape)
        
        # Build proximity structure on your original mesh
        pq = ProximityQuery(mesh)
        
        # For very large N, do it in chunks to be memory-safe
        chunk = 200_000
        face_ids = np.empty(N, dtype=np.int64)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            # on_surface returns (points, distances, face_indices)
            _, _, fidx = pq.on_surface(centers_world[s:e])
            face_ids[s:e] = fidx
        
        # Map face -> RGBA
        face_rgba = mesh.visual.face_colors  # (F,4) uint8
        voxel_rgba = face_rgba[face_ids]     # (N,4) uint8
        # print("voxel_rgba shape/dtype:", voxel_rgba.shape, voxel_rgba.dtype)
        
        # Save colored point cloud
        pc = trimesh.points.PointCloud(vertices=centers_world, colors=voxel_rgba)
        colored_points_ply = os.path.join(out_dir, "voxels_filled_colored_points.ply")
        pc.export(colored_points_ply)
        # print("saved:", colored_points_ply)
        
        # Save colored boxes
        colored_boxes_ply = os.path.join(out_dir, "voxels_filled_colored_boxes.ply")
        
        boxes = vg.as_boxes()
        try:
            # Try setting per-face colors uniformly per cube by expanding RGBA
            # Each cube contributes 12 triangles (36 verts) → we broadcast colors
            # Safer approach: set face colors after creation
            F = len(boxes.faces)
            # Build per-face colors by assigning same RGBA to each face of a cube.
            # Each cube has 12 faces; faces are grouped per cube in as_boxes output.
            if F % 12 != 0:
                raise RuntimeError("Unexpected face count grouping; cannot assign per-cube colors safely.")
            faces_per_cube = 12
            C = F // faces_per_cube
            if C != len(voxel_rgba):
                # In some versions, as_boxes can produce a different cube ordering.
                # If counts mismatch, skip coloring to avoid wrong mapping.
                raise RuntimeError("Cube count doesn't match voxel count; skipping color assignment.")

            # Repeat each voxel color 12 times (one per face of the cube)
            face_colors = np.repeat(voxel_rgba[:C], faces_per_cube, axis=0)
            boxes.visual.face_colors = face_colors
            boxes.export(colored_boxes_ply)
            # print("saved:", colored_boxes_ply)
        except Exception as e:
            # print("Could not assign colors to boxes reliably:", e)
            # Save uncolored boxes as a fallback
            fallback_ply = os.path.join(out_dir, "voxels_filled_boxes_uncolored.ply")
            boxes.export(fallback_ply)
            # print("saved:", fallback_ply, "(uncolored)")
    
    # Save voxel indices and colors for FEA analysis
    if has_face_colors:
        # Use the voxel_rgba from color propagation
        np.savez_compressed(
            os.path.join(out_dir, "voxels_filled_indices_colors.npz"),
            indices=vg_filled.sparse_indices.astype(np.int32),          # (N,3)
            colors=voxel_rgba.astype(np.uint8),                        # (N,4)
            pitch=np.array([vg_filled.pitch], np.float32),
            transform=vg_filled.transform.astype(np.float32)  # 4x4
        )
        # print("saved:", os.path.join(out_dir, "voxels_filled_indices_colors.npz"))
    else:
        # Create default colors (white) for FEA analysis even without face colors
        num_voxels = len(vg_filled.sparse_indices)
        default_colors = np.full((num_voxels, 4), 255, dtype=np.uint8)  # White RGBA
        np.savez_compressed(
            os.path.join(out_dir, "voxels_filled_indices_colors.npz"),
            indices=vg_filled.sparse_indices.astype(np.int32),          # (N,3)
            colors=default_colors,                                     # (N,4) white
            pitch=np.array([vg_filled.pitch], np.float32),
            transform=vg_filled.transform.astype(np.float32)  # 4x4
        )
        # print("saved:", os.path.join(out_dir, "voxels_filled_indices_colors.npz"), "(with default white colors)")
    
    # print(f"\n=== COMPLETE for {base_name} ===")
    # print(f"All outputs saved to: {out_dir}/")
    # print("Files created:")
    # for file in os.listdir(out_dir):
    #     file_path = os.path.join(out_dir, file)
    #     size = os.path.getsize(file_path)
    #     print(f"  - {file} ({size:,} bytes)")
    
    return out_dir

def main():
    import sys
    
    # Parse command line arguments
    target_res = 32  # Default resolution (optimal balance of speed and accuracy)
    if len(sys.argv) > 1:
        try:
            target_res = int(sys.argv[1])
            # print(f"Using custom resolution: {target_res}")
        except ValueError:
            pass  # Use default resolution
    
    # Find PLY files in input directory
    INPUT_DIR = "input"
    OUTPUT_BASE_DIR = "voxel_out"
    
    # Create input directory if it doesn't exist
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        # print(f"Created {INPUT_DIR} directory. Please add your .ply files there.")
        return
    
    # print(f"Looking for PLY files in: {INPUT_DIR}")
    # print("Files in input directory:", os.listdir(INPUT_DIR))
    
    ply_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.ply")))
    if not ply_files:
        # print(f"No .ply files found in {INPUT_DIR} directory!")
        # print("Please add your .ply files to the 'input' folder.")
        return
    
    # print(f"Found {len(ply_files)} PLY files:", [os.path.basename(f) for f in ply_files])
    
    # Process each PLY file
    processed_dirs = []
    for ply_path in ply_files:
        try:
            output_dir = process_single_ply(ply_path, OUTPUT_BASE_DIR, target_res)
            processed_dirs.append(output_dir)
        except Exception as e:
            # print(f"Error processing {ply_path}: {e}")
            continue
    
    # print(f"\n*** PROCESSING COMPLETE! ***")
    # print(f"Processed {len(processed_dirs)} files successfully:")
    # for dir_path in processed_dirs:
    #     print(f"  - {dir_path}")

if __name__ == "__main__":
    main()
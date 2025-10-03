from __future__ import annotations
import os
import numpy as np
import trimesh
from typing import Optional, Tuple


try:
from trimesh.proximity import ProximityQuery
_HAS_RTREE = True
except Exception:
_HAS_RTREE = False




class VoxelizeResult:
def __init__(self,
vg: trimesh.voxel.VoxelGrid,
centers_world: Optional[np.ndarray],
voxel_rgba: Optional[np.ndarray],
used_pitch: float,
surface_boxes: trimesh.Trimesh,
filled_boxes: Optional[trimesh.Trimesh]):
self.vg = vg
self.centers_world = centers_world
self.voxel_rgba = voxel_rgba
self.used_pitch = used_pitch
self.surface_boxes = surface_boxes
self.filled_boxes = filled_boxes




def load_mesh(path: str) -> trimesh.Trimesh:
mesh = trimesh.load(path, process=False)
if not isinstance(mesh, trimesh.Trimesh):
raise TypeError(f"Expected a Trimesh; got {type(mesh)} from {path}")
return mesh




def compute_pitch(mesh: trimesh.Trimesh, target_res: int, explicit_pitch: Optional[float]) -> float:
if explicit_pitch is not None and explicit_pitch > 0:
return float(explicit_pitch)
longest = float(max(mesh.extents))
if target_res <= 0 or longest <= 0:
raise ValueError("Non-positive target_res or mesh extents; cannot compute pitch.")
return longest / float(target_res)




def voxelize(mesh: trimesh.Trimesh,
pitch: float,
fill: bool = True) -> Tuple[trimesh.voxel.VoxelGrid, trimesh.voxel.VoxelGrid]:
vg_surface = mesh.voxelized(pitch=pitch)
vg_filled = vg_surface.fill() if fill else vg_surface
return vg_surface, vg_filled




def voxel_centers_world(vg: trimesh.voxel.VoxelGrid) -> np.ndarray:
idx = vg.sparse_indices # (N,3)
N = len(idx)
homo = np.hstack([idx, np.ones((N, 1), dtype=idx.dtype)]) # (N,4)
centers_world = (vg.transform @ homo.T).T[:, :3]
return centers_world




def propagate_face_colors(mesh: trimesh.Trimesh,
centers_world: np.ndarray) -> Optional[np.ndarray]:
has_fc = (hasattr(mesh, "visual") and getattr(mesh.visual, "face_colors", None) is not None)
)

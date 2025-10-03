# Voxelizer


Clean local implementation of a PLY → voxel pipeline using trimesh lib (surface & fill) with per-face color sgmentation to voxels.


## Quickstart


```bash
# 1) Create and activate a venv (recommended)
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate


# 2) Install deps requirments , add the ply files 
pip install -r requirements.txt


# 3) Run
python voxelize.py \
--input path/to/model.ply \
--outdir voxel_out \
--target-res 128 \ # more makes it hard to run on bad pc
--fill # remove this flag to export surface-only and you dont want volume

Dev notes

This pipeline does not require watertight meshes for surface voxels; filled voxels use trimesh's fill().

Per-face color propagation uses trimesh.proximity.ProximityQuery.on_surface (requires rtree). If colors are missing, colored outputs are skipped. 

The NPZ stores the index→world affine transform; to get voxel centers in world space, apply it to [i,j,k,1]^T.

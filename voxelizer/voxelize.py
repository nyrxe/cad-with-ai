import argparse
from src.voxelizer.voxelize import run


def parse_args():
p = argparse.ArgumentParser(description="PLY → voxel utilities")
p.add_argument("--input", required=True, help="Path to input .ply")
p.add_argument("--outdir", default="voxel_out", help="Output directory")
p.add_argument("--target-res", type=int, default=128, help="Grid resolution along longest axis")
p.add_argument("--pitch", type=float, default=None, help="Override voxel pitch (mesh units)")
p.add_argument("--fill", action="store_true", help="Fill interior voxels")
p.add_argument("--no-colors", action="store_true", help="Disable color propagation")
return p.parse_args()


if __name__ == "__main__":
args = parse_args()
run(
input_path=args.input,
outdir=args.outdir,
target_res=args.target_res,
pitch=args.pitch,
fill=args.fill,
do_colors=not args.no_colors,
)

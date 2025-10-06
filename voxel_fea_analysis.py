#!/usr/bin/env python3
"""
Voxel to FEA Analysis Pipeline
Converts voxelized PLY models to CalculiX FEA models and runs stress analysis
"""

import os
import numpy as np
import csv
import math
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

def load_voxel_data(npz_path):
    """Load voxel indices and colors from NPZ file"""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Voxel data not found: {npz_path}")
    
    data = np.load(npz_path)
    idx = data["indices"].astype(np.int64)   # (N,3)
    colors = data["colors"].astype(np.uint8)    # (N,4)
    pitch_arr = data["pitch"]
    pitch = float(pitch_arr.item() if pitch_arr.size == 1 else np.ravel(pitch_arr)[0])
    transform = data["transform"].astype(np.float64)  # (4,4)
    
    print(f"Loaded voxel data: {idx.shape[0]} voxels")
    print(f"Pitch: {pitch:.6f}")
    print(f"Transform shape: {transform.shape}")
    
    return idx, colors, pitch, transform

def generate_fea_mesh(idx, colors, pitch, transform):
    """Generate CalculiX mesh from voxel data"""
    print("\n=== GENERATING FEA MESH ===")
    
    # Calculate grid dimensions
    imin, jmin, kmin = idx.min(axis=0)
    imax, jmax, kmax = idx.max(axis=0)
    
    Nx = int(imax - imin + 1)  # voxel count (x direction)
    Ny = int(jmax - jmin + 1)  # voxel count (y direction)
    Nz = int(kmax - kmin + 1)  # voxel count (z direction)
    
    nx, ny, nz = Nx + 1, Ny + 1, Nz + 1   # node count per axis
    num_nodes = nx * ny * nz
    
    print(f"Grid (voxels): Nx={Nx}, Ny={Ny}, Nz={Nz}")
    print(f"Nodes per axis: nx={nx}, ny={ny}, nz={nz}")
    print(f"Total nodes: {num_nodes}")
    print("Index mins:", (imin, jmin, kmin), "maxs:", (imax, jmax, kmax))
    
    # Node ID function (1-based)
    def node_id(i, j, k):
        return 1 + i + nx * (j + ny * k)
    
    # Generate all node coordinates
    print("Generating node coordinates...")
    node_coords = np.zeros((num_nodes, 3), dtype=np.float64)
    
    for k in range(nz):
        for j in range(ny):
            base = 1 + nx * (j + ny * k)  # node_id(i=0, j, k)
            # Vectorized transformation for i-axis
            i_vals = np.arange(nx, dtype=np.float64)
            ones = np.ones_like(i_vals)
            # Homogeneous coordinates: [i, j, k, 1]
            H = np.stack([i_vals, ones * j, ones * k, ones], axis=1)  # (nx,4)
            P = (transform @ H.T).T[:, :3]  # (nx,3)
            node_coords[base-1 : base-1 + nx] = P
    
    print("Node coordinates generated")
    
    # Generate elements and ELSETs
    print("Generating elements and ELSETs...")
    
    def voxel_to_elem_nodes(ii, jj, kk):
        """Convert voxel (ii,jj,kk) to C3D8R nodes (CalculiX standard order)"""
        n1 = node_id(ii - imin,     jj - jmin,     kk - kmin)
        n2 = node_id(ii - imin + 1, jj - jmin,     kk - kmin)
        n3 = node_id(ii - imin + 1, jj - jmin + 1, kk - kmin)
        n4 = node_id(ii - imin,     jj - jmin + 1, kk - kmin)
        n5 = node_id(ii - imin,     jj - jmin,     kk - kmin + 1)
        n6 = node_id(ii - imin + 1, jj - jmin,     kk - kmin + 1)
        n7 = node_id(ii - imin + 1, jj - jmin + 1, kk - kmin + 1)
        n8 = node_id(ii - imin,     jj - jmin + 1, kk - kmin + 1)
        return (n1,n2,n3,n4,n5,n6,n7,n8)
    
    # Map colors to unique parts
    uniq_colors, inv = np.unique(colors, axis=0, return_inverse=True)
    num_parts = uniq_colors.shape[0]
    print(f"Unique color parts: {num_parts}")
    
    # Generate elements and ELSETs
    elements = []
    elsets = defaultdict(list)
    
    for e_id, (ijk, lbl) in enumerate(zip(idx, inv), start=1):
        ii, jj, kk = map(int, ijk)
        nodes = voxel_to_elem_nodes(ii, jj, kk)
        elements.append((e_id, nodes))
        elsets[int(lbl)].append(e_id)
    
    print(f"Total elements: {len(elements)}")
    
    return node_coords, elements, elsets, uniq_colors, imin, jmin, kmin

def write_calculix_input(node_coords, elements, elsets, uniq_colors, output_dir):
    """Write CalculiX input file"""
    print("\n=== WRITING CALCULIX INPUT ===")
    
    inp_path = os.path.join(output_dir, "model.inp")
    num_nodes = node_coords.shape[0]
    
    def write_wrapped_ids(f, ids, per_line=16):
        line = []
        for i, val in enumerate(ids, start=1):
            line.append(str(val))
            if i % per_line == 0:
                f.write(",".join(line) + "\n")
                line = []
        if line:
            f.write(",".join(line) + "\n")
    
    with open(inp_path, "w") as f:
        # Heading
        f.write("*HEADING\n")
        f.write("Voxel grid to C3D8R (SI units: m, Pa, kg)\n")
        
        # NODE
        f.write("*NODE\n")
        for nid in range(1, num_nodes+1):
            x, y, z = node_coords[nid-1]
            f.write(f"{nid}, {x:.9g}, {y:.9g}, {z:.9g}\n")
        
        # ELEMENT
        f.write("\n*ELEMENT, TYPE=C3D8R, ELSET=ALL\n")
        for e_id, nodes in elements:
            f.write(f"{e_id}, " + ",".join(str(n) for n in nodes) + "\n")
        
        # ELSET (one per color)
        for lbl, eids in elsets.items():
            rgba = uniq_colors[lbl]
            name = f"PART_{int(lbl):03d}_R{int(rgba[0])}G{int(rgba[1])}B{int(rgba[2])}A{int(rgba[3])}"
            f.write(f"\n*ELSET, ELSET={name}\n")
            write_wrapped_ids(f, eids, per_line=16)
        
        # Material & Section
        f.write("\n*MATERIAL, NAME=Mat1\n")
        f.write("*ELASTIC\n7.0e10, 0.33\n*DENSITY\n2700.\n")
        f.write("\n*SOLID SECTION, ELSET=ALL, MATERIAL=Mat1\n")
        
        # Boundary sets
        xmin = node_coords[:, 0].min()
        xmax = node_coords[:, 0].max()
        tol = max(1e-9, 1e-6 * max(abs(xmin), abs(xmax), 1.0))
        
        xmin_nodes = [i+1 for i,(x,y,z) in enumerate(node_coords) if abs(x - xmin) <= tol]
        xmax_elems = []
        for e_id, (n1,n2,n3,n4,n5,n6,n7,n8) in elements:
            if any(abs(node_coords[n-1][0] - xmax) <= tol for n in (n1,n2,n3,n4,n5,n6,n7,n8)):
                xmax_elems.append(e_id)
        
        f.write("\n*NSET, NSET=XMIN\n")
        write_wrapped_ids(f, xmin_nodes, per_line=16)
        
        # SURFACE on +X face
        f.write("\n*SURFACE, NAME=FACE_XMAX, TYPE=ELEMENT\n")
        for eid in xmax_elems:
            f.write(f"{eid}, S5\n")
        
        # STEP with PRINT outputs
        f.write("\n*STEP\n*STATIC\n")
        f.write("*BOUNDARY\nXMIN, 1, 3\n")
        f.write("\n*DLOAD\nFACE_XMAX, P, -1.0e6\n")
        f.write("\n*NODE PRINT, NSET=ALL\nU\n*EL PRINT, ELSET=ALL\nS, E\n*END STEP\n")
    
    print(f"CalculiX input written: {inp_path}")
    return inp_path

def run_calculix(input_path, output_dir):
    """Run CalculiX analysis"""
    print("\n=== RUNNING CALCULIX ===")
    
    # Check if CalculiX is available
    try:
        result = subprocess.run(['ccx', '--version'], capture_output=True, text=True)
        print("CalculiX found:", result.stdout.strip())
    except FileNotFoundError:
        print("CalculiX not found. Please install CalculiX:")
        print("  Windows: Download from https://www.calculix.de/")
        print("  Linux: sudo apt-get install calculix-ccx")
        print("  macOS: brew install calculix")
        return False
    
    # Run CalculiX
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    input_dir = os.path.dirname(input_path)
    
    try:
        result = subprocess.run(['ccx', '-i', base_name], 
                              cwd=input_dir, 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("CalculiX analysis completed successfully")
            return True
        else:
            print(f"CalculiX failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("CalculiX analysis timed out")
        return False
    except Exception as e:
        print(f"Error running CalculiX: {e}")
        return False

def parse_calculix_results(dat_path, elements, elsets, uniq_colors, node_coords, output_dir):
    """Parse CalculiX results and create CSV files"""
    print("\n=== PARSING CALCULIX RESULTS ===")
    
    if not os.path.exists(dat_path):
        print(f"Results file not found: {dat_path}")
        return
    
    # Helper functions
    def is_int(s):
        try: int(s); return True
        except: return False
    
    def is_float(s):
        try: float(s); return True
        except: return False
    
    def von_mises(Sxx, Syy, Szz, Sxy, Syz, Szx):
        return math.sqrt(0.5*((Sxx-Syy)**2 + (Syy-Szz)**2 + (Szz-Sxx)**2) + 3*(Sxy**2 + Syz**2 + Szx**2))
    
    # Read and parse results
    with open(dat_path, "r", errors="ignore") as f:
        lines = [ln.strip() for ln in f]
    
    # Parse NODE PRINT (U)
    node_rows = []
    in_node = False
    for ln in lines:
        s = ln.strip()
        if not s:
            if in_node:
                in_node = False
            continue
        
        if not in_node and s.startswith("U") and "N O D E" in s:
            in_node = True
            continue
        
        if in_node:
            parts = s.split()
            if len(parts) >= 4 and is_int(parts[0]) and all(is_float(p) for p in parts[1:4]):
                nid = int(parts[0])
                ux, uy, uz = map(float, parts[1:4])
                node_rows.append([nid, ux, uy, uz])
    
    # Parse EL PRINT (S, E)
    elem_rows = []
    in_elem = False
    for ln in lines:
        s = ln.strip()
        if not s:
            if in_elem:
                in_elem = False
            continue
        
        if not in_elem and s.startswith("S") and "E L E M E N T" in s:
            in_elem = True
            continue
        
        if in_elem:
            parts = s.split()
            # Handle both formats: eid Sxx... and eid IP Sxx...
            if len(parts) >= 7 and is_int(parts[0]):
                if all(is_float(p) for p in parts[1:7]):
                    # Format: eid Sxx Syy Szz Sxy Syz Szx
                    eid = int(parts[0])
                    Sxx, Syy, Szz, Sxy, Syz, Szx = map(float, parts[1:7])
                elif len(parts) >= 8 and is_int(parts[1]) and all(is_float(p) for p in parts[2:8]):
                    # Format: eid IP Sxx Syy Szz Sxy Syz Szx
                    eid = int(parts[0])
                    Sxx, Syy, Szz, Sxy, Syz, Szx = map(float, parts[2:8])
                else:
                    continue
                
                vm = von_mises(Sxx, Syy, Szz, Sxy, Syz, Szx)
                elem_rows.append([eid, Sxx, Syy, Szz, Sxy, Syz, Szx, vm])
    
    # Write CSV files
    csv_nodes = os.path.join(output_dir, "nodal_displacements.csv")
    csv_elems = os.path.join(output_dir, "element_stresses.csv")
    
    with open(csv_nodes, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["NodeID", "Ux", "Uy", "Uz"])
        w.writerows(node_rows)
    
    with open(csv_elems, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ElemID", "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx", "vonMises"])
        w.writerows(elem_rows)
    
    print(f"Results written:")
    print(f"  - {csv_nodes} ({len(node_rows)} nodes)")
    print(f"  - {csv_elems} ({len(elem_rows)} elements)")
    
    # Create detailed element analysis with parts
    create_detailed_analysis(elem_rows, elements, elsets, uniq_colors, node_coords, output_dir)

def create_detailed_analysis(elem_rows, elements, elsets, uniq_colors, node_coords, output_dir):
    """Create detailed analysis with part information"""
    print("\n=== CREATING DETAILED ANALYSIS ===")
    
    # Create element -> part mapping
    part_names = {}
    for lbl, eids in elsets.items():
        rgba = tuple(int(x) for x in uniq_colors[lbl])
        name = f"PART_{int(lbl):03d}_R{rgba[0]}G{rgba[1]}B{rgba[2]}A{rgba[3]}"
        for eid in eids:
            part_names[int(eid)] = name
    
    # Calculate element volumes
    def detJ_center(node_ids):
        X = np.array([node_coords[n-1] for n in node_ids], float)
        dN = np.array([
            [-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1],
            [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1],
        ], float) * 0.125
        J = dN.T @ X
        return float(np.linalg.det(J))
    
    elem_nodes = {e_id: nodes for (e_id, nodes) in elements}
    elem_vol = {}
    for e_id in range(1, len(elements)+1):
        detJ = detJ_center(elem_nodes[e_id])
        elem_vol[e_id] = 8.0 * abs(detJ)  # m^3
    
    density = 2700.0  # kg/m^3
    
    # Create detailed CSV
    csv_detailed = os.path.join(output_dir, "element_stresses_by_part.csv")
    with open(csv_detailed, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ElemID", "Part", "Volume_m3", "Mass_kg", "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx", "vonMises_Pa"])
        
        for (eid, Sxx, Syy, Szz, Sxy, Syz, Szx, vm) in elem_rows:
            part = part_names.get(eid, "UNLABELED")
            vol = elem_vol.get(eid, 0.0)
            mass = density * vol
            w.writerow([eid, part, vol, mass, Sxx, Syy, Szz, Sxy, Syz, Szx, vm])
    
    # Create part summary
    df = pd.read_csv(csv_detailed)
    
    def p95(x):
        return float(np.percentile(np.asarray(x, dtype=float), 95)) if len(x) else float("nan")
    
    g = df.groupby("Part", dropna=False)
    summary = pd.DataFrame({
        "Part": g.size().index,
        "ElemCount": g.size().values,
        "TotalVolume_m3": g["Volume_m3"].sum().values,
        "TotalMass_kg": g["Mass_kg"].sum().values,
        "vonMises_max_Pa": g["vonMises_Pa"].max().values,
        "vonMises_mean_Pa": g["vonMises_Pa"].mean().values,
        "vonMises_p95_Pa": g["vonMises_Pa"].apply(p95).values,
    })
    
    summary = summary.sort_values("Part").reset_index(drop=True)
    csv_summary = os.path.join(output_dir, "part_summary.csv")
    summary.to_csv(csv_summary, index=False)
    
    print(f"Detailed analysis written:")
    print(f"  - {csv_detailed}")
    print(f"  - {csv_summary}")
    
    # Display summary
    print("\nPart Summary:")
    print(summary.to_string(index=False))

def main():
    """Main FEA analysis pipeline"""
    print("=== VOXEL TO FEA ANALYSIS PIPELINE ===")
    
    # Find voxel data files
    voxel_out_dir = "voxel_out"
    if not os.path.exists(voxel_out_dir):
        print(f"Voxel output directory not found: {voxel_out_dir}")
        print("Please run voxelize_local.py first to generate voxel data.")
        return
    
    # Find all voxel data files
    voxel_dirs = [d for d in os.listdir(voxel_out_dir) 
                  if os.path.isdir(os.path.join(voxel_out_dir, d))]
    
    if not voxel_dirs:
        print("No voxel data directories found in voxel_out/")
        return
    
    print(f"Found {len(voxel_dirs)} voxel datasets: {voxel_dirs}")
    
    # Process each voxel dataset
    for voxel_dir in voxel_dirs:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {voxel_dir}")
        print(f"{'='*60}")
        
        voxel_path = os.path.join(voxel_out_dir, voxel_dir)
        npz_path = os.path.join(voxel_path, "voxels_filled_indices_colors.npz")
        
        if not os.path.exists(npz_path):
            print(f"Voxel data not found: {npz_path}")
            print("Skipping this dataset...")
            continue
        
        try:
            # Load voxel data
            idx, colors, pitch, transform = load_voxel_data(npz_path)
            
            # Generate FEA mesh
            node_coords, elements, elsets, uniq_colors, imin, jmin, kmin = generate_fea_mesh(
                idx, colors, pitch, transform)
            
            # Create FEA output directory
            fea_output_dir = os.path.join(voxel_path, "fea_analysis")
            os.makedirs(fea_output_dir, exist_ok=True)
            
            # Write CalculiX input
            inp_path = write_calculix_input(node_coords, elements, elsets, uniq_colors, fea_output_dir)
            
            # Run CalculiX
            if run_calculix(inp_path, fea_output_dir):
                # Parse results
                dat_path = os.path.join(fea_output_dir, "model.dat")
                parse_calculix_results(dat_path, elements, elsets, uniq_colors, node_coords, fea_output_dir)
                print(f"\n✅ FEA analysis completed for {voxel_dir}")
            else:
                print(f"\n❌ FEA analysis failed for {voxel_dir}")
                
        except Exception as e:
            print(f"\n❌ Error processing {voxel_dir}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("FEA ANALYSIS PIPELINE COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Complete Voxel to FEA Analysis Pipeline
Combines FEA mesh generation, CalculiX analysis, and part-based results
"""

import os
import numpy as np
import csv
import math
import subprocess
from collections import defaultdict

def von_mises(sxx, syy, szz, sxy, syz, sxz):
    """Calculate von Mises stress"""
    return math.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3*(sxy**2 + syz**2 + sxz**2))

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
        
        # SURFACE on +X face (auto-detect face number)
        f.write("\n*SURFACE, NAME=FACE_XMAX, TYPE=ELEMENT\n")
        for eid in xmax_elems:
            # Auto-detect face: for C3D8R, face 5 is typically +X
            # This could be improved with element face normal analysis
            f.write(f"{eid}, S5\n")
        
        # STEP with PRINT outputs
        f.write("\n*STEP\n*STATIC\n")
        f.write("*BOUNDARY\nXMIN, 1, 3\n")
        f.write("\n*DLOAD\nFACE_XMAX, P, -1.0e6\n")
        f.write("\n*NODE PRINT, NSET=ALL\nU\n*EL PRINT, ELSET=ALL\nS, E\n*END STEP\n")
    
    print(f"CalculiX input written: {inp_path}")
    return inp_path

def run_calculix(input_path, output_dir, num_nodes=None, num_elements=None):
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
    
    # Check model size and warn if large
    if num_nodes and num_elements:
        print(f"Model size: {num_nodes} nodes, {num_elements} elements")
        
        if num_nodes > 10000 or num_elements > 10000:
            print(f"⚠️  Large model detected: {num_nodes} nodes, {num_elements} elements")
            print("   This may take 10-30 minutes to solve...")
            
            # Ask user if they want to continue
            while True:
                response = input("   Continue with analysis? (y/n/skip): ").lower().strip()
                if response in ['y', 'yes']:
                    print("   Proceeding with analysis...")
                    break
                elif response in ['n', 'no', 'skip']:
                    print("   Skipping this model...")
                    return False
                else:
                    print("   Please enter 'y' for yes, 'n' for no, or 'skip'")
    
    # Run CalculiX
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    input_dir = os.path.dirname(input_path)
    
    try:
        result = subprocess.run(['ccx', base_name], 
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

def extract_stress_data(dat_path, frd_path=None):
    """Extract stress data from CalculiX results (DAT file with FRD fallback)"""
    stress_data = []
    
    # Try DAT file first (text format)
    if os.path.exists(dat_path):
        print(f"Reading stress data from: {dat_path}")
        stress_data = _parse_dat_stress(dat_path)
    
    # Fallback to FRD file if DAT parsing fails
    if not stress_data and frd_path and os.path.exists(frd_path):
        print(f"Falling back to FRD file: {frd_path}")
        stress_data = _parse_frd_stress(frd_path)
    
    if not stress_data:
        print("No stress data found in DAT or FRD files")
        return []
    
    print(f"Extracted {len(stress_data)} stress results")
    return stress_data

def _parse_dat_stress(dat_path):
    """Parse stress data from DAT file"""
    with open(dat_path, "r") as f:
        lines = f.readlines()
    
    # Find stress section
    stress_start = None
    for i, line in enumerate(lines):
        if "stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz)" in line:
            stress_start = i + 1
            break
    
    if stress_start is None:
        return []
    
    # Extract stress data
    stress_data = []
    for i in range(stress_start, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 8:
            try:
                elem_id = int(parts[0])
                sxx = float(parts[2])
                syy = float(parts[3]) 
                szz = float(parts[4])
                sxy = float(parts[5])
                sxz = float(parts[6])
                syz = float(parts[7])
                
                vm = von_mises(sxx, syy, szz, sxy, syz, sxz)
                stress_data.append([elem_id, sxx, syy, szz, sxy, syz, sxz, vm])
            except (ValueError, IndexError):
                continue
    
    return stress_data

def _parse_frd_stress(frd_path):
    """Parse stress data from FRD file (binary format)"""
    # Note: FRD parsing would require additional binary reading logic
    # For now, return empty list - this is a placeholder for future implementation
    print("FRD parsing not yet implemented - using DAT file only")
    return []

def analyze_parts(stress_data, colors, uniq_colors, output_dir):
    """Analyze stress by part/color"""
    print("\n=== PART-BASED STRESS ANALYSIS ===")
    
    if not stress_data:
        print("No stress data to analyze")
        return
    
    # Get unique colors and create mapping
    unique_colors, color_indices = np.unique(colors, axis=0, return_inverse=True)
    num_parts = len(unique_colors)
    print(f"Found {num_parts} unique parts (colors)")
    
    # Show parts
    for i, color in enumerate(unique_colors):
        part_name = f"PART_{i:03d}_R{color[0]}G{color[1]}B{color[2]}A{color[3]}"
        count = np.sum(color_indices == i)
        print(f"  {part_name}: {count} voxels")
    
    # Group elements by part
    part_stresses = defaultdict(list)
    for elem_data in stress_data:
        elem_id = elem_data[0]
        vm_stress = elem_data[7]  # von Mises stress
        
        # Find which part this element belongs to
        if elem_id <= len(color_indices):
            part_idx = color_indices[elem_id - 1]  # elem_id is 1-based
            part_stresses[part_idx].append(vm_stress)
    
    # Calculate part statistics
    part_summary = []
    for part_idx in range(num_parts):
        if part_idx in part_stresses:
            stresses = np.array(part_stresses[part_idx])
            color = unique_colors[part_idx]
            part_name = f"PART_{part_idx:03d}_R{color[0]}G{color[1]}B{color[2]}A{color[3]}"
            
            summary = {
                'Part': part_name,
                'Color_RGBA': f"({color[0]},{color[1]},{color[2]},{color[3]})",
                'ElementCount': len(stresses),
                'vonMises_max_Pa': np.max(stresses),
                'vonMises_mean_Pa': np.mean(stresses),
                'vonMises_min_Pa': np.min(stresses),
                'vonMises_p95_Pa': np.percentile(stresses, 95),
                'vonMises_std_Pa': np.std(stresses)
            }
            part_summary.append(summary)
            
            print(f"\n{part_name}:")
            print(f"  Elements: {len(stresses)}")
            print(f"  Max stress: {np.max(stresses):.2e} Pa ({np.max(stresses)/1e6:.1f} MPa)")
            print(f"  Mean stress: {np.mean(stresses):.2e} Pa ({np.mean(stresses)/1e6:.1f} MPa)")
            print(f"  95th percentile: {np.percentile(stresses, 95):.2e} Pa ({np.percentile(stresses, 95)/1e6:.1f} MPa)")
    
    # Save part summary CSV
    csv_path = os.path.join(output_dir, "part_stress_summary.csv")
    with open(csv_path, "w", newline="") as f:
        if part_summary:
            writer = csv.DictWriter(f, fieldnames=part_summary[0].keys())
            writer.writeheader()
            writer.writerows(part_summary)
    
    print(f"\nPart summary saved to: {csv_path}")
    
    # Save detailed part data
    detailed_path = os.path.join(output_dir, "part_detailed_results.csv")
    with open(detailed_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Part", "ElementID", "vonMises_Pa", "Sxx_Pa", "Syy_Pa", "Szz_Pa", "Sxy_Pa", "Syz_Pa", "Szx_Pa"])
        
        for elem_data in stress_data:
            elem_id = elem_data[0]
            if elem_id <= len(color_indices):
                part_idx = color_indices[elem_id - 1]
                color = unique_colors[part_idx]
                part_name = f"PART_{part_idx:03d}_R{color[0]}G{color[1]}B{color[2]}A{color[3]}"
                
                writer.writerow([part_name, elem_id] + elem_data[1:])
    
    print(f"Detailed results saved to: {detailed_path}")

def process_single_model(model_dir):
    """Process a single model through complete pipeline"""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {os.path.basename(model_dir)}")
    print(f"{'='*60}")
    
    npz_path = os.path.join(model_dir, "voxels_filled_indices_colors.npz")
    
    if not os.path.exists(npz_path):
        print(f"Voxel data not found: {npz_path}")
        print("Skipping this dataset...")
        return False
    
    try:
        # Load voxel data
        idx, colors, pitch, transform = load_voxel_data(npz_path)
        
        # Generate FEA mesh
        node_coords, elements, elsets, uniq_colors, imin, jmin, kmin = generate_fea_mesh(
            idx, colors, pitch, transform)
        
        # Create FEA output directory
        fea_output_dir = os.path.join(model_dir, "fea_analysis")
        os.makedirs(fea_output_dir, exist_ok=True)
        
        # Write CalculiX input
        inp_path = write_calculix_input(node_coords, elements, elsets, uniq_colors, fea_output_dir)
        
        # Run CalculiX
        if run_calculix(inp_path, fea_output_dir, len(node_coords), len(elements)):
            # Extract stress data (try DAT first, FRD as fallback)
            dat_path = os.path.join(fea_output_dir, "model.dat")
            frd_path = os.path.join(fea_output_dir, "model.frd")
            stress_data = extract_stress_data(dat_path, frd_path)
            
            if stress_data:
                # Analyze parts
                analyze_parts(stress_data, colors, uniq_colors, fea_output_dir)
                print(f"\n✅ Complete analysis finished for {os.path.basename(model_dir)}")
                return True
            else:
                print(f"\n❌ No stress data extracted for {os.path.basename(model_dir)}")
                return False
        else:
            print(f"\n❌ FEA analysis failed for {os.path.basename(model_dir)}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error processing {os.path.basename(model_dir)}: {e}")
        return False

def test_voxel_to_elem_nodes():
    """Unit test for voxel_to_elem_nodes orientation"""
    print("\n=== TESTING VOXEL TO ELEMENT NODE ORIENTATION ===")
    
    # Test parameters
    imin, jmin, kmin = 0, 0, 0
    nx, ny, nz = 3, 3, 3  # Small test grid
    
    def node_id(i, j, k):
        return 1 + i + nx * (j + ny * k)
    
    def voxel_to_elem_nodes(ii, jj, kk):
        """Test version of voxel_to_elem_nodes"""
        n1 = node_id(ii - imin,     jj - jmin,     kk - kmin)
        n2 = node_id(ii - imin + 1, jj - jmin,     kk - kmin)
        n3 = node_id(ii - imin + 1, jj - jmin + 1, kk - kmin)
        n4 = node_id(ii - imin,     jj - jmin + 1, kk - kmin)
        n5 = node_id(ii - imin,     jj - jmin,     kk - kmin + 1)
        n6 = node_id(ii - imin + 1, jj - jmin,     kk - kmin + 1)
        n7 = node_id(ii - imin + 1, jj - jmin + 1, kk - kmin + 1)
        n8 = node_id(ii - imin,     jj - jmin + 1, kk - kmin + 1)
        return (n1,n2,n3,n4,n5,n6,n7,n8)
    
    # Test voxel (1,1,1) should give nodes 1,2,3,4,5,6,7,8
    test_nodes = voxel_to_elem_nodes(1, 1, 1)
    expected_nodes = (1, 2, 3, 4, 5, 6, 7, 8)
    
    if test_nodes == expected_nodes:
        print("✅ Voxel to element node mapping test PASSED")
        print(f"   Voxel (1,1,1) → Nodes {test_nodes}")
        return True
    else:
        print("❌ Voxel to element node mapping test FAILED")
        print(f"   Expected: {expected_nodes}")
        print(f"   Got: {test_nodes}")
        return False

def main():
    """Main complete FEA analysis pipeline"""
    print("=== COMPLETE VOXEL TO FEA ANALYSIS PIPELINE ===")
    
    # Run unit test first
    test_voxel_to_elem_nodes()
    
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
    successful = 0
    for voxel_dir in voxel_dirs:
        voxel_path = os.path.join(voxel_out_dir, voxel_dir)
        if process_single_model(voxel_path):
            successful += 1
    
    print(f"\n{'='*60}")
    print(f"COMPLETE FEA ANALYSIS PIPELINE FINISHED")
    print(f"Successfully processed: {successful}/{len(voxel_dirs)} models")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

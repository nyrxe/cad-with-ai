#!/usr/bin/env python3
"""
Test FEA Analysis Safely
Runs FEA on a single model with comprehensive error handling
Use this to diagnose FEA issues without crashing the frontend
"""

import os
import sys
import subprocess
import traceback

def test_calculix():
    """Test if CalculiX is installed"""
    print("=== TESTING CALCULIX INSTALLATION ===")
    
    calculix_found = False
    calculix_cmd = None
    
    # Try multiple possible command names
    for cmd in ['ccx', 'calculix', 'ccx.exe']:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0 or 'CalculiX' in result.stdout or 'CalculiX' in result.stderr:
                calculix_found = True
                calculix_cmd = cmd
                print(f"✅ CalculiX found: {cmd}")
                print(f"   Version info: {result.stdout.strip() or result.stderr.strip()}")
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    if not calculix_found:
        print("❌ CalculiX not found!")
        print("   Please install CalculiX:")
        print("   Windows: Download from https://www.calculix.de/")
        print("   Or place ccx.exe in project directory")
        return None
    
    return calculix_cmd

def check_model_size(model_dir):
    """Check voxel model size"""
    print(f"\n=== CHECKING MODEL SIZE ===")
    
    npz_path = os.path.join(model_dir, "voxels_filled_indices_colors.npz")
    if not os.path.exists(npz_path):
        print(f"❌ Voxel data not found: {npz_path}")
        return None
    
    try:
        import numpy as np
        data = np.load(npz_path)
        indices = data['indices']
        pitch_arr = data['pitch']
        pitch = float(pitch_arr.item() if pitch_arr.size == 1 else pitch_arr.ravel()[0])
        
        num_voxels = len(indices)
        # Estimate nodes (voxels + 1 per dimension)
        imin, jmin, kmin = indices.min(axis=0)
        imax, jmax, kmax = indices.max(axis=0)
        nx = imax - imin + 2
        ny = jmax - jmin + 2
        nz = kmax - kmin + 2
        estimated_nodes = nx * ny * nz
        estimated_elements = num_voxels
        
        print(f"✅ Model loaded:")
        print(f"   Voxels: {num_voxels:,}")
        print(f"   Estimated nodes: {estimated_nodes:,}")
        print(f"   Estimated elements: {estimated_elements:,}")
        print(f"   Pitch: {pitch:.6f} m ({pitch*1000:.3f} mm)")
        
        # Estimate memory
        memory_mb = (estimated_nodes * 8 * 3 + estimated_elements * 8 * 8) / (1024 * 1024)
        print(f"   Estimated RAM needed: {memory_mb:.0f} MB")
        
        # Estimate solve time
        if estimated_elements < 10000:
            solve_time = "2-5 minutes"
        elif estimated_elements < 50000:
            solve_time = "5-15 minutes"
        elif estimated_elements < 100000:
            solve_time = "15-30 minutes"
        else:
            solve_time = "30+ minutes (may timeout)"
        
        print(f"   Estimated solve time: {solve_time}")
        
        # Warning if too large
        if estimated_elements > 100000:
            print(f"\n⚠️  WARNING: Very large model!")
            print(f"   Consider reducing voxel resolution")
            print(f"   Run: python voxelize_local.py 16")
        
        return {
            'voxels': num_voxels,
            'nodes': estimated_nodes,
            'elements': estimated_elements,
            'memory_mb': memory_mb
        }
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_fea_on_model(model_dir, calculix_cmd):
    """Test FEA on a single model"""
    print(f"\n=== TESTING FEA ON MODEL ===")
    print(f"Model: {os.path.basename(model_dir)}")
    
    try:
        # Import FEA functions
        sys.path.insert(0, os.path.dirname(__file__))
        from voxel_to_fea_complete import process_single_model
        
        # Run FEA with error handling
        print("Starting FEA analysis...")
        success = process_single_model(model_dir, model_type="original")
        
        if success:
            print("✅ FEA analysis completed successfully!")
            return True
        else:
            print("❌ FEA analysis failed (check output above)")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return False
    except MemoryError as e:
        print(f"\n❌ Out of memory: {e}")
        print("   Try reducing voxel resolution")
        return False
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("FEA SAFETY TEST")
    print("=" * 60)
    
    # Test CalculiX
    calculix_cmd = test_calculix()
    if not calculix_cmd:
        print("\n❌ Cannot proceed without CalculiX")
        return
    
    # Find models
    voxel_out_dir = "voxel_out"
    if not os.path.exists(voxel_out_dir):
        print(f"\n❌ Voxel output directory not found: {voxel_out_dir}")
        print("   Run voxelize_local.py first")
        return
    
    # List available models
    model_dirs = [d for d in os.listdir(voxel_out_dir) 
                  if os.path.isdir(os.path.join(voxel_out_dir, d))]
    
    if not model_dirs:
        print(f"\n❌ No models found in {voxel_out_dir}")
        return
    
    print(f"\n=== AVAILABLE MODELS ===")
    for i, model_dir in enumerate(model_dirs, 1):
        print(f"  {i}. {model_dir}")
    
    # Test first model (or ask user)
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        # Use first model
        model_name = model_dirs[0]
        print(f"\nTesting first model: {model_name}")
        print("(To test specific model: python test_fea_safely.py <model_name>)")
    
    model_path = os.path.join(voxel_out_dir, model_name)
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        return
    
    # Check model size
    model_info = check_model_size(model_path)
    if not model_info:
        return
    
    # Ask to continue if large
    if model_info['elements'] > 50000:
        print(f"\n⚠️  Large model detected!")
        response = input("Continue with FEA test? (y/n): ").lower().strip()
        if response != 'y':
            print("Test cancelled")
            return
    
    # Test FEA
    print(f"\n{'='*60}")
    success = test_fea_on_model(model_path, calculix_cmd)
    
    if success:
        print(f"\n{'='*60}")
        print("✅ TEST PASSED: FEA works correctly!")
        print("   You can now use the frontend safely")
    else:
        print(f"\n{'='*60}")
        print("❌ TEST FAILED: FEA has issues")
        print("   Check error messages above")
        print("   Fix issues before using frontend")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        traceback.print_exc()


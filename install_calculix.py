#!/usr/bin/env python3
"""
CalculiX Installation Helper
Provides instructions and checks for CalculiX installation on different platforms
"""

import os
import sys
import subprocess
import platform

def check_calculix():
    """Check if CalculiX is installed and working"""
    try:
        # Test with a complete finite element model
        test_input = """*HEADING
CalculiX Test Model
*NODE
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT,TYPE=C3D8R,ELSET=ALL
1,1,2,3,4,5,6,7,8
*MATERIAL,NAME=Mat1
*ELASTIC
7.0e10,0.33
*DENSITY
2700.
*SOLID SECTION,ELSET=ALL,MATERIAL=Mat1
*NSET,NSET=FIXED
1,4,5,8
*BOUNDARY
FIXED,1,3
*STEP
*STATIC
*DLOAD
ALL,P,1000
*END STEP"""
        
        # Create a temporary test file
        with open("test_calculix.inp", "w") as f:
            f.write(test_input)
        
        # Run CalculiX on the test file
        result = subprocess.run(['ccx', 'test_calculix'], capture_output=True, text=True, timeout=30)
        
        # Clean up test files
        for ext in ['.inp', '.dat', '.frd', '.sta', '.cvg']:
            try:
                os.remove(f"test_calculix{ext}")
            except FileNotFoundError:
                pass
        
        if result.returncode == 0:
            print("✅ CalculiX is installed and working")
            # Extract version from stderr (CalculiX prints version info to stderr)
            if "CalculiX Version" in result.stderr:
                version_line = [line for line in result.stderr.split('\n') if 'CalculiX Version' in line]
                if version_line:
                    print(f"Version info: {version_line[0].strip()}")
            return True
        else:
            print("❌ CalculiX found but not working properly")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ CalculiX not found")
        return False
    except subprocess.TimeoutExpired:
        print("❌ CalculiX test timed out")
        return False

def get_installation_instructions():
    """Get platform-specific installation instructions"""
    system = platform.system().lower()
    
    if system == "windows":
        return """
Windows Installation:
1. Download CalculiX from: https://www.calculix.de/
2. Extract to a folder (e.g., C:\\CalculiX)
3. Add the bin folder to your PATH environment variable
4. Or place ccx.exe in your project directory
        """
    elif system == "linux":
        return """
Linux Installation:
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install calculix-ccx

# Or build from source:
# Download from https://www.calculix.de/
# Follow build instructions in the documentation
        """
    elif system == "darwin":  # macOS
        return """
macOS Installation:
# Using Homebrew:
brew install calculix

# Or download pre-built binaries from:
# https://www.calculix.de/
        """
    else:
        return """
Unknown platform. Please visit https://www.calculix.de/ for installation instructions.
        """

def main():
    print("=== CalculiX Installation Checker ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    if check_calculix():
        print("\n🎉 CalculiX is ready to use!")
        print("You can now run: python voxel_fea_analysis.py")
    else:
        print("\n📋 Installation Instructions:")
        print(get_installation_instructions())
        print("\nAfter installation, run this script again to verify.")

if __name__ == "__main__":
    main()

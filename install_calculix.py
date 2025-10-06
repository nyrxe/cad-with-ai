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
        result = subprocess.run(['ccx', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CalculiX is installed and working")
            print(f"Version info: {result.stdout.strip()}")
            return True
        else:
            print("❌ CalculiX found but not working properly")
            return False
    except FileNotFoundError:
        print("❌ CalculiX not found")
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

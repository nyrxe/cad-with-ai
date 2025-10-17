#!/usr/bin/env python3
"""
Simple PLY Processor
Drag & drop PLY file, run your existing pipeline, show results
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import shutil
import subprocess
import threading
import time
import sys

class SimplePLYProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("CAD with AI - Simple PLY Processor")
        self.root.geometry("600x500")
        
        self.selected_file = None
        self.python_exe = None
        self.setup_ui()
        self.check_environment()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Configure window
        self.root.configure(bg="#f8f9fa")
        self.root.geometry("950x850")
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Segoe UI', 20, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Segoe UI', 12), foreground='#7f8c8d')
        style.configure('Success.TLabel', font=('Segoe UI', 11, 'bold'), foreground='#27ae60')
        style.configure('Info.TLabel', font=('Segoe UI', 10), foreground='#3498db')
        style.configure('Primary.TButton', font=('Segoe UI', 12, 'bold'))
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="30")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header section
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Title with emoji
        title = ttk.Label(header_frame, text="🤖 CAD with AI", style='Title.TLabel')
        title.pack()
        
        subtitle = ttk.Label(header_frame, text="Intelligent Material Optimization", style='Subtitle.TLabel')
        subtitle.pack(pady=(5, 0))
        
        # File drop area with modern styling
        self.drop_area = tk.Frame(main_frame, bg="#ffffff", height=130, relief="solid", bd=2)
        self.drop_area.pack(fill=tk.X, pady=(0, 25))
        
        # Inner drop area with solid border
        inner_drop = tk.Frame(self.drop_area, bg="#f8f9ff", relief="solid", bd=2, height=110)
        inner_drop.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Drop icon
        drop_icon = tk.Label(inner_drop, text="📁", font=("Segoe UI", 28), bg="#f8f9ff", fg="#3498db")
        drop_icon.pack(pady=(15, 5))
        
        drop_label = tk.Label(inner_drop, text="Drag & Drop PLY file here", 
                            bg="#f8f9ff", font=("Segoe UI", 14, "bold"), fg="#2c3e50")
        drop_label.pack()
        
        drop_subtitle = tk.Label(inner_drop, text="or click to browse • Supports .ply files", 
                                bg="#f8f9ff", font=("Segoe UI", 10), fg="#7f8c8d")
        drop_subtitle.pack()
        
        # Bind click to browse
        self.drop_area.bind("<Button-1>", self.browse_file)
        inner_drop.bind("<Button-1>", self.browse_file)
        drop_icon.bind("<Button-1>", self.browse_file)
        drop_label.bind("<Button-1>", self.browse_file)
        drop_subtitle.bind("<Button-1>", self.browse_file)
        
        # File info with better styling
        self.file_var = tk.StringVar(value="No file selected")
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        file_label = ttk.Label(file_frame, text="Selected File:", style='Info.TLabel')
        file_label.pack(side=tk.LEFT)
        
        self.file_display = ttk.Label(file_frame, textvariable=self.file_var, 
                                     font=("Segoe UI", 10, "bold"), foreground="#27ae60")
        self.file_display.pack(side=tk.LEFT, padx=(10, 0))
        
        # Process button with modern styling
        self.process_btn = ttk.Button(main_frame, text="🚀 Run AI Pipeline", 
                                    command=self.run_pipeline, state="disabled", style='Primary.TButton')
        self.process_btn.pack(pady=(0, 25))
        
        # Progress area with better styling
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 20))
        
        progress_label = ttk.Label(progress_frame, text="Processing Status:", style='Info.TLabel')
        progress_label.pack(anchor=tk.W)
        
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(5, 0))
        
        # Log area with modern styling
        log_frame = ttk.LabelFrame(main_frame, text="📊 Processing Log", padding="15")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, width=85, wrap=tk.WORD,
                                                 font=("Consolas", 9), bg="#f8f9fa", 
                                                 fg="#2c3e50", relief="flat", bd=0)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Results area with success styling
        self.results_frame = ttk.LabelFrame(main_frame, text="🎯 AI Recommendations", padding="15")
        self.results_frame.pack(fill=tk.X)
        
        self.results_text = tk.Text(self.results_frame, height=5, width=85, wrap=tk.WORD, 
                                  font=("Segoe UI", 10), bg="#f0f8ff", fg="#2c3e50", 
                                  state="disabled", relief="flat", bd=0)
        self.results_text.pack(fill=tk.X)
    
    def check_environment(self):
        """Check Python environment and dependencies"""
        try:
            # Set up virtual environment path
            venv_python = os.path.join(".venv", "Scripts", "python.exe")
            if os.path.exists(venv_python):
                self.python_exe = venv_python
            else:
                self.python_exe = "python"
            
            # Run diagnostics
            self.log_message("Checking Python environment...")
            result = subprocess.run([
                self.python_exe, "-c", 
                "import sys; import numpy; import os; "
                "print(f'Python: {sys.executable}'); "
                "print(f'NumPy version: {numpy.__version__}'); "
                "print(f'NumPy path: {numpy.__file__}'); "
                "print(f'Working dir: {os.getcwd()}'); "
                "print(f'Python path: {sys.path[:3]}')"
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                self.log_message("Environment diagnostics:")
                for line in result.stdout.strip().split('\n'):
                    self.log_message(f"  {line}")
                
                # Check if using virtual environment
                if ".venv" in result.stdout and "site-packages" in result.stdout:
                    self.log_message("OK Virtual environment detected")
                else:
                    self.log_message("WARNING: Not using virtual environment or NumPy not from venv")
                    self.log_message("WARNING: This may cause numpy import errors")
                    
            else:
                self.log_message(f"ERROR: Environment check failed: {result.stderr}")
                self.log_message("ERROR: Cannot proceed - Python environment issues")
                self.process_btn.config(state="disabled")
                
        except Exception as e:
            self.log_message(f"ERROR: Environment check error: {str(e)}")
            self.process_btn.config(state="disabled")
    
    def browse_file(self, event=None):
        """Browse for PLY file"""
        file_path = filedialog.askopenfilename(
            title="Select PLY File",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
        )
        if file_path:
            self.select_file(file_path)
    
    def select_file(self, file_path):
        """Select a file"""
        self.selected_file = file_path
        filename = os.path.basename(file_path)
        self.file_var.set(f"Selected: {filename}")
        self.process_btn.config(state="normal")
        self.log_message(f"Selected: {filename}")
        
        # Update drop area
        self.drop_area.config(bg="#e8f5e8")
        for widget in self.drop_area.winfo_children():
            try:
                widget.config(bg="#e8f5e8", fg="#2d5a2d")
            except:
                # Some widgets don't support fg option
                widget.config(bg="#e8f5e8")
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def run_pipeline(self):
        """Run the complete AI pipeline"""
        if not self.selected_file:
            messagebox.showerror("Error", "Please select a PLY file first")
            return
        
        # Disable button and start progress
        self.process_btn.config(state="disabled")
        self.progress.start()
        self.log_text.delete(1.0, tk.END)
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state="disabled")
        
        # Run in separate thread
        thread = threading.Thread(target=self._run_pipeline_thread)
        thread.daemon = True
        thread.start()
    
    def _run_pipeline_thread(self):
        """Run pipeline in separate thread"""
        try:
            self.log_message("Starting AI Pipeline...")
            
            # Step 1: Copy file to input directory
            self.log_message("Copying PLY file to input directory...")
            if not self.copy_to_input():
                return
            
            # Step 2: Voxelization
            self.log_message("Voxelizing...")
            if not self.run_script("voxelize_local.py"):
                return
            
            # Step 3: Thinning
            self.log_message("Creating thinned version...")
            if not self.run_script("voxel_thinning_advanced.py"):
                return
            
            # Step 4: FEA Analysis
            self.log_message("Running FEA analysis...")
            if not self.run_script("voxel_to_fea_complete.py"):
                return
            
            # Step 5: Combine Results
            self.log_message("Combining FEA results...")
            if not self.run_script("combine_results.py"):
                return
            
            # Step 6: Mass/Volume
            self.log_message("Adding mass/volume calculations...")
            if not self.run_script("add_mass_volume.py"):
                return
            
            # Step 7: Pairwise Data
            self.log_message("Building pairwise data...")
            if not self.run_script("build_pairwise.py"):
                return
            
            # Step 8: Update Original Parts
            self.log_message("Updating original parts data...")
            if not self.run_script("create_original_parts.py"):
                return
            
            # Step 9: AI Recommendations
            self.log_message("Running AI predictions...")
            if not self.run_script("recommend_thinning_final.py"):
                return
            
            # Step 10: Show Results
            self.log_message("Pipeline completed successfully!")
            self.show_results()
            
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.show_error_results()
        finally:
            # Cleanup
            self.progress.stop()
            self.process_btn.config(state="normal")
    
    def copy_to_input(self):
        """Copy PLY file to input directory"""
        try:
            # Create input directory if it doesn't exist
            os.makedirs("input", exist_ok=True)
            
            # Copy file
            filename = os.path.basename(self.selected_file)
            dest_path = os.path.join("input", filename)
            shutil.copy2(self.selected_file, dest_path)
            
            self.log_message(f"Copied {filename} to input directory")
            return True
            
        except Exception as e:
            self.log_message(f"Failed to copy file: {str(e)}")
            return False
    
    def run_script(self, script_name):
        """Run a Python script using virtual environment"""
        try:
            # Set environment variables for Unicode support
            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"
            env["PYTHONIOENCODING"] = "UTF-8"
            env["PYTHONUTF8"] = "1"
            
            # Use virtual environment Python
            result = subprocess.run([
                self.python_exe, script_name
            ], capture_output=True, text=True, cwd=os.getcwd(), env=env, encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                self.log_message(f"{script_name} completed")
                return True
            else:
                # Check if error is due to Unicode encoding
                if "UnicodeEncodeError" in result.stderr or "charmap" in result.stderr:
                    self.log_message(f"{script_name} failed: Output contained characters not supported by the active code page; replaced with ASCII.")
                else:
                    self.log_message(f"{script_name} failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log_message(f"Error running {script_name}: {str(e)}")
            return False
    
    def show_results(self):
        """Show AI recommendations"""
        try:
            # Try to load actual results using subprocess to avoid numpy issues
            results_file = "thinning_recommendations_final.csv"
            if os.path.exists(results_file):
                # Use subprocess to load results with correct environment
                result = self.load_results_with_subprocess(results_file)
                if result:
                    self.display_results(result)
                    return
                else:
                    self.log_message("No AI recommendations found in results file")
            else:
                self.log_message("AI recommendations file not found - pipeline may not have completed successfully")
            
            # Fallback to example results
            self.show_example_results()
                
        except Exception as e:
            self.log_message(f"Could not load results: {str(e)}")
            self.show_example_results()
    
    def load_results_with_subprocess(self, results_file):
        """Load results using subprocess to avoid numpy issues"""
        try:
            # Create a simple Python script to extract results
            model_name = os.path.basename(self.selected_file).replace('.ply', '')
            
            # Write script to temp file
            script_path = "temp_results_loader.py"
            with open(script_path, 'w') as f:
                f.write(f'''import pandas as pd
import sys
import os

try:
    df = pd.read_csv("{results_file}")
    model_name = "{model_name}"
    model_results = df[df['model_id'] == model_name]
    
    if not model_results.empty:
        # Return ALL parts for this model
        for _, row in model_results.iterrows():
            part_id = row.get('part_id', 'Part')
            reduction = row.get('recommended_reduction_pct', 0)
            stress_change = row.get('predicted_delta_p95_pct', 0)
            mass_saving = row.get('expected_mass_saving_kg', 0)
            
            print(f"PART|{{part_id}}|{{reduction}}|{{stress_change}}|{{mass_saving}}")
        print("END")
    else:
        print("NO_DATA")
except Exception as e:
    print(f"ERROR|{{str(e)}}")
''')
            
            # Run with virtual environment
            env = os.environ.copy()
            env['PYTHONNOUSERSITE'] = '1'
            env['PYTHONIOENCODING'] = 'UTF-8'
            env['PYTHONUTF8'] = '1'
            
            result = subprocess.run([
                '.venv\\Scripts\\python.exe', script_path
            ], capture_output=True, text=True, cwd=os.getcwd(), env=env)
            
            # Clean up temp file
            if os.path.exists(script_path):
                os.remove(script_path)
            
            if result.returncode == 0 and result.stdout.strip():
                output_lines = result.stdout.strip().split('\n')
                if output_lines[0] == "NO_DATA":
                    return None
                elif output_lines[0].startswith("ERROR|"):
                    self.log_message(f"Results loading failed: {output_lines[0]}")
                    return None
                else:
                    # Parse multiple parts
                    all_parts = []
                    for line in output_lines:
                        if line.startswith("PART|"):
                            parts = line.split("|")
                            all_parts.append({
                                'part_id': parts[1],
                                'reduction': float(parts[2]),
                                'stress_change': float(parts[3]),
                                'mass_saving': float(parts[4])
                            })
                    return all_parts if all_parts else None
            else:
                self.log_message(f"Results loading failed: {result.stderr}")
                return None
                
        except Exception as e:
            self.log_message(f"Error loading results: {str(e)}")
            return None
    
    def display_results(self, results):
        """Display the loaded results"""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        
        if isinstance(results, list):
            # Multiple results
            total_mass_saving = 0
            for i, result in enumerate(results):
                part_id = result['part_id']
                reduction = result['reduction']
                stress_change = result['stress_change']
                mass_saving = result['mass_saving']
                total_mass_saving += mass_saving
                
                if reduction > 0:
                    self.results_text.insert(tk.END, f"Part {part_id}: {reduction:.1f}% reduction\n")
                    self.results_text.insert(tk.END, f"  Stress change: {stress_change:.1f}%, Mass saving: {mass_saving:.3f} kg\n\n")
                else:
                    self.results_text.insert(tk.END, f"Part {part_id}: No thinning recommended\n\n")
            
            self.results_text.insert(tk.END, f"TOTAL MASS SAVING: {total_mass_saving:.3f} kg")
            self.log_message(f"AI Recommendations: {len(results)} parts analyzed, {total_mass_saving:.3f} kg total savings")
        else:
            # Single result
            part_id = results['part_id']
            reduction = results['reduction']
            stress_change = results['stress_change']
            mass_saving = results['mass_saving']
            
            if reduction > 0:
                recommendation = f"Part {part_id} can safely be reduced by {reduction:.1f}%"
                details = f"Predicted stress change: {stress_change:.1f}%\nExpected mass saving: {mass_saving:.3f} kg"
            else:
                recommendation = f"Part {part_id} - No thinning recommended"
                details = "Safety risk detected"
            
            self.results_text.insert(tk.END, f"{recommendation}\n")
            self.results_text.insert(tk.END, f"Details: {details}")
            self.log_message(f"AI Recommendation: {recommendation}")
        
        self.results_text.config(state="disabled")

    def show_example_results(self):
        """Show example results"""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "⚠️ No AI recommendations available\n")
        self.results_text.insert(tk.END, "The AI pipeline may not have completed successfully.\n")
        self.results_text.insert(tk.END, "Please check the log for errors and try again.")
        self.results_text.config(state="disabled")
        
        self.log_message("⚠️ No AI recommendations available - check pipeline completion")
    
    def show_error_results(self):
        """Show error results"""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Pipeline failed to complete\n")
        self.results_text.insert(tk.END, "Please check the log for details")
        self.results_text.config(state="disabled")

def main():
    """Main application"""
    root = tk.Tk()
    app = SimplePLYProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()

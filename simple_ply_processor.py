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
        self.root.geometry("1200x800")
        
        # Create main scrollable frame
        main_canvas = tk.Canvas(self.root, bg="#f8f9fa")
        main_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=main_scrollbar.set)
        
        # Pack scrollable area
        main_canvas.pack(side="left", fill="both", expand=True)
        main_scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel scrolling
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Segoe UI', 20, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Segoe UI', 12), foreground='#7f8c8d')
        style.configure('Success.TLabel', font=('Segoe UI', 11, 'bold'), foreground='#27ae60')
        style.configure('Info.TLabel', font=('Segoe UI', 10), foreground='#3498db')
        style.configure('Primary.TButton', font=('Segoe UI', 12, 'bold'))
        style.configure('Accent.TButton', font=('Segoe UI', 11, 'bold'))
        style.configure('Part.TCheckbutton', font=('Segoe UI', 11, 'bold'))
        
        # Main frame
        main_frame = ttk.Frame(scrollable_frame, padding="30")
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
        
        # AI Thinning Options
        thinning_frame = ttk.LabelFrame(main_frame, text="🔧 AI Thinning Options", padding="15")
        thinning_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.apply_thinning_var = tk.BooleanVar(value=False)  # Default OFF - not needed
        self.apply_thinning_cb = ttk.Checkbutton(thinning_frame, 
                                               text="Apply AI thinning to voxels and export modified geometry",
                                               variable=self.apply_thinning_var, 
                                               style='Part.TCheckbutton')
        self.apply_thinning_cb.pack(anchor=tk.W, pady=(0, 5))
        
        # Surface offset thinning option (COMMENTED OUT - not needed)
        # self.surface_offset_var = tk.BooleanVar(value=False)  # Default OFF
        # self.surface_offset_cb = ttk.Checkbutton(thinning_frame, 
        #                                        text="Slice surface to thin (offset shell) - produces watertight surface",
        #                                        variable=self.surface_offset_var, 
        #                                        style='Part.TCheckbutton')
        # self.surface_offset_cb.pack(anchor=tk.W, pady=(0, 5))
        
        # Voxel SDF thinning option
        self.voxel_thinning_var = tk.BooleanVar(value=True)  # Default ON
        self.voxel_thinning_cb = ttk.Checkbutton(thinning_frame, 
                                               text="Apply AI thinning to voxels (SDF offset) - removes voxel layers",
                                               variable=self.voxel_thinning_var, 
                                               style='Part.TCheckbutton')
        self.voxel_thinning_cb.pack(anchor=tk.W, pady=(0, 10))
        
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
        
        # Log area with modern styling and larger size
        log_frame = ttk.LabelFrame(main_frame, text="📊 Processing Log", padding="15")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=100, wrap=tk.WORD,
                                                 font=("Consolas", 10), bg="#f8f9fa", 
                                                 fg="#2c3e50", relief="flat", bd=0)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Part selection area
        select_frame = ttk.LabelFrame(main_frame, text="🧩 Select Parts for AI Recommendation", padding="15")
        select_frame.pack(fill=tk.X, pady=(0, 15))
        # Scrollable checkbox list
        self.part_vars = {}  # part_id -> tk.BooleanVar
        part_canvas = tk.Canvas(select_frame, highlightthickness=0, height=180, bg="#ffffff")
        part_scroll = ttk.Scrollbar(select_frame, orient='vertical', command=part_canvas.yview)
        part_canvas.configure(yscrollcommand=part_scroll.set)
        self.part_container = ttk.Frame(part_canvas)
        part_canvas.create_window((0, 0), window=self.part_container, anchor='nw')
        def _pcfg(_):
            part_canvas.configure(scrollregion=part_canvas.bbox('all'))
            part_canvas.itemconfig(1, width=part_canvas.winfo_width())
        self.part_container.bind('<Configure>', _pcfg)
        part_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        part_scroll.pack(side=tk.LEFT, fill=tk.Y, padx=(6,0))
        self.run_selected_btn = ttk.Button(select_frame, text="Run AI on Selected Parts", style='Accent.TButton', command=self.run_selected_recommendations, state='disabled')
        self.run_selected_btn.pack(side=tk.LEFT, padx=10)
        
        # Results area with success styling and scrollbars
        self.results_frame = ttk.LabelFrame(main_frame, text="🎯 AI Recommendations", padding="15")
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for the results text with scrollbar
        results_container = tk.Frame(self.results_frame)
        results_container.pack(fill=tk.BOTH, expand=True)
        
        # Results text with scrollbar - MUCH BIGGER
        self.results_text = scrolledtext.ScrolledText(results_container, height=20, width=120, wrap=tk.WORD, 
                                  font=("Segoe UI", 12), bg="#f0f8ff", fg="#2c3e50", 
                                  state="disabled", relief="flat", bd=0)
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
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
            self.log_message("Starting setup (voxelize to discover parts)...")
            
            # Step 1: Copy file to input directory
            self.log_message("Copying PLY file to input directory...")
            if not self.copy_to_input():
                return
            
            # Step 2: Voxelization (fast)
            self.log_message("Voxelizing (discovering parts)...")
            if not self.run_script("voxelize_local.py"):
                return

            # Populate part selector from voxels and stop here for user to choose
            self.populate_parts_from_voxels()
            self.log_message("Parts ready: Select and click 'Run AI on Selected Parts' to continue.")
            
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.show_error_results()
        finally:
            # Cleanup
            self.progress.stop()
            self.process_btn.config(state="normal")

    def populate_part_selector(self):
        """(Deprecated) Load parts from original_parts.csv. Kept for compatibility."""
        try:
            import pandas as pd
            if not os.path.exists("original_parts.csv"):
                self.log_message("original_parts.csv not found; cannot populate parts list")
                return
            df = pd.read_csv("original_parts.csv")
            # Derive model id from selected file name (without extension)
            model_id = os.path.basename(self.selected_file).replace('.ply', '')
            # If the CSV has many models, filter; else fallback to all
            if 'model_id' in df.columns:
                dff = df[df['model_id'] == model_id]
                if dff.empty:
                    dff = df
            else:
                dff = df
            parts = list(dict.fromkeys(dff.get('part_id', [])))
            self._set_part_checkboxes(parts)
        except Exception as e:
            self.log_message(f"Failed to populate parts: {e}")

    def populate_parts_from_voxels(self):
        """Populate part list from voxel colors (no analysis needed)."""
        try:
            import numpy as np
            model_id = os.path.basename(self.selected_file).replace('.ply', '')
            npz_path = os.path.join('voxel_out', model_id, 'voxels_filled_indices_colors.npz')
            if not os.path.exists(npz_path):
                self.log_message(f"Voxel data not found: {npz_path}")
                return
            data = np.load(npz_path)
            colors = data.get('colors')
            if colors is None or len(colors) == 0:
                self.log_message("No colors found in voxel data.")
                return
            uniq, inv = np.unique(colors, axis=0, return_inverse=True)
            parts = []
            for idx, rgba in enumerate(uniq):
                pname = f"PART_{idx:03d}_R{int(rgba[0])}G{int(rgba[1])}B{int(rgba[2])}A{int(rgba[3])}"
                parts.append(pname)
            self._set_part_checkboxes(parts)
        except Exception as e:
            self.log_message(f"Failed to read voxel parts: {e}")

    def _set_part_checkboxes(self, parts):
        # Clear existing
        for w in list(self.part_container.winfo_children()):
            w.destroy()
        self.part_vars = {}
        for p in parts:
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(self.part_container, text=p, variable=var, style='Part.TCheckbutton')
            cb.pack(anchor='w', pady=2)
            self.part_vars[p] = var
        self.run_selected_btn.config(state=('normal' if parts else 'disabled'))

    def run_selected_recommendations(self):
        """Run recommend_thinning_final.py only for selected parts (stylish, safe)."""
        try:
            import pandas as pd
            # Build remaining pipeline before recommendation to ensure data
            self.progress.start()
            # Skip thinning for now; run AI on original parts only
            self.log_message("Skipping thinning (running AI on original parts)…")
            # FEA
            self.log_message("Running FEA analysis…")
            if not self.run_script("voxel_to_fea_complete.py"):
                self.progress.stop()
                return
            # Combine
            self.log_message("Combining FEA results…")
            if not self.run_script("combine_results.py"):
                self.progress.stop()
                return
            # Mass/Volume
            self.log_message("Adding mass/volume…")
            if not self.run_script("add_mass_volume.py"):
                self.progress.stop()
                return
            # Pairwise
            self.log_message("Building pairwise…")
            if not self.run_script("build_pairwise.py"):
                self.progress.stop()
                return
            # Ensure original_parts
            self.log_message("Updating original parts…")
            if not self.update_original_parts():
                self.progress.stop()
                return
            
            if not os.path.exists("original_parts.csv"):
                messagebox.showerror("Missing data", "original_parts.csv not found; run pipeline first.")
                self.progress.stop()
                return
            selected_parts = [p for p, v in self.part_vars.items() if v.get()]
            if not selected_parts:
                messagebox.showinfo("No parts selected", "Select one or more parts to run recommendations.")
                self.progress.stop()
                return
            # Backup and filter
            df_full = pd.read_csv("original_parts.csv")
            if 'part_id' not in df_full.columns:
                messagebox.showerror("Invalid file", "original_parts.csv missing 'part_id' column.")
                self.progress.stop()
                return
            df_sel = df_full[df_full['part_id'].isin(selected_parts)]
            if df_sel.empty:
                messagebox.showinfo("No match", "No rows matched the selected parts.")
                self.progress.stop()
                return
            backup_path = "original_parts.backup.csv"
            df_full.to_csv(backup_path, index=False)
            df_sel.to_csv("original_parts.csv", index=False)

            # Run recommender
            self.log_message(f"Running AI predictions for {len(selected_parts)} selected parts…")
            if not self.run_script("recommend_thinning_final.py"):
                # Restore and stop
                shutil.copy2(backup_path, "original_parts.csv")
                self.progress.stop()
                return
            
            # Apply AI thinning if enabled (COMMENTED OUT - using voxel thinning instead)
            # if self.apply_thinning_var.get():
            #     self.log_message("Applying AI thinning to voxels...")
            #     if not self.run_script("apply_ai_thinning.py"):
            #         self.log_message("Warning: AI thinning application failed, but continuing with results...")
            
            # Apply voxel SDF thinning if enabled
            if self.voxel_thinning_var.get():
                self.log_message("Applying voxel SDF thinning...")
                if not self.run_script("voxel_sdf_thinning.py"):
                    self.log_message("Warning: Voxel SDF thinning failed, but continuing with results...")
            
            # Apply surface offset thinning if enabled (COMMENTED OUT - not needed)
            # if self.surface_offset_var.get():
            #     self.log_message("Applying surface offset thinning...")
            #     if not self.run_script("surface_offset_thinning.py"):
            #         self.log_message("Warning: Surface offset thinning failed, but continuing with results...")
            
            # Restore full file
            shutil.copy2(backup_path, "original_parts.csv")
            # Show results
            self.show_results()
            self.progress.stop()
        except Exception as e:
            self.log_message(f"Failed to run selected recommendations: {e}")
    
    # Feasibility check removed per request; AI runs on original parts only

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
    
    def update_original_parts(self):
        """Update original_parts.csv with latest data"""
        try:
            import pandas as pd
            
            # Load the dataset
            if not os.path.exists("dataset_with_mass_volume.csv"):
                self.log_message("ERROR: dataset_with_mass_volume.csv not found")
                return False
            
            df = pd.read_csv("dataset_with_mass_volume.csv")
            
            # Filter for original parts only
            original_df = df[df['Model_Type'] == 'original'].copy()
            
            # Create the required columns
            original_df['model_id'] = original_df['Model']
            original_df['part_id'] = original_df['Part']
            original_df['orig_el'] = original_df['ElementCount']
            original_df['orig_mean'] = original_df['vonMises_mean_Pa']
            original_df['orig_std'] = original_df['vonMises_std_Pa']
            
            # Select only the required columns
            required_cols = ['model_id', 'part_id', 'orig_el', 'orig_mean', 'orig_std', 'pitch_m', 'volume_m3', 'density_kg_m3', 'mass_kg']
            original_parts = original_df[required_cols].copy()
            
            # Save the file
            original_parts.to_csv("original_parts.csv", index=False)
            
            self.log_message(f"Updated original_parts.csv with {len(original_parts)} parts")
            return True
            
        except Exception as e:
            self.log_message(f"Error updating original parts: {e}")
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
            optimized_parts = 0
            
            self.results_text.insert(tk.END, f"🎉 AI Analysis Complete! Found {len(results)} parts:\n\n")
            
            # Load thinning application report if available (COMMENTED OUT - using voxel thinning instead)
            # thinning_report = self.load_thinning_report()
            
            # Load surface offset thinning report if available (COMMENTED OUT - not needed)
            # surface_offset_report = self.load_surface_offset_report()
            
            # Load voxel thinning report if available
            voxel_thinning_report = self.load_voxel_thinning_report()
            
            for i, result in enumerate(results, 1):
                part_id = result['part_id']
                reduction = result['reduction']
                stress_change = result['stress_change']
                mass_saving = result['mass_saving']
                total_mass_saving += mass_saving
                
                if reduction > 0:
                    optimized_parts += 1
                    self.results_text.insert(tk.END, f"📋 Part {i}: {part_id}\n")
                    self.results_text.insert(tk.END, f"   ✅ Target reduction: {reduction:.1f}%\n")
                    
                    # Show achieved reduction if thinning was applied (COMMENTED OUT - using voxel thinning instead)
                    # if thinning_report and part_id in thinning_report:
                    #     achieved = thinning_report[part_id]['achieved_pct']
                    #     stop_reason = thinning_report[part_id]['stop_reason']
                    #     if achieved > 0:
                    #         self.results_text.insert(tk.END, f"   🎯 Achieved reduction: {achieved:.1f}%")
                    #         if stop_reason != "target_met":
                    #             self.results_text.insert(tk.END, f" (bound: {stop_reason})")
                    #         self.results_text.insert(tk.END, "\n")
                    #     else:
                    #         self.results_text.insert(tk.END, f"   ⚠️ No thinning applied ({stop_reason})\n")
                    
                    # Show voxel thinning results
                    if voxel_thinning_report and part_id in voxel_thinning_report:
                        voxel_data = voxel_thinning_report[part_id]
                        voxel_achieved = voxel_data['achieved_pct']
                        voxel_delta = voxel_data['delta_mm_used']
                        voxel_removed = voxel_data['removed_voxels']
                        voxel_reason = voxel_data['stop_reason']
                        if voxel_achieved > 0:
                            self.results_text.insert(tk.END, f"   📦 Voxel thinning: {voxel_achieved:.1f}% (Δ={voxel_delta:.3f}mm, removed={voxel_removed})")
                            if voxel_reason != "target_met":
                                self.results_text.insert(tk.END, f" (bound: {voxel_reason})")
                            self.results_text.insert(tk.END, "\n")
                        else:
                            self.results_text.insert(tk.END, f"   ⚠️ No voxel thinning applied ({voxel_reason})\n")
                    
                    # Show surface offset thinning results (COMMENTED OUT - not needed)
                    # if surface_offset_report and part_id in surface_offset_report:
                    #     surface_data = surface_offset_report[part_id]
                    #     surface_achieved = surface_data['achieved_pct']
                    #     surface_offset = surface_data['offset_mm']
                    #     surface_reason = surface_data['stop_reason']
                    #     if surface_achieved > 0:
                    #         self.results_text.insert(tk.END, f"   🔪 Surface offset: {surface_achieved:.1f}% (Δ={surface_offset:.3f}mm)")
                    #         if surface_reason != "target_met":
                    #             self.results_text.insert(tk.END, f" (bound: {surface_reason})")
                    #         self.results_text.insert(tk.END, "\n")
                    #     else:
                    #         self.results_text.insert(tk.END, f"   ⚠️ No surface offset applied ({surface_reason})\n")
                    
                    # Show prediction data if no thinning was applied
                    if not voxel_thinning_report or part_id not in voxel_thinning_report:
                        self.results_text.insert(tk.END, f"   📊 Predicted stress change: {stress_change:.1f}%\n")
                        self.results_text.insert(tk.END, f"   💰 Expected mass saving: {mass_saving:.3f} kg\n")
                else:
                    self.results_text.insert(tk.END, f"📋 Part {i}: {part_id}\n")
                    self.results_text.insert(tk.END, f"   ⚠️ No thinning recommended (safety constraints)\n")
                
                self.results_text.insert(tk.END, "\n")
            
            self.results_text.insert(tk.END, f"🎯 SUMMARY:\n")
            self.results_text.insert(tk.END, f"   Parts optimized: {optimized_parts}/{len(results)}\n")
            self.results_text.insert(tk.END, f"   Total mass saving: {total_mass_saving:.3f} kg\n")
            if optimized_parts > 0:
                avg_reduction = sum(r['reduction'] for r in results if r['reduction'] > 0) / optimized_parts
                self.results_text.insert(tk.END, f"   Average reduction: {avg_reduction:.1f}%\n")
            
            # Show thinning application status (COMMENTED OUT - using voxel thinning instead)
            # if self.apply_thinning_var.get():
            #     if thinning_report:
            #         applied_parts = len([r for r in thinning_report.values() if r['achieved_pct'] > 0])
            #         self.results_text.insert(tk.END, f"   🔧 Thinning applied: {applied_parts} parts\n")
            #     else:
            #         self.results_text.insert(tk.END, f"   🔧 Thinning: Not applied (no report found)\n")
            
            # Show voxel thinning status
            if self.voxel_thinning_var.get():
                if voxel_thinning_report:
                    voxel_applied_parts = len([r for r in voxel_thinning_report.values() if r['achieved_pct'] > 0])
                    avg_delta = sum(r['delta_mm_used'] for r in voxel_thinning_report.values() if r['achieved_pct'] > 0) / max(voxel_applied_parts, 1)
                    total_removed = sum(r['removed_voxels'] for r in voxel_thinning_report.values())
                    self.results_text.insert(tk.END, f"   📦 Voxel thinning applied: {voxel_applied_parts} parts (avg Δ={avg_delta:.3f}mm, removed {total_removed} voxels)\n")
                else:
                    self.results_text.insert(tk.END, f"   📦 Voxel thinning: Not applied (no report found)\n")
            
            # Show surface offset thinning status (COMMENTED OUT - not needed)
            # if self.surface_offset_var.get():
            #     if surface_offset_report:
            #         surface_applied_parts = len([r for r in surface_offset_report.values() if r['achieved_pct'] > 0])
            #         avg_offset = sum(r['offset_mm'] for r in surface_offset_report.values() if r['achieved_pct'] > 0) / max(surface_applied_parts, 1)
            #         self.results_text.insert(tk.END, f"   🔪 Surface offset applied: {surface_applied_parts} parts (avg Δ={avg_offset:.3f}mm)\n")
            #     else:
            #         self.results_text.insert(tk.END, f"   🔪 Surface offset: Not applied (no report found)\n")
            
            self.log_message(f"AI Recommendations: {len(results)} parts analyzed, {optimized_parts} optimized, {total_mass_saving:.3f} kg total savings")
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
    
    def load_thinning_report(self):
        """Load thinning application report if available"""
        try:
            if os.path.exists("thinning_apply_report.csv"):
                import pandas as pd
                df = pd.read_csv("thinning_apply_report.csv")
                return {row['part_id']: row.to_dict() for _, row in df.iterrows()}
        except Exception as e:
            self.log_message(f"Could not load thinning report: {e}")
        return None
    
    def load_surface_offset_report(self):
        """Load surface offset thinning report if available"""
        try:
            if os.path.exists("surface_offset_thinning_report.csv"):
                import pandas as pd
                df = pd.read_csv("surface_offset_thinning_report.csv")
                return {row['part_id']: row.to_dict() for _, row in df.iterrows()}
        except Exception as e:
            self.log_message(f"Could not load surface offset report: {e}")
        return None
    
    def load_voxel_thinning_report(self):
        """Load voxel thinning report if available"""
        try:
            if os.path.exists("voxel_thinning_apply_report.csv"):
                import pandas as pd
                df = pd.read_csv("voxel_thinning_apply_report.csv")
                return {row['part_id']: row.to_dict() for _, row in df.iterrows()}
        except Exception as e:
            self.log_message(f"Could not load voxel thinning report: {e}")
        return None

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

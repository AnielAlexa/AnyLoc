#!/usr/bin/env python3
"""
Aerial Video GPS Estimation GUI

Interactive GUI application that visualizes:
1. Satellite image patches in a grid layout
2. Current video frame being processed
3. Real-time matching between video frames and satellite patches
4. GPS trajectory plotting and confidence visualization

Usage:
    python3 aerial_video_gui.py --video flight.mp4 --database ./flight_results/anyloc_database
"""

import os
import sys
from pathlib import Path
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import torch
import threading
import time
from queue import Queue

from drone_navigation_system import DroneNavigationSystem
from utilities import seed_everything

class AerialVideoGUI:
    """Interactive GUI for aerial video GPS estimation visualization"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Aerial Video GPS Estimation - AnyLoc")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.video_path = None
        self.database_path = None
        self.nav_system = None
        self.video_cap = None
        self.satellite_patches = {}
        self.gps_metadata = {}
        self.current_frame = None
        self.current_matches = []
        self.is_playing = False
        self.frame_queue = Queue()
        self.satellite_composite_image = None
        self.patch_coordinates = {}  # Store patch positions in composite image
        
        # GUI components
        self.satellite_grid_frame = None
        self.video_frame_label = None
        self.control_panel = None
        self.info_panel = None
        self.progress_var = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Initialize the GUI layout"""
        
        # Create main menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Video", command=self.load_video)
        file_menu.add_command(label="Load Database", command=self.load_database)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Create main layout with paned windows
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: Split between whole satellite image and patch grid
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)
        
        # Top: Whole satellite image with highlighted patches
        satellite_overview_frame = ttk.LabelFrame(left_frame, text="Satellite Overview", padding=5)
        satellite_overview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.satellite_overview_label = ttk.Label(satellite_overview_frame, text="No satellite image loaded", 
                                                background="gray", anchor="center")
        self.satellite_overview_label.pack(fill=tk.BOTH, expand=True)
        
        # Bottom: Individual satellite patches grid
        patches_frame = ttk.LabelFrame(left_frame, text="Individual Patches", padding=5)
        patches_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable satellite grid
        self.setup_satellite_grid(patches_frame)
        
        # Right panel: Video and controls
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # Video display
        video_frame = ttk.LabelFrame(right_frame, text="Current Video Frame", padding=5)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.video_frame_label = ttk.Label(video_frame, text="No video loaded", 
                                         background="gray", anchor="center")
        self.video_frame_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        self.setup_control_panel(right_frame)
        
        # Information panel
        self.setup_info_panel(right_frame)
        
    def setup_satellite_grid(self, parent):
        """Setup scrollable grid for satellite patches"""
        
        # Create canvas with scrollbars
        canvas = tk.Canvas(parent, bg="white")
        v_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=canvas.xview)
        
        self.satellite_grid_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        canvas.create_window((0, 0), window=self.satellite_grid_frame, anchor="nw")
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Update scroll region when frame changes size
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        self.satellite_grid_frame.bind("<Configure>", configure_scroll_region)
        self.satellite_canvas = canvas
        
    def setup_control_panel(self, parent):
        """Setup video control panel"""
        
        control_frame = ttk.LabelFrame(parent, text="Controls", padding=5)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Button frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.play_button = ttk.Button(button_frame, text="Play", 
                                    command=self.toggle_playback, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.step_button = ttk.Button(button_frame, text="Step", 
                                    command=self.step_frame, state=tk.DISABLED)
        self.step_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.reset_button = ttk.Button(button_frame, text="Reset", 
                                     command=self.reset_video, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                          maximum=100, length=300)
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Speed control
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(speed_frame, from_=0.1, to=3.0, 
                              variable=self.speed_var, length=150)
        speed_scale.pack(side=tk.LEFT, padx=(5, 5))
        
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_label.pack(side=tk.LEFT)
        
        # Update speed label
        def update_speed_label(*args):
            self.speed_label.config(text=f"{self.speed_var.get():.1f}x")
        self.speed_var.trace('w', update_speed_label)
        
    def setup_info_panel(self, parent):
        """Setup information display panel"""
        
        info_frame = ttk.LabelFrame(parent, text="Match Information", padding=5)
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(info_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.info_text = tk.Text(text_frame, wrap=tk.WORD, height=8, 
                               font=("Consolas", 9))
        info_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, 
                                     command=self.info_text.yview)
        
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add initial text
        self.update_info_panel("Ready. Load video and database to begin.")
        
    def load_video(self):
        """Load video file"""
        
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            self.load_video_file()
            
    def load_database(self):
        """Load satellite image database"""
        
        dir_path = filedialog.askdirectory(
            title="Select Database Directory"
        )
        
        if dir_path:
            self.database_path = dir_path
            self.load_database_files()
            
    def load_video_file(self):
        """Initialize video capture"""
        
        try:
            if self.video_cap:
                self.video_cap.release()
                
            self.video_cap = cv2.VideoCapture(self.video_path)
            
            if not self.video_cap.isOpened():
                raise Exception("Could not open video file")
                
            # Get video properties
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            
            # Enable controls
            self.play_button.config(state=tk.NORMAL)
            self.step_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            
            # Load first frame
            self.load_current_frame()
            
            self.update_info_panel(f"Video loaded: {Path(self.video_path).name}\n"
                                 f"Frames: {self.total_frames}, FPS: {self.fps:.1f}")
                                 
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")
            
    def load_database_files(self):
        """Load satellite database and patches"""
        
        try:
            # Initialize navigation system
            self.nav_system = DroneNavigationSystem(self.database_path)
            
            # Load database
            database_file = f"{self.database_path}/satellite_database.json"
            features_file = f"{self.database_path}/satellite_features.pt"
            
            if not os.path.exists(database_file) or not os.path.exists(features_file):
                raise Exception("Database files not found")
                
            with open(database_file, 'r') as f:
                db_data = json.load(f)
                
            # Load satellite patches
            satellite_dir = Path(self.database_path).parent / "satellite_patches"
            patches_dir = satellite_dir / "patches"
            gps_file = satellite_dir / "gps_metadata.json"
            
            if gps_file.exists():
                with open(gps_file, 'r') as f:
                    self.gps_metadata = json.load(f)
                    
            # Load patch images
            self.satellite_patches = {}
            if patches_dir.exists():
                for img_file in patches_dir.glob("*.png"):
                    patch_name = img_file.stem
                    self.satellite_patches[patch_name] = str(img_file)
                    
            # Setup satellite grid display
            self.setup_satellite_patches_display()
            
            # Load navigation system models and database
            self.nav_system.load_satellite_database()
            
            self.update_info_panel(f"Database loaded: {len(self.satellite_patches)} patches\n"
                                 f"GPS metadata: {len(self.gps_metadata)} entries")
                                 
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database: {e}")
            
    def setup_satellite_patches_display(self):
        """Display satellite patches in grid layout"""
        
        # Clear existing widgets
        for widget in self.satellite_grid_frame.winfo_children():
            widget.destroy()
            
        if not self.satellite_patches:
            return
            
        # Calculate grid layout
        patch_count = len(self.satellite_patches)
        cols = min(4, patch_count)  # Max 4 columns
        rows = (patch_count + cols - 1) // cols
        
        # Create patch thumbnails
        patch_size = 120
        self.patch_labels = {}
        
        for i, (patch_name, patch_path) in enumerate(self.satellite_patches.items()):
            row = i // cols
            col = i % cols
            
            try:
                # Load and resize image
                img = Image.open(patch_path)
                img.thumbnail((patch_size, patch_size), Image.Resampling.LANCZOS)
                
                # Create frame for patch
                patch_frame = ttk.Frame(self.satellite_grid_frame, relief=tk.RAISED, borderwidth=1)
                patch_frame.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Create label
                patch_label = ttk.Label(patch_frame, image=photo)
                patch_label.image = photo  # Keep reference
                patch_label.pack(padx=2, pady=2)
                
                # Add patch name
                name_label = ttk.Label(patch_frame, text=patch_name, font=("Arial", 8))
                name_label.pack()
                
                # Store reference
                self.patch_labels[patch_name] = {
                    'frame': patch_frame,
                    'label': patch_label,
                    'name_label': name_label,
                    'original_photo': photo
                }
                
            except Exception as e:
                print(f"Error loading patch {patch_name}: {e}")
                
        # Update scroll region
        self.satellite_grid_frame.update_idletasks()
        self.satellite_canvas.configure(scrollregion=self.satellite_canvas.bbox("all"))
        
        # Create composite satellite image
        self.create_satellite_composite()
        
    def load_current_frame(self):
        """Load and display current video frame"""
        
        if not self.video_cap:
            return
            
        ret, frame = self.video_cap.read()
        if ret:
            self.current_frame = frame
            self.display_video_frame(frame)
            
            # Update progress
            current_pos = self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)
            progress = (current_pos / self.total_frames) * 100
            self.progress_var.set(progress)
            
    def display_video_frame(self, frame):
        """Display video frame in GUI"""
        
        # Resize frame for display
        display_height = 300
        aspect_ratio = frame.shape[1] / frame.shape[0]
        display_width = int(display_height * aspect_ratio)
        
        frame_resized = cv2.resize(frame, (display_width, display_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(img)
        
        # Update label
        self.video_frame_label.config(image=photo, text="")
        self.video_frame_label.image = photo
        
    def process_current_frame(self):
        """Process current frame for GPS matching"""
        
        if not self.current_frame.any() or not self.nav_system:
            return
            
        try:
            # Process frame with navigation system
            match_result = self.nav_system.match_drone_frame(self.current_frame)
            
            # Update patch highlighting
            self.update_patch_highlighting(match_result)
            
            # Update information panel
            self.update_match_info(match_result)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            
    def update_patch_highlighting(self, match_result):
        """Highlight matched satellite patches"""
        
        # Reset all patches to normal
        for patch_name, patch_info in self.patch_labels.items():
            patch_info['frame'].config(relief=tk.RAISED, borderwidth=1)
            
        # Highlight best match
        if match_result and match_result.get('best_match_image'):
            best_match = match_result['best_match_image']
            confidence = match_result.get('similarity', 0)
            
            if best_match in self.patch_labels:
                # Highlight with thick green border
                self.patch_labels[best_match]['frame'].config(
                    relief=tk.SOLID, borderwidth=3
                )
                
                # Create colored border overlay
                self.create_match_overlay(best_match, confidence)
                
            # Update composite satellite image with highlighting
            self.display_satellite_composite(best_match, confidence)
        else:
            # Clear highlighting in composite image
            self.display_satellite_composite()
                
    def create_match_overlay(self, patch_name, confidence):
        """Create colored overlay for matched patch"""
        
        if patch_name not in self.patch_labels:
            return
            
        patch_info = self.patch_labels[patch_name]
        original_photo = patch_info['original_photo']
        
        # Create overlay with confidence color
        if confidence > 0.5:
            border_color = "green"
        elif confidence > 0.3:
            border_color = "orange"  
        else:
            border_color = "red"
            
        # Add confidence text overlay
        patch_info['frame'].config(relief=tk.SOLID, borderwidth=3)
        
    def update_match_info(self, match_result):
        """Update information panel with match details"""
        
        if not match_result:
            return
            
        info_text = f"Frame: {int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))}\n"
        
        if match_result.get('is_valid_match', False):
            info_text += f"✅ MATCH FOUND\n"
            info_text += f"Best Match: {match_result.get('best_match_image', 'N/A')}\n"
            info_text += f"Confidence: {match_result.get('similarity', 0):.3f}\n"
            
            if match_result.get('gps_estimate'):
                gps = match_result['gps_estimate']
                info_text += f"GPS: {gps.get('lat', 0):.6f}, {gps.get('lon', 0):.6f}\n"
                info_text += f"Altitude: {gps.get('altitude', 'N/A')}m\n"
        else:
            info_text += f"❌ No valid match\n"
            info_text += f"Best similarity: {match_result.get('similarity', 0):.3f}\n"
            
        info_text += f"Processing time: {match_result.get('processing_time', 0):.3f}s\n"
        info_text += "=" * 40 + "\n"
        
        # Update text widget
        self.info_text.insert(tk.END, info_text)
        self.info_text.see(tk.END)  # Scroll to bottom
        
    def update_info_panel(self, message):
        """Update information panel with general message"""
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, message)
        
    def toggle_playback(self):
        """Toggle video playback"""
        
        if not self.is_playing:
            self.start_playback()
        else:
            self.stop_playback()
            
    def start_playback(self):
        """Start video playback"""
        
        self.is_playing = True
        self.play_button.config(text="Pause")
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self.playback_loop, daemon=True)
        self.playback_thread.start()
        
    def stop_playback(self):
        """Stop video playback"""
        
        self.is_playing = False
        self.play_button.config(text="Play")
        
    def playback_loop(self):
        """Main playback loop (runs in separate thread)"""
        
        while self.is_playing and self.video_cap:
            start_time = time.time()
            
            # Load next frame
            ret, frame = self.video_cap.read()
            if not ret:
                # End of video
                self.is_playing = False
                self.root.after(0, lambda: self.play_button.config(text="Play"))
                break
                
            # Update GUI in main thread
            self.current_frame = frame
            self.root.after(0, lambda f=frame: self.display_video_frame(f))
            self.root.after(0, self.process_current_frame)
            
            # Update progress
            current_pos = self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)
            progress = (current_pos / self.total_frames) * 100
            self.root.after(0, lambda p=progress: self.progress_var.set(p))
            
            # Control playback speed
            elapsed = time.time() - start_time
            target_delay = (1.0 / self.fps) / self.speed_var.get()
            sleep_time = max(0, target_delay - elapsed)
            time.sleep(sleep_time)
            
    def step_frame(self):
        """Step one frame forward"""
        
        if self.video_cap:
            self.load_current_frame()
            self.process_current_frame()
            
    def reset_video(self):
        """Reset video to beginning"""
        
        if self.video_cap:
            self.stop_playback()
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.load_current_frame()
            self.progress_var.set(0)
            
            # Clear info panel
            self.update_info_panel("Video reset to beginning.")
    
    def create_satellite_composite(self):
        """Create composite satellite image from individual patches"""
        
        if not self.satellite_patches or not self.gps_metadata:
            return
            
        try:
            # Extract coordinates and sort patches by GPS position
            patch_coords = []
            for patch_name, patch_path in self.satellite_patches.items():
                if patch_name in self.gps_metadata:
                    gps_data = self.gps_metadata[patch_name]
                    lat = gps_data.get('lat', 0)
                    lon = gps_data.get('lon', 0)
                    patch_coords.append({
                        'name': patch_name,
                        'path': patch_path,
                        'lat': lat,
                        'lon': lon
                    })
            
            if not patch_coords:
                return
                
            # Sort by lat/lon to determine grid layout
            patch_coords.sort(key=lambda x: (-x['lat'], x['lon']))  # Sort by lat desc, lon asc
            
            # Determine grid dimensions
            unique_lats = sorted(list(set([p['lat'] for p in patch_coords])), reverse=True)
            unique_lons = sorted(list(set([p['lon'] for p in patch_coords])))
            
            rows = len(unique_lats)
            cols = len(unique_lons)
            
            if rows == 0 or cols == 0:
                return
                
            # Load first patch to get dimensions
            first_patch = Image.open(patch_coords[0]['path'])
            patch_width, patch_height = first_patch.size
            
            # Create composite image
            composite_width = cols * patch_width
            composite_height = rows * patch_height
            composite = Image.new('RGB', (composite_width, composite_height), color='black')
            
            # Place patches in composite and store coordinates
            self.patch_coordinates = {}
            
            for patch_info in patch_coords:
                # Find grid position
                lat_idx = unique_lats.index(patch_info['lat'])
                lon_idx = unique_lons.index(patch_info['lon'])
                
                # Calculate pixel position
                x = lon_idx * patch_width
                y = lat_idx * patch_height
                
                # Load and paste patch
                patch_img = Image.open(patch_info['path'])
                composite.paste(patch_img, (x, y))
                
                # Store coordinates for highlighting
                self.patch_coordinates[patch_info['name']] = {
                    'x': x,
                    'y': y,
                    'width': patch_width,
                    'height': patch_height
                }
            
            # Store composite for highlighting
            self.satellite_composite_image = composite
            
            # Display composite in GUI
            self.display_satellite_composite()
            
        except Exception as e:
            print(f"Error creating satellite composite: {e}")
    
    def display_satellite_composite(self, highlighted_patch=None, confidence=0.0):
        """Display satellite composite with optional patch highlighting"""
        
        if not self.satellite_composite_image:
            return
            
        try:
            # Create working copy
            display_image = self.satellite_composite_image.copy()
            
            # Add highlighting if patch specified
            if highlighted_patch and highlighted_patch in self.patch_coordinates:
                draw = ImageDraw.Draw(display_image)
                coords = self.patch_coordinates[highlighted_patch]
                
                # Determine highlight color based on confidence
                if confidence > 0.5:
                    color = 'lime'
                    width = 4
                elif confidence > 0.3:
                    color = 'orange'  
                    width = 3
                else:
                    color = 'red'
                    width = 2
                
                # Draw highlight border
                x, y = coords['x'], coords['y']
                w, h = coords['width'], coords['height']
                
                # Draw thick border
                for i in range(width):
                    draw.rectangle([x-i, y-i, x+w+i, y+h+i], outline=color, width=1)
                
                # Add confidence text
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                text = f"{confidence:.3f}"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Position text at top-left of highlighted patch
                text_x = x + 5
                text_y = y + 5
                
                # Draw text background
                draw.rectangle([text_x-2, text_y-2, text_x+text_width+2, text_y+text_height+2], 
                             fill='white', outline=color)
                draw.text((text_x, text_y), text, fill=color, font=font)
            
            # Resize for display
            display_width = 400
            aspect_ratio = display_image.width / display_image.height
            display_height = int(display_width / aspect_ratio)
            
            display_image = display_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_image)
            
            # Update label
            self.satellite_overview_label.config(image=photo, text="")
            self.satellite_overview_label.image = photo  # Keep reference
            
        except Exception as e:
            print(f"Error displaying satellite composite: {e}")

def main():
    """Main application entry point"""
    
    parser = argparse.ArgumentParser(description="Aerial Video GPS Estimation GUI")
    parser.add_argument("--video", help="Path to video file")
    parser.add_argument("--database", help="Path to satellite database directory")
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(42)
    
    # Create GUI
    root = tk.Tk()
    app = AerialVideoGUI(root)
    
    # Load files if provided
    if args.video:
        app.video_path = args.video
        app.load_video_file()
        
    if args.database:
        app.database_path = args.database
        app.load_database_files()
        
    # Start GUI
    root.mainloop()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Step 3: Match Video Against Satellite Database with GUI

Processes flight_100m.mp4 and matches frames against satellite patches.
Shows real-time GUI with matches and saves results to JSON.
"""

import os
import sys
from pathlib import Path
import yaml
import json
import time
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
from queue import Queue

# Add parent directory to path for utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities import DinoV2ExtractFeatures, VLAD, seed_everything


class VideoMatcherGUI:
    """GUI for real-time video matching."""

    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.root.title("DINOv2 ViT-S Layer Testing - Video Matcher")
        self.root.geometry("1600x900")

        # State variables
        self.video_cap = None
        self.current_config = None
        self.extractor = None
        self.vlad = None
        self.db_vlads = None
        self.patch_names = []
        self.gps_metadata = {}
        self.is_playing = False
        self.current_frame_idx = 0
        self.total_frames = 0
        self.results = {}

        # Load GPS metadata
        self.load_gps_metadata()

        # Setup GUI
        self.setup_gui()

        # Load video
        self.load_video()

        # Get available configurations
        self.configurations = self.get_configurations()
        self.config_var.set(self.configurations[0] if self.configurations else "")

        # Load first configuration
        if self.configurations:
            self.load_configuration(self.configurations[0])

    def load_gps_metadata(self):
        """Load GPS metadata for patches."""
        metadata_path = os.path.join(self.config['patches']['output_dir'], 'gps_metadata.json')
        with open(metadata_path, 'r') as f:
            self.gps_metadata = json.load(f)

    def setup_gui(self):
        """Setup GUI layout."""

        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Main layout
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Video
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)

        # Video display
        video_frame = ttk.LabelFrame(left_frame, text="Video Frame", padding=5)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.video_label = ttk.Label(video_frame, text="Loading video...", background="gray")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Controls
        controls_frame = ttk.LabelFrame(left_frame, text="Controls", padding=5)
        controls_frame.pack(fill=tk.X, pady=(0, 5))

        # Configuration selector
        ttk.Label(controls_frame, text="Configuration:").pack(side=tk.LEFT, padx=(0, 5))
        self.config_var = tk.StringVar()
        self.config_combo = ttk.Combobox(controls_frame, textvariable=self.config_var,
                                        state="readonly", width=25)
        self.config_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.config_combo.bind("<<ComboboxSelected>>", self.on_config_change)

        # Buttons
        self.play_button = ttk.Button(controls_frame, text="▶ Play",
                                     command=self.toggle_playback, width=10)
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))

        self.step_button = ttk.Button(controls_frame, text="Step",
                                     command=self.step_frame, width=10)
        self.step_button.pack(side=tk.LEFT, padx=(0, 5))

        self.reset_button = ttk.Button(controls_frame, text="Reset",
                                      command=self.reset_video, width=10)
        self.reset_button.pack(side=tk.LEFT)

        # Progress
        progress_frame = ttk.Frame(controls_frame)
        progress_frame.pack(fill=tk.X, pady=(5, 0))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100)
        self.progress_bar.pack(fill=tk.X)

        # Performance display
        perf_frame = ttk.LabelFrame(left_frame, text="Performance", padding=5)
        perf_frame.pack(fill=tk.X)

        self.perf_label = ttk.Label(perf_frame, text="Ready", font=("Arial", 10))
        self.perf_label.pack()

        # Right panel: Info and matches
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # Info panel
        info_frame = ttk.LabelFrame(right_frame, text="Match Information", padding=5)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.info_text = tk.Text(info_frame, wrap=tk.WORD, height=15,
                                font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL,
                                 command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Best match display
        match_frame = ttk.LabelFrame(right_frame, text="Best Match", padding=5)
        match_frame.pack(fill=tk.BOTH, expand=True)

        self.match_label = ttk.Label(match_frame, text="No match yet",
                                    background="gray")
        self.match_label.pack(fill=tk.BOTH, expand=True)

    def get_configurations(self):
        """Get list of available configurations."""
        desc_dir = "data/descriptors"
        if not os.path.exists(desc_dir):
            return []

        configs = []
        for folder in sorted(os.listdir(desc_dir)):
            if folder.startswith("layer_"):
                configs.append(folder)

        return configs

    def load_configuration(self, config_name):
        """Load a specific layer/facet configuration."""

        # Parse configuration name (e.g., "layer_9_key")
        parts = config_name.split('_')
        layer = int(parts[1])
        facet = parts[2]

        print(f"\n📥 Loading configuration: Layer {layer}, Facet '{facet}'")

        # Initialize extractor
        self.extractor = DinoV2ExtractFeatures(
            dino_model=self.config['dinov2']['model'],
            layer=layer,
            facet=facet,
            use_cls=False,
            norm_descs=True,
            device=self.config['dinov2']['device']
        )

        # Load VLAD descriptors
        desc_dir = f"data/descriptors/{config_name}"
        self.db_vlads = torch.load(os.path.join(desc_dir, "vlad_descriptors.pt"))

        # Normalize database descriptors and move to device
        self.db_vlads = F.normalize(self.db_vlads, p=2, dim=-1)
        self.db_vlads = self.db_vlads.to(self.config['dinov2']['device'])

        # Load patch names
        with open(os.path.join(desc_dir, "patch_names.txt"), 'r') as f:
            self.patch_names = [line.strip() for line in f.readlines()]

        # Load VLAD cluster centers
        self.vlad = VLAD(
            num_clusters=self.config['vlad']['num_clusters'],
            desc_dim=None,  # Will be inferred
            cache_dir=desc_dir
        )
        self.vlad.fit(None)  # Load from cache

        self.current_config = config_name

        # Initialize results for this configuration
        if config_name not in self.results:
            self.results[config_name] = {
                "layer": layer,
                "facet": facet,
                "matches": [],
                "avg_confidence": 0.0,
                "avg_fps": 0.0
            }

        print(f"✓ Configuration loaded: {len(self.patch_names)} patches in database")

    def load_video(self):
        """Load video file."""
        video_path = os.path.join("..", self.config['video']['path'])

        if not os.path.exists(video_path):
            print(f"⚠️  Video not found: {video_path}")
            return

        self.video_cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"✓ Video loaded: {self.total_frames} frames")

        # Show first frame
        self.show_frame()

    def process_frame(self, frame):
        """Process a single frame and return top matches."""

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Transform
        transform = T.Compose([
            T.Resize((self.config['dinov2']['image_size'],
                     self.config['dinov2']['image_size'])),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        frame_tensor = transform(frame_pil).unsqueeze(0).to(self.config['dinov2']['device'])

        start_time = time.time()

        # Extract features
        with torch.no_grad():
            descriptors = self.extractor(frame_tensor)  # (1, num_patches, desc_dim)

            # Generate VLAD descriptor (move to CPU for VLAD computation)
            query_vlad = self.vlad.generate_multi(descriptors.cpu())  # (1, vlad_dim)
            query_vlad = F.normalize(query_vlad, p=2, dim=-1)

            # Move back to device for matching
            query_vlad = query_vlad.to(self.config['dinov2']['device'])

            # Match against database
            similarities = torch.mm(query_vlad, self.db_vlads.T)  # (1, N)
            top_k_vals, top_k_idx = torch.topk(similarities[0], k=self.config['video']['top_k_matches'])

        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        # Get results
        top_matches = []
        for i in range(len(top_k_idx)):
            idx = top_k_idx[i].item()
            confidence = top_k_vals[i].item()
            patch_name = self.patch_names[idx]

            top_matches.append({
                "patch_name": patch_name,
                "confidence": confidence,
                "gps": self.gps_metadata.get(patch_name, {})
            })

        return top_matches, fps, processing_time

    def show_frame(self):
        """Display current video frame."""
        if self.video_cap is None:
            return

        ret, frame = self.video_cap.read()
        if not ret:
            self.is_playing = False
            self.play_button.config(text="▶ Play")
            return

        # Resize for display
        display_width = 800
        h, w = frame.shape[:2]
        scale = display_width / w
        display_height = int(h * scale)
        frame_resized = cv2.resize(frame, (display_width, display_height))

        # Convert to PIL
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(frame_pil)
        self.video_label.config(image=photo)
        self.video_label.image = photo

        return frame

    def step_frame(self):
        """Process and display next frame."""
        if self.video_cap is None or self.current_config is None:
            return

        # Check if we should process this frame
        if self.current_frame_idx % self.config['video']['frame_sampling'] != 0:
            self.current_frame_idx += 1
            frame = self.show_frame()
            self.update_progress()
            return

        # Show frame
        frame = self.show_frame()
        if frame is None:
            return

        # Process frame
        top_matches, fps, proc_time = self.process_frame(frame)

        # Update display
        self.update_info_display(top_matches, fps, proc_time)
        self.update_match_display(top_matches[0])

        # Save result
        timestamp = self.current_frame_idx / 30.0  # Assume 30 FPS
        self.results[self.current_config]["matches"].append({
            "frame_idx": self.current_frame_idx,
            "timestamp": timestamp,
            "top_match": top_matches[0]["patch_name"],
            "confidence": top_matches[0]["confidence"],
            "gps": top_matches[0]["gps"],
            "top_5": [m["patch_name"] for m in top_matches]
        })

        self.current_frame_idx += 1
        self.update_progress()

    def update_info_display(self, top_matches, fps, proc_time):
        """Update info text with match results."""
        self.info_text.delete(1.0, tk.END)

        info = f"Frame: {self.current_frame_idx}/{self.total_frames}\n"
        info += f"FPS: {fps:.1f}\n"
        info += f"Processing Time: {proc_time * 1000:.1f} ms\n\n"
        info += "Top 5 Matches:\n"
        info += "-" * 50 + "\n"

        for i, match in enumerate(top_matches, 1):
            confidence = match['confidence']
            symbol = "✓" if confidence > self.config['video']['confidence_threshold'] else ""
            info += f"{i}. {match['patch_name']}\n"
            info += f"   Confidence: {confidence:.3f} {symbol}\n"
            if 'lat' in match['gps']:
                info += f"   GPS: {match['gps']['lat']:.6f}, {match['gps']['lon']:.6f}\n"
            info += "\n"

        self.info_text.insert(1.0, info)

        # Update performance label
        self.perf_label.config(text=f"FPS: {fps:.1f} | Time: {proc_time * 1000:.1f} ms")

    def update_match_display(self, best_match):
        """Update best match satellite image display."""
        patch_name = best_match['patch_name']
        patch_path = os.path.join(self.config['patches']['output_dir'],
                                  f"{patch_name}.png")

        if os.path.exists(patch_path):
            patch_img = Image.open(patch_path)

            # Resize for display
            display_size = 400
            patch_img = patch_img.resize((display_size, display_size), Image.Resampling.LANCZOS)

            # Add confidence text
            draw = ImageDraw.Draw(patch_img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            text = f"Confidence: {best_match['confidence']:.3f}"
            draw.text((10, 10), text, fill=(0, 255, 0), font=font)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(patch_img)
            self.match_label.config(image=photo)
            self.match_label.image = photo

    def update_progress(self):
        """Update progress bar."""
        if self.total_frames > 0:
            progress = (self.current_frame_idx / self.total_frames) * 100
            self.progress_var.set(progress)

    def toggle_playback(self):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_button.config(text="⏸ Pause")
            self.play_video()
        else:
            self.play_button.config(text="▶ Play")

    def play_video(self):
        """Play video continuously."""
        if not self.is_playing:
            return

        self.step_frame()

        if self.current_frame_idx < self.total_frames:
            self.root.after(33, self.play_video)  # ~30 FPS
        else:
            self.is_playing = False
            self.play_button.config(text="▶ Play")

    def reset_video(self):
        """Reset video to beginning."""
        if self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self.show_frame()
            self.update_progress()

    def on_config_change(self, event):
        """Handle configuration change."""
        new_config = self.config_var.get()
        self.load_configuration(new_config)
        self.reset_video()

    def export_results(self):
        """Export results to JSON."""
        # Calculate averages
        for config_name, config_results in self.results.items():
            matches = config_results["matches"]
            if matches:
                avg_conf = np.mean([m["confidence"] for m in matches])
                config_results["avg_confidence"] = float(avg_conf)

        output = {
            "video": self.config['video']['path'],
            "total_frames_processed": sum(len(r["matches"]) for r in self.results.values()),
            "configurations": self.results
        }

        output_path = "data/results/flight_100m_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")


def main():
    """Main execution function."""

    # Set seed
    seed_everything(42)

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("🎥 Starting video matcher GUI...")

    # Create GUI
    root = tk.Tk()
    app = VideoMatcherGUI(root, config)

    # Update configurations dropdown
    app.config_combo['values'] = app.configurations

    # Run GUI
    root.mainloop()

    # Save results on exit
    app.export_results()

    print("\n✅ Video matching complete!")


if __name__ == "__main__":
    main()

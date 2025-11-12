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

try:
    from lightglue import SuperPoint, LightGlue
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False

try:
    from copy import deepcopy
    sys.path.insert(0, str(Path(__file__).parent / "EfficientLoFTR"))
    from src.loftr import LoFTR, opt_default_cfg, reparameter
    EFFICIENTLOFTR_AVAILABLE = True
except ImportError:
    EFFICIENTLOFTR_AVAILABLE = False
    print("⚠️  EfficientLoFTR not available - using SuperPoint+LightGlue only")


# MatchAnything preprocessing functions (from official repo)
def process_resize(w, h, resize=None, df=None):
    """Calculate new dimensions divisible by df."""
    if resize is not None:
        assert(len(resize) > 0 and len(resize) <= 2)
        if len(resize) == 1 and resize[0] > -1:
            scale = resize[0] / max(h, w)
            w_new, h_new = int(round(w*scale)), int(round(h*scale))
        elif len(resize) == 1 and resize[0] == -1:
            w_new, h_new = w, h
        else:  # len(resize) == 2
            w_new, h_new = resize[0], resize[1]
    else:
        w_new, h_new = w, h

    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])
    return w_new, h_new


def resize_image_pil(image, size):
    """Resize image using PIL LANCZOS."""
    import PIL.Image
    resized = PIL.Image.fromarray(image.astype(np.uint8))
    resized = resized.resize(size, resample=PIL.Image.LANCZOS)
    return np.asarray(resized, dtype=image.dtype)


def pad_bottom_right(inp, pad_size, ret_mask=False):
    """Pad image to square size, optionally return mask."""
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:])
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    else:
        raise NotImplementedError()
    return padded, mask


def resize_and_pad(img, df=32, padding=True):
    """MatchAnything preprocessing: resize to multiples of df and pad."""
    h, w = img.shape[:2]
    w_new, h_new = process_resize(w, h, resize=None, df=df)
    img_new = resize_image_pil(img, (w_new, h_new)) if (w_new, h_new) != (w, h) else img

    # Calculate scales for later keypoint correction
    h_scale, w_scale = h / img_new.shape[0], w / img_new.shape[1]

    mask = None
    if padding and df is not None:
        img_new, mask = pad_bottom_right(img_new, max(h_new, w_new), ret_mask=True)

    return img_new, (h_scale, w_scale), mask


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

        # SuperPoint + LightGlue
        self.superpoint = None
        self.lightglue = None
        self.superpoint_kpts = None
        self.superpoint_descs = None
        self.current_frame = None

        # EfficientLoFTR
        self.efficientloftr_matcher = None
        self.matcher_type = None
        self.patch_images = []  # Store patch PIL images for EfficientLoFTR

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

        # Top 3 matches display
        match_frame = ttk.LabelFrame(right_frame, text="Top 3 Matches", padding=5)
        match_frame.pack(fill=tk.BOTH, expand=True)

        self.match_label = ttk.Label(match_frame, text="No matches yet",
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

        # Determine matcher type from config
        self.matcher_type = self.config.get('geometric_verification', {}).get('matcher', 'superpoint_lightglue')

        # Load appropriate matcher
        if self.matcher_type == "superpoint_lightglue" and LIGHTGLUE_AVAILABLE:
            sp_kpts_path = os.path.join(desc_dir, "superpoint_kpts.pt")
            sp_descs_path = os.path.join(desc_dir, "superpoint_descs.pt")

            if os.path.exists(sp_kpts_path) and os.path.exists(sp_descs_path):
                self.superpoint_kpts = torch.load(sp_kpts_path)
                self.superpoint_descs = torch.load(sp_descs_path)

                # Initialize SuperPoint and LightGlue with more keypoints but higher quality
                self.superpoint = SuperPoint(
                    max_num_keypoints=4096,  # More keypoints
                    detection_threshold=0.001  # Lower threshold = more keypoints detected
                ).eval().to(self.config['dinov2']['device'])
                self.lightglue = LightGlue(
                    features="superpoint",
                    filter_threshold=0.5  # Higher = more confident matches only
                ).eval().to(self.config['dinov2']['device'])

                print(f"✓ SuperPoint+LightGlue loaded")

        elif self.matcher_type == "efficientloftr" and EFFICIENTLOFTR_AVAILABLE:
            # Load EfficientLoFTR native implementation

            # Determine which weight variant to use
            variant = self.config.get('geometric_verification', {}).get('efficientloftr_variant', 'outdoor')
            weights_map = {
                'outdoor': 'eloftr_outdoor.ckpt',
                'matchanything': 'matchanything_eloftr.ckpt'
            }

            if variant not in weights_map:
                print(f"⚠️  Unknown EfficientLoFTR variant: '{variant}', defaulting to 'outdoor'")
                variant = 'outdoor'

            print(f"Loading EfficientLoFTR-opt model (variant: {variant})...")

            # Load config for 'opt' variant (fast)
            _default_cfg = deepcopy(opt_default_cfg)

            # Check if we should use MatchAnything's tuned config
            use_ma_config = self.config.get('geometric_verification', {}).get('efficientloftr_use_matchanything_config', False)

            if use_ma_config and variant == 'matchanything':
                # Apply MatchAnything's research-tuned settings (from eloftr_model.py)
                print(f"   Applying MatchAnything tuned config...")
                _default_cfg['match_coarse']['match_type'] = 'dual_softmax'  # More robust than skip_softmax
                _default_cfg['match_coarse']['skip_softmax'] = False
                _default_cfg['match_coarse']['force_nearest'] = True  # Prevent bad matches
                _default_cfg['match_coarse']['dsmax_temperature'] = 0.1  # Sharpness
                _default_cfg['match_coarse']['thr'] = self.config.get('geometric_verification', {}).get('efficientloftr_threshold', 0.15)
                print(f"   • Match type: dual_softmax")
                print(f"   • Force nearest: True")
                print(f"   • Temperature: 0.1")
                print(f"   • Threshold: {_default_cfg['match_coarse']['thr']} (0-1 probability scale)")
            else:
                # Use raw EfficientLoFTR config (legacy mode)
                loftr_threshold = self.config.get('geometric_verification', {}).get('efficientloftr_threshold', 0.15)
                # Note: If threshold > 1.0, it's likely using old scale (0-40), convert it
                if loftr_threshold > 1.0:
                    print(f"⚠️  Warning: threshold {loftr_threshold} seems to be on old scale (0-40)")
                    print(f"   Converting to 0-1 scale by dividing by 100")
                    loftr_threshold = loftr_threshold / 100.0
                _default_cfg['match_coarse']['thr'] = loftr_threshold
                print(f"   Matching threshold: {loftr_threshold} (0-1 scale)")

            # Initialize matcher
            self.efficientloftr_matcher = LoFTR(config=_default_cfg)

            # Try to load weights if available
            weights_filename = weights_map[variant]
            weights_path = Path(__file__).parent / "EfficientLoFTR" / "weights" / weights_filename

            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
                self.efficientloftr_matcher.load_state_dict(state_dict)
                self.efficientloftr_matcher = reparameter(self.efficientloftr_matcher)
                print(f"✓ Loaded {variant} weights from {weights_path.name}")
            else:
                print(f"⚠️  Weights not found at {weights_path}")
                print(f"   Available variants: {', '.join(weights_map.keys())}")
                print(f"   Download from: https://github.com/zju3dv/EfficientLoFTR")
                self.efficientloftr_matcher = None
                self.matcher_type = "superpoint_lightglue"  # Fallback

            if self.efficientloftr_matcher is not None:
                self.efficientloftr_matcher = self.efficientloftr_matcher.eval().to(self.config['dinov2']['device'])

                # Load patch images for EfficientLoFTR matching
                patches_dir = self.config['patches']['output_dir']
                self.patch_images = []
                for patch_name in self.patch_names:
                    patch_path = os.path.join(patches_dir, f"{patch_name}.png")
                    if os.path.exists(patch_path):
                        self.patch_images.append(Image.open(patch_path).convert('RGB'))
                    else:
                        self.patch_images.append(None)

                print(f"✓ EfficientLoFTR loaded")

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

        # Save current frame for keypoint visualization
        self.current_frame = frame.copy()

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
        self.update_match_display(top_matches[:3])  # Pass top 3 matches

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

    def extract_keypoints(self, frame_img):
        """Extract SuperPoint keypoints from image."""
        if self.superpoint is None or not LIGHTGLUE_AVAILABLE:
            return None, None

        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])

        img_tensor = transform(frame_img).unsqueeze(0).to(self.config['dinov2']['device'])

        with torch.no_grad():
            features = self.superpoint({"image": img_tensor})

        kpts = features.get("keypoints", None)  # (1, N, 2)
        desc = features.get("descriptors", None)  # (1, N, 256)

        return kpts, desc

    def draw_keypoints_and_matches(self, frame_cv, patch_cv, patch_idx):
        """Draw SuperPoint keypoints and LightGlue matches between frame and patch."""
        num_matches = 0

        if self.superpoint is None or not LIGHTGLUE_AVAILABLE:
            return frame_cv, patch_cv, num_matches

        # Convert to PIL for keypoint extraction
        frame_pil = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
        patch_pil = Image.fromarray(cv2.cvtColor(patch_cv, cv2.COLOR_BGR2RGB))

        # Extract keypoints
        query_kpts, query_desc = self.extract_keypoints(frame_pil)
        db_kpts, db_desc = self.superpoint_kpts[patch_idx], self.superpoint_descs[patch_idx]

        if query_kpts is None or db_kpts is None:
            return frame_cv, patch_cv, num_matches

        # Match with LightGlue
        with torch.no_grad():
            matches = self.lightglue({
                "image0": {"keypoints": query_kpts, "descriptors": query_desc},
                "image1": {"keypoints": db_kpts, "descriptors": db_desc}
            })

        matches_idx = matches["matches"]  # Match indices

        if isinstance(matches_idx, list):
            num_matches = len(matches_idx)
        else:
            num_matches = matches_idx.shape[0]

        # Draw keypoints on both images
        frame_np = np.array(frame_pil)
        patch_np = np.array(patch_pil)

        # Draw query keypoints
        if query_kpts is not None and query_kpts.shape[1] > 0:
            kpts_np = query_kpts[0].cpu().numpy()  # (N, 2)
            for kpt in kpts_np:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(frame_np, (x, y), 3, (0, 255, 0), -1)

        # Draw database keypoints
        if db_kpts is not None and db_kpts.shape[1] > 0:
            kpts_np = db_kpts[0].cpu().numpy()  # (N, 2)
            for kpt in kpts_np:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(patch_np, (x, y), 3, (0, 255, 0), -1)

        # Draw match lines and highlighted circles if matches exist
        if num_matches > 0 and not isinstance(matches_idx, list):
            matches_np = matches_idx.cpu().numpy()
            query_kpts_np = query_kpts[0].cpu().numpy()
            db_kpts_np = db_kpts[0].cpu().numpy()

            # Draw all matches with lines and highlighted circles
            colors = [(255, 0, 0), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255),
                     (128, 0, 255), (255, 128, 0), (0, 128, 255), (255, 0, 128), (128, 255, 0)]

            for idx, match in enumerate(matches_np[:min(10, len(matches_np))]):
                q_idx, d_idx = int(match[0]), int(match[1])
                if q_idx < len(query_kpts_np) and d_idx < len(db_kpts_np):
                    q_kpt = query_kpts_np[q_idx]
                    d_kpt = db_kpts_np[d_idx]

                    # Use different color for each match (cycling through color list)
                    color = colors[idx % len(colors)]

                    # Draw larger circle for matched point (frame)
                    cv2.circle(frame_np, (int(q_kpt[0]), int(q_kpt[1])), 5, color, 2)

                    # Draw larger circle for matched point (patch)
                    cv2.circle(patch_np, (int(d_kpt[0]), int(d_kpt[1])), 5, color, 2)

                    # Draw correspondence line connecting to edge (simulating connection)
                    # For visual clarity, draw colored border around matched points
                    cv2.rectangle(frame_np,
                                 (int(q_kpt[0])-6, int(q_kpt[1])-6),
                                 (int(q_kpt[0])+6, int(q_kpt[1])+6),
                                 color, 1)
                    cv2.rectangle(patch_np,
                                 (int(d_kpt[0])-6, int(d_kpt[1])-6),
                                 (int(d_kpt[0])+6, int(d_kpt[1])+6),
                                 color, 1)

        return frame_np, patch_np, num_matches

    def compute_homography_warp(self, patch_resized, query_kpts, query_kpts_np, db_kpts_np, matches_np, display_size=300):
        """Compute homography from matches and warp patch image.

        Note: Keypoints are extracted at 512x512 but patch is displayed at display_size x display_size.
        We need to scale keypoints to match the display resolution.
        """
        if matches_np is None or len(matches_np) < 4:
            return patch_resized, None  # Need at least 4 matches for homography

        try:
            # Scale factor: keypoints extracted at 512x512, displayed at display_size
            kpt_size = 512
            scale = display_size / kpt_size

            # Get matched point coordinates and scale them
            matched_query_pts = []
            matched_db_pts = []

            for match in matches_np:
                q_idx, d_idx = int(match[0]), int(match[1])
                if q_idx < len(query_kpts_np) and d_idx < len(db_kpts_np):
                    # Scale keypoints from 512x512 to display_size x display_size
                    q_pt = query_kpts_np[q_idx] * scale
                    d_pt = db_kpts_np[d_idx] * scale
                    matched_query_pts.append(q_pt)
                    matched_db_pts.append(d_pt)

            if len(matched_query_pts) < 4:
                return patch_resized, None

            matched_query_pts = np.array(matched_query_pts, dtype=np.float32)
            matched_db_pts = np.array(matched_db_pts, dtype=np.float32)

            # Compute homography using RANSAC
            H, mask = cv2.findHomography(matched_db_pts, matched_query_pts, cv2.RANSAC, 5.0)

            if H is None:
                return patch_resized, H

            # Warp patch image to align with frame perspective
            warped_patch = cv2.warpPerspective(patch_resized, H,
                                               (patch_resized.shape[1], patch_resized.shape[0]))

            return warped_patch, H

        except Exception as e:
            print(f"⚠️  Homography computation failed: {e}")
            return patch_resized, None

    def update_match_display(self, top_matches):
        """Update display with top 3 matches with keypoints and connecting lines."""
        if not top_matches or len(top_matches) == 0:
            return

        # Process up to 3 matches
        num_matches_to_show = min(3, len(top_matches))
        match_images = []

        for i in range(num_matches_to_show):
            match_img = self._create_single_match_display(top_matches[i], rank=i+1)
            if match_img is not None:
                match_images.append(match_img)

        if not match_images:
            return

        # Combine all match images horizontally
        total_width = sum(img.width for img in match_images)
        max_height = max(img.height for img in match_images)

        combined = Image.new('RGB', (total_width, max_height), color='black')
        x_offset = 0
        for img in match_images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(combined)
        self.match_label.config(image=photo)
        self.match_label.image = photo

    def _create_single_match_display(self, best_match, rank=1):
        """Create display for a single match with keypoints and connecting lines."""
        patch_name = best_match['patch_name']
        patch_idx = best_match.get('idx', self.patch_names.index(patch_name) if patch_name in self.patch_names else 0)
        patch_path = os.path.join(self.config['patches']['output_dir'],
                                  f"{patch_name}.png")

        if not os.path.exists(patch_path):
            return None

        # Get current video frame
        if self.current_frame is None:
            return None

        # Load patch
        patch_cv = cv2.imread(patch_path)
        if patch_cv is None:
            return None

        # Resize to same size for display
        display_size = 300
        frame_resized = cv2.resize(self.current_frame, (display_size, display_size))
        patch_resized = cv2.resize(patch_cv, (display_size, display_size))

        # Initialize homography as None (will be computed if matches are found)
        H = None

        # Get keypoints and matches info
        if self.matcher_type is None:
            # Fallback without keypoints
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            patch_rgb = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            patch_pil = Image.fromarray(patch_rgb)
            combined = Image.new('RGB', (620, 340), color='black')
            combined.paste(frame_pil, (10, 10))
            combined.paste(patch_pil, (320, 10))
            num_matches = 0

        elif self.matcher_type == "superpoint_lightglue":
            # SuperPoint + LightGlue matching
            frame_pil_kpt = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            patch_pil_kpt = Image.fromarray(cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB))

            query_kpts, query_desc = self.extract_keypoints(frame_pil_kpt)
            db_kpts, db_desc = self.superpoint_kpts[patch_idx], self.superpoint_descs[patch_idx]

            if query_kpts is not None and db_kpts is not None:
                # Get matches
                with torch.no_grad():
                    matches = self.lightglue({
                        "image0": {"keypoints": query_kpts, "descriptors": query_desc},
                        "image1": {"keypoints": db_kpts, "descriptors": db_desc}
                    })

                matches_idx = matches["matches"]
                match_scores = matches.get("scores", None)  # Get confidence scores

                if isinstance(matches_idx, list):
                    num_matches = len(matches_idx)
                    matches_tensor = matches_idx[0] if num_matches > 0 else None
                    scores_tensor = match_scores[0] if match_scores is not None and isinstance(match_scores, list) and len(match_scores) > 0 else match_scores
                else:
                    num_matches = matches_idx.shape[0]
                    matches_tensor = matches_idx
                    scores_tensor = match_scores

                # Get numpy arrays
                query_kpts_np = query_kpts[0].cpu().numpy()
                db_kpts_np = db_kpts[0].cpu().numpy()
                matches_np = matches_tensor.cpu().numpy() if matches_tensor is not None and num_matches > 0 else None
                scores_np = scores_tensor.cpu().numpy() if scores_tensor is not None and num_matches > 0 else None

        elif self.matcher_type == "efficientloftr":
            # EfficientLoFTR matching (native API)
            frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            patch_pil_orig = self.patch_images[patch_idx]

            if patch_pil_orig is not None and self.efficientloftr_matcher is not None:
                # Convert patch to grayscale CV2
                patch_cv_orig = cv2.cvtColor(np.array(patch_pil_orig), cv2.COLOR_RGB2GRAY)

                # Use MatchAnything preprocessing if enabled
                use_ma_config = self.config.get('geometric_verification', {}).get('efficientloftr_use_matchanything_config', False)

                if use_ma_config:
                    # MatchAnything preprocessing: proper resize/pad with scale tracking
                    frame_proc, (h_scale0, w_scale0), mask0 = resize_and_pad(frame_gray, df=32, padding=True)
                    patch_proc, (h_scale1, w_scale1), mask1 = resize_and_pad(patch_cv_orig, df=32, padding=True)
                    frame_gray, patch_cv_orig = frame_proc, patch_proc
                    # Store original sizes for keypoint scaling
                    self._frame_orig_size = frame_resized.shape[:2]
                    self._patch_orig_size = np.array(patch_pil_orig).shape[:2]
                else:
                    # Legacy: simple resize to multiples of 32
                    h0, w0 = frame_gray.shape
                    h1, w1 = patch_cv_orig.shape
                    frame_gray = cv2.resize(frame_gray, (w0//32*32, h0//32*32))
                    patch_cv_orig = cv2.resize(patch_cv_orig, (w1//32*32, h1//32*32))
                    self._frame_orig_size = None
                    self._patch_orig_size = None

                # Convert to tensors
                img0_tensor = torch.from_numpy(frame_gray)[None][None].cuda() / 255.
                img1_tensor = torch.from_numpy(patch_cv_orig)[None][None].cuda() / 255.

                # Create batch
                batch = {'image0': img0_tensor, 'image1': img1_tensor}

                # Match with EfficientLoFTR
                with torch.no_grad():
                    self.efficientloftr_matcher(batch)

                # Extract matches
                mkpts0 = batch['mkpts0_f'].cpu().numpy()  # Frame keypoints (Nx2)
                mkpts1 = batch['mkpts1_f'].cpu().numpy()  # Patch keypoints (Nx2)
                mconf = batch['mconf'].cpu().numpy()      # Match confidence (N,)

                # Post-filtering: Apply confidence threshold and top-K filtering
                min_conf = self.config.get('geometric_verification', {}).get('efficientloftr_min_confidence', 0.0)
                top_k = self.config.get('geometric_verification', {}).get('efficientloftr_top_k', 0)

                if len(mconf) > 0 and (min_conf > 0 or top_k > 0):
                    # Filter by minimum confidence
                    if min_conf > 0:
                        valid_mask = mconf > min_conf
                        mkpts0 = mkpts0[valid_mask]
                        mkpts1 = mkpts1[valid_mask]
                        mconf = mconf[valid_mask]

                    # Keep only top-K matches by confidence
                    if top_k > 0 and len(mconf) > top_k:
                        top_indices = np.argsort(mconf)[-top_k:]  # Get indices of top-K
                        mkpts0 = mkpts0[top_indices]
                        mkpts1 = mkpts1[top_indices]
                        mconf = mconf[top_indices]

                # Check if any matches were found after filtering
                if len(mconf) > 0:
                    # Post-process confidence scores for 'opt' model
                    scores_np = (mconf - min(20.0, mconf.min())) / (max(30.0, mconf.max()) - min(20.0, mconf.min()))
                    num_matches = len(scores_np)

                    # Store for visualization
                    query_kpts_np = mkpts0  # Frame keypoints
                    db_kpts_np = mkpts1     # Patch keypoints

                    # Create matches array (each match is [query_idx, db_idx])
                    matches_np = np.column_stack([np.arange(num_matches), np.arange(num_matches)])
                else:
                    # No matches found
                    query_kpts_np = np.array([])
                    db_kpts_np = np.array([])
                    matches_np = None
                    scores_np = None
                    num_matches = 0
            else:
                query_kpts_np = np.array([])
                db_kpts_np = np.array([])
                matches_np = None
                scores_np = None
                num_matches = 0

        else:
            # Unknown matcher
            query_kpts_np = np.array([])
            db_kpts_np = np.array([])
            matches_np = None
            scores_np = None
            num_matches = 0

        # Continue with visualization (same for all matchers)
        if num_matches > 0 and matches_np is not None:

                # Compute homography and warp patch
                patch_warped = patch_resized.copy()

                # Scale keypoints based on matcher type
                if self.matcher_type == "superpoint_lightglue":
                    # SuperPoint extracts at 512x512, need to scale to display_size
                    kpt_size = 512
                    scale = display_size / kpt_size
                    query_kpts_scaled = query_kpts_np * scale
                    db_kpts_scaled = db_kpts_np * scale
                elif self.matcher_type == "efficientloftr":
                    # EfficientLoFTR returns keypoints in original image coordinates
                    # Frame was resized to display_size, patch might be different size
                    frame_scale = display_size / frame_resized.shape[1]
                    patch_scale = display_size / patch_resized.shape[1]
                    query_kpts_scaled = query_kpts_np * frame_scale
                    db_kpts_scaled = db_kpts_np * patch_scale
                else:
                    query_kpts_scaled = query_kpts_np
                    db_kpts_scaled = db_kpts_np

                # Compute homography if enough matches
                if matches_np is not None and num_matches >= 4:
                    # Extract matched points for homography
                    matched_query_pts = []
                    matched_db_pts = []
                    for match in matches_np:
                        q_idx, d_idx = int(match[0]), int(match[1])
                        if q_idx < len(query_kpts_scaled) and d_idx < len(db_kpts_scaled):
                            matched_query_pts.append(query_kpts_scaled[q_idx])
                            matched_db_pts.append(db_kpts_scaled[d_idx])

                    if len(matched_query_pts) >= 4:
                        matched_query_pts = np.array(matched_query_pts, dtype=np.float32)
                        matched_db_pts = np.array(matched_db_pts, dtype=np.float32)
                        H, mask = cv2.findHomography(matched_db_pts, matched_query_pts, cv2.RANSAC, 5.0)

                        if H is not None and mask is not None:
                            # Check homography quality
                            num_inliers = int(mask.sum())
                            inlier_ratio = num_inliers / len(matched_query_pts)

                            # Only use homography if we have enough inliers (at least 30% or 8+ points)
                            if num_inliers >= max(8, int(0.3 * len(matched_query_pts))):
                                # Additional check: verify homography doesn't produce extreme distortions
                                try:
                                    # Test corner warping to detect degenerate homographies
                                    h, w = patch_resized.shape[:2]
                                    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
                                    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

                                    # Check if warped corners are reasonable (within 2x image dimensions)
                                    if (np.abs(warped_corners).max() < 2 * max(h, w) and
                                        not np.isnan(warped_corners).any() and
                                        not np.isinf(warped_corners).any()):
                                        patch_warped = cv2.warpPerspective(patch_resized, H,
                                                                           (patch_resized.shape[1], patch_resized.shape[0]))
                                    else:
                                        # Degenerate homography - don't use it
                                        H = None
                                except:
                                    # Error in warping - don't use homography
                                    H = None
                            else:
                                # Not enough inliers - don't use homography
                                H = None

                # If we have homography, transform the patch keypoints
                if H is not None:
                    # Transform db keypoints using homography
                    db_kpts_homogeneous = np.concatenate([db_kpts_scaled, np.ones((len(db_kpts_scaled), 1))], axis=1)
                    db_kpts_warped = (H @ db_kpts_homogeneous.T).T
                    db_kpts_warped = db_kpts_warped[:, :2] / db_kpts_warped[:, 2:3]
                else:
                    db_kpts_warped = db_kpts_scaled

                # Create combined canvas to draw matching lines
                combined_cv = np.zeros((display_size, display_size * 2, 3), dtype=np.uint8)
                combined_cv[:, :display_size] = frame_resized
                combined_cv[:, display_size:] = patch_warped  # Use warped patch

                # Draw all keypoints first (green)
                for kpt in query_kpts_scaled:
                    cv2.circle(combined_cv, (int(kpt[0]), int(kpt[1])), 3, (0, 255, 0), -1)

                for kpt in db_kpts_warped:
                    cv2.circle(combined_cv, (int(kpt[0]) + display_size, int(kpt[1])), 3, (0, 255, 0), -1)

                # Draw matching lines with colors (ALL matches, colored by confidence)
                if matches_np is not None and num_matches > 0:
                    for idx, match in enumerate(matches_np):  # Draw ALL matches
                        q_idx, d_idx = int(match[0]), int(match[1])
                        if q_idx < len(query_kpts_scaled) and d_idx < len(db_kpts_warped):
                            q_kpt = query_kpts_scaled[q_idx]
                            d_kpt = db_kpts_warped[d_idx]

                            # Color by confidence: green (high) -> yellow -> red (low)
                            if scores_np is not None and idx < len(scores_np):
                                score = scores_np[idx]
                                # High confidence (>0.8): green
                                # Medium confidence (0.5-0.8): yellow
                                # Low confidence (<0.5): red
                                if score > 0.8:
                                    color = (0, 255, 0)  # Green
                                elif score > 0.5:
                                    # Interpolate between green and yellow
                                    t = (score - 0.5) / 0.3
                                    color = (0, 255, int(255 * (1 - t)))
                                else:
                                    # Interpolate between yellow and red
                                    t = score / 0.5
                                    color = (0, int(255 * t), 255)
                            else:
                                color = (255, 255, 255)  # White if no score

                            # Draw line connecting the two points
                            cv2.line(combined_cv,
                                    (int(q_kpt[0]), int(q_kpt[1])),
                                    (int(d_kpt[0]) + display_size, int(d_kpt[1])),
                                    color, 1)

                            # Draw larger circles at both ends
                            cv2.circle(combined_cv, (int(q_kpt[0]), int(q_kpt[1])), 4, color, -1)
                            cv2.circle(combined_cv, (int(d_kpt[0]) + display_size, int(d_kpt[1])), 4, color, -1)

                # Convert to PIL
                combined_rgb = cv2.cvtColor(combined_cv, cv2.COLOR_BGR2RGB)
                combined = Image.fromarray(combined_rgb)
        else:
            # No matches - show fallback visualization
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            patch_rgb = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            patch_pil = Image.fromarray(patch_rgb)
            combined = Image.new('RGB', (620, 340), color='black')
            combined.paste(frame_pil, (10, 10))
            combined.paste(patch_pil, (320, 10))

        # Add labels
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            small_font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            small_font = font

        # Labels
        draw.text((10, 315), "Video Frame", fill=(0, 255, 0), font=font)

        # Indicate if patch is warped or original
        warp_status = "WARPED" if H is not None else "ORIGINAL"
        warp_color = (0, 255, 255) if H is not None else (255, 200, 0)

        # Add rank indicator
        rank_colors = [(255, 215, 0), (192, 192, 192), (205, 127, 50)]  # Gold, Silver, Bronze
        rank_color = rank_colors[rank-1] if rank <= 3 else (255, 255, 255)
        draw.text((10, 5), f"#{rank}", fill=rank_color, font=font)

        draw.text((320, 315), f"{patch_name} [{warp_status}]", fill=warp_color, font=small_font)

        # Info
        conf_text = f"Conf: {best_match['confidence']:.3f} | Pts: {num_matches}"
        draw.text((10, 330), conf_text, fill=(255, 255, 255), font=small_font)

        # Return the image instead of setting it
        return combined

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

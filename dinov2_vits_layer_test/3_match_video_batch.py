#!/usr/bin/env python3
"""
Step 3 (Batch): Match Video Against All Configurations Automatically

Non-interactive version that tests all layer/facet combinations automatically.
Processes video frames and saves results for all configurations.
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
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add parent directory to path for utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities import DinoV2ExtractFeatures, VLAD, seed_everything


class VideoBatchMatcher:
    """Batch video matcher for all configurations."""

    def __init__(self, config):
        self.config = config
        self.video_cap = None
        self.gps_metadata = {}

        # Load GPS metadata
        self.load_gps_metadata()

        # Load video
        self.load_video()

        # Get all configurations
        self.configurations = self.get_configurations()

    def load_gps_metadata(self):
        """Load GPS metadata for patches."""
        metadata_path = os.path.join(self.config['patches']['output_dir'], 'gps_metadata.json')
        with open(metadata_path, 'r') as f:
            self.gps_metadata = json.load(f)

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

    def load_video(self):
        """Load video file."""
        video_path = os.path.join("..", self.config['video']['path'])

        if not os.path.exists(video_path):
            print(f"⚠️  Video not found: {video_path}")
            return

        self.video_cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"✓ Video loaded: {self.total_frames} frames")
        print(f"  Processing every {self.config['video']['frame_sampling']}th frame")

    def load_configuration(self, config_name):
        """Load a specific layer/facet configuration."""

        # Parse configuration name
        parts = config_name.split('_')
        layer = int(parts[1])
        facet = parts[2]

        print(f"\n📥 Loading: Layer {layer}, Facet '{facet}'")

        # Initialize extractor
        extractor = DinoV2ExtractFeatures(
            dino_model=self.config['dinov2']['model'],
            layer=layer,
            facet=facet,
            use_cls=False,
            norm_descs=True,
            device=self.config['dinov2']['device']
        )

        # Load VLAD descriptors
        desc_dir = f"data/descriptors/{config_name}"
        db_vlads = torch.load(os.path.join(desc_dir, "vlad_descriptors.pt"))

        # Normalize and move to device
        db_vlads = F.normalize(db_vlads, p=2, dim=-1)
        db_vlads = db_vlads.to(self.config['dinov2']['device'])

        # Load patch names
        with open(os.path.join(desc_dir, "patch_names.txt"), 'r') as f:
            patch_names = [line.strip() for line in f.readlines()]

        # Load VLAD
        vlad = VLAD(
            num_clusters=self.config['vlad']['num_clusters'],
            desc_dim=None,
            cache_dir=desc_dir
        )
        vlad.fit(None)

        return extractor, vlad, db_vlads, patch_names, layer, facet

    def process_frame(self, frame, extractor, vlad, db_vlads, patch_names):
        """Process a single frame."""

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
            descriptors = extractor(frame_tensor)

            # Generate VLAD descriptor (move to CPU)
            query_vlad = vlad.generate_multi(descriptors.cpu())
            query_vlad = F.normalize(query_vlad, p=2, dim=-1)

            # Move back to device for matching
            query_vlad = query_vlad.to(self.config['dinov2']['device'])

            # Match against database
            similarities = torch.mm(query_vlad, db_vlads.T)
            top_k_vals, top_k_idx = torch.topk(similarities[0], k=self.config['video']['top_k_matches'])

        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        # Get results
        top_matches = []
        for i in range(len(top_k_idx)):
            idx = top_k_idx[i].item()
            confidence = top_k_vals[i].item()
            patch_name = patch_names[idx]

            top_matches.append({
                "patch_name": patch_name,
                "confidence": confidence,
                "gps": self.gps_metadata.get(patch_name, {})
            })

        return top_matches, fps, processing_time

    def process_configuration(self, config_name):
        """Process entire video for one configuration."""

        # Load configuration
        extractor, vlad, db_vlads, patch_names, layer, facet = self.load_configuration(config_name)

        # Reset video to beginning
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        results = {
            "layer": layer,
            "facet": facet,
            "matches": []
        }

        frame_idx = 0
        frames_processed = 0
        fps_list = []

        # Calculate total frames to process
        total_to_process = self.total_frames // self.config['video']['frame_sampling']

        pbar = tqdm(total=total_to_process, desc=f"  Layer {layer} {facet}", unit="frames")

        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break

            # Process only sampled frames
            if frame_idx % self.config['video']['frame_sampling'] == 0:
                top_matches, fps, proc_time = self.process_frame(frame, extractor, vlad, db_vlads, patch_names)
                fps_list.append(fps)

                # Save result
                timestamp = frame_idx / 30.0  # Assume 30 FPS
                results["matches"].append({
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "top_match": top_matches[0]["patch_name"],
                    "confidence": top_matches[0]["confidence"],
                    "gps": top_matches[0]["gps"],
                    "top_5": [m["patch_name"] for m in top_matches],
                    "fps": fps
                })

                frames_processed += 1
                pbar.update(1)

            frame_idx += 1

        pbar.close()

        # Calculate average FPS and confidence
        results["avg_fps"] = np.mean(fps_list) if fps_list else 0
        results["avg_confidence"] = np.mean([m["confidence"] for m in results["matches"]]) if results["matches"] else 0

        print(f"  ✓ Processed {frames_processed} frames | Avg FPS: {results['avg_fps']:.1f} | Avg Conf: {results['avg_confidence']:.3f}")

        return results

    def process_all(self):
        """Process video for all configurations."""

        print(f"\n🎥 Processing video for all {len(self.configurations)} configurations...")
        print("=" * 70)

        all_results = {
            "video": self.config['video']['path'],
            "total_frames": self.total_frames,
            "frame_sampling": self.config['video']['frame_sampling'],
            "configurations": {}
        }

        for i, config_name in enumerate(self.configurations, 1):
            print(f"\n[{i}/{len(self.configurations)}] Configuration: {config_name}")

            results = self.process_configuration(config_name)
            all_results["configurations"][config_name] = results

        # Save results
        output_path = "data/results/flight_100m_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        for config_name, results in all_results["configurations"].items():
            print(f"{config_name:20} | Frames: {len(results['matches']):4} | "
                  f"Avg FPS: {results['avg_fps']:5.1f} | Avg Conf: {results['avg_confidence']:.3f}")

        print("=" * 70)


def main():
    """Main execution function."""

    # Set seed
    seed_everything(42)

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("🎥 Batch Video Matcher - Testing All Configurations")
    print("=" * 70)

    # Create matcher
    matcher = VideoBatchMatcher(config)

    print(f"\nConfigurations to test: {len(matcher.configurations)}")
    for cfg in matcher.configurations:
        print(f"  • {cfg}")

    # Process all configurations
    matcher.process_all()

    print("\n✅ Batch processing complete!")
    print("\nNext step: Run analysis")
    print("  python 4_analyze_results.py")


if __name__ == "__main__":
    main()

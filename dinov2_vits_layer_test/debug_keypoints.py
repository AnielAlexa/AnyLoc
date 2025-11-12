#!/usr/bin/env python3
"""
Debug script to check SuperPoint keypoints and LightGlue matches.

Shows:
- Keypoints detected in video frame
- Keypoints detected in satellite patch
- Number of matches found by LightGlue
"""

import os
import sys
from pathlib import Path
import yaml
import cv2
import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities import DinoV2ExtractFeatures, VLAD, seed_everything

try:
    from lightglue import SuperPoint, LightGlue
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    print("❌ LightGlue not available")
    exit(1)


def main():
    """Debug keypoint extraction and matching."""

    seed_everything(42)

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = config['dinov2']['device']

    # Load video frame
    video_path = os.path.join("..", config['video']['path'])
    video_cap = cv2.VideoCapture(video_path)

    # Get frame 100
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = video_cap.read()
    video_cap.release()

    if not ret:
        print("❌ Could not read video frame")
        return

    # Load satellite patch
    patches_dir = config['patches']['output_dir']
    patch_path = os.path.join(patches_dir, "patch_0_0.png")
    patch = cv2.imread(patch_path)

    if patch is None:
        print(f"❌ Could not load patch: {patch_path}")
        return

    # Resize to reasonable size
    frame_small = cv2.resize(frame, (512, 512))
    patch_small = cv2.resize(patch, (512, 512))

    print(f"📊 Frame size: {frame_small.shape}")
    print(f"📊 Patch size: {patch_small.shape}")

    # Initialize SuperPoint
    print(f"\n🔍 Initializing SuperPoint...")
    superpoint = SuperPoint(max_num_keypoints=2048).eval().to(device)

    # Extract keypoints
    print(f"\n📍 Extracting keypoints from frame...")
    transform = T.Compose([T.ToTensor()])

    frame_pil = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
    patch_pil = Image.fromarray(cv2.cvtColor(patch_small, cv2.COLOR_BGR2RGB))

    frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
    patch_tensor = transform(patch_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        frame_features = superpoint({"image": frame_tensor})
        patch_features = superpoint({"image": patch_tensor})

    frame_kpts = frame_features.get("keypoints")
    frame_desc = frame_features.get("descriptors")
    patch_kpts = patch_features.get("keypoints")
    patch_desc = patch_features.get("descriptors")

    print(f"   Frame keypoints: {frame_kpts.shape if frame_kpts is not None else 'None'}")
    print(f"   Frame descriptors: {frame_desc.shape if frame_desc is not None else 'None'}")
    print(f"   Patch keypoints: {patch_kpts.shape if patch_kpts is not None else 'None'}")
    print(f"   Patch descriptors: {patch_desc.shape if patch_desc is not None else 'None'}")

    # Try LightGlue matching
    print(f"\n⚡ Initializing LightGlue...")
    lightglue = LightGlue(features="superpoint").eval().to(device)

    print(f"\n🔗 Attempting to match keypoints...")
    with torch.no_grad():
        matches = lightglue({
            "image0": {"keypoints": frame_kpts, "descriptors": frame_desc},
            "image1": {"keypoints": patch_kpts, "descriptors": patch_desc}
        })

    matches_idx = matches["matches"]

    if isinstance(matches_idx, list):
        num_matches = len(matches_idx)
    else:
        num_matches = matches_idx.shape[0]

    print(f"   Matches found: {num_matches}")

    # Visualize
    if num_matches > 0:
        print(f"\n✅ {num_matches} matches found! Drawing visualization...")

        # Create combined image
        combined = np.zeros((512, 1024, 3), dtype=np.uint8)
        combined[:, :512] = frame_small
        combined[:, 512:] = patch_small

        frame_kpts_np = frame_kpts[0].cpu().numpy()
        patch_kpts_np = patch_kpts[0].cpu().numpy()

        # Draw all keypoints
        for kpt in frame_kpts_np:
            cv2.circle(combined, (int(kpt[0]), int(kpt[1])), 3, (0, 255, 0), -1)

        for kpt in patch_kpts_np:
            cv2.circle(combined, (int(kpt[0]) + 512, int(kpt[1])), 3, (0, 255, 0), -1)

            # Draw matches
        if isinstance(matches_idx, list):
            matches_tensor = matches_idx[0]  # Get the tensor from list
        else:
            matches_tensor = matches_idx

        matches_np = matches_tensor.cpu().numpy()

        colors = [(255, 0, 0), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255)]

        for idx, match in enumerate(matches_np[:10]):
            q_idx, d_idx = int(match[0]), int(match[1])
            q_kpt = frame_kpts_np[q_idx]
            d_kpt = patch_kpts_np[d_idx]

            color = colors[idx % len(colors)]

            # Draw line
            cv2.line(combined,
                    (int(q_kpt[0]), int(q_kpt[1])),
                    (int(d_kpt[0]) + 512, int(d_kpt[1])),
                    color, 2)

            # Draw circles
            cv2.circle(combined, (int(q_kpt[0]), int(q_kpt[1])), 5, color, 2)
            cv2.circle(combined, (int(d_kpt[0]) + 512, int(d_kpt[1])), 5, color, 2)

        # Save
        output_path = "debug_matches.png"
        cv2.imwrite(output_path, combined)
        print(f"   ✓ Saved to: {output_path}")

        # Also show in terminal (if possible)
        print(f"\n📸 Visualization saved. Open 'debug_matches.png' to see matches.")

    else:
        print(f"\n⚠️  NO MATCHES FOUND!")
        print(f"\n   Possible reasons:")
        print(f"   1. Video frame and satellite patch are too different")
        print(f"   2. Different lighting conditions (day/night)")
        print(f"   3. Different viewpoints (drone view vs. satellite)")
        print(f"   4. LightGlue confidence threshold too high")
        print(f"\n   Still visualizing all keypoints...")

        # Create combined image with just keypoints
        combined = np.zeros((512, 1024, 3), dtype=np.uint8)
        combined[:, :512] = frame_small
        combined[:, 512:] = patch_small

        frame_kpts_np = frame_kpts[0].cpu().numpy()
        patch_kpts_np = patch_kpts[0].cpu().numpy()

        # Draw all keypoints
        for kpt in frame_kpts_np:
            cv2.circle(combined, (int(kpt[0]), int(kpt[1])), 3, (0, 255, 0), -1)

        for kpt in patch_kpts_np:
            cv2.circle(combined, (int(kpt[0]) + 512, int(kpt[1])), 3, (0, 255, 0), -1)

        # Save
        output_path = "debug_keypoints.png"
        cv2.imwrite(output_path, combined)
        print(f"   ✓ Saved keypoints to: {output_path}")

    print(f"\n📊 Summary:")
    print(f"   Frame keypoints detected: {frame_kpts.shape[1] if frame_kpts is not None else 0}")
    print(f"   Patch keypoints detected: {patch_kpts.shape[1] if patch_kpts is not None else 0}")
    print(f"   LightGlue matches: {num_matches}")


if __name__ == "__main__":
    main()

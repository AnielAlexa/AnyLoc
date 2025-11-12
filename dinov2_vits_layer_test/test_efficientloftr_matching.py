#!/usr/bin/env python3
"""
Test EfficientLoFTR matching on a sample frame and patch pair.
This helps diagnose why there are few matches.
"""

import sys
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image

# Add EfficientLoFTR to path
sys.path.insert(0, str(Path(__file__).parent / "EfficientLoFTR"))
from src.loftr import LoFTR, opt_default_cfg, reparameter
from copy import deepcopy

def test_efficientloftr():
    print("🔬 Testing EfficientLoFTR matching...")

    # Load model
    print("Loading EfficientLoFTR-opt...")
    _default_cfg = deepcopy(opt_default_cfg)
    matcher = LoFTR(config=_default_cfg)

    weights_path = Path(__file__).parent / "EfficientLoFTR" / "weights" / "eloftr_outdoor.ckpt"
    state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
    matcher.load_state_dict(state_dict)
    matcher = reparameter(matcher)
    matcher = matcher.eval().cuda()
    print("✓ Model loaded")

    # Load a sample video frame
    video_path = Path(__file__).parent.parent / "flight_100m.mp4"
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # Skip to frame 100
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Could not load video frame")
        return

    print(f"✓ Loaded frame: {frame.shape}")

    # Load a sample patch
    patch_dir = Path(__file__).parent / "data" / "patches"
    patch_files = sorted(patch_dir.glob("*.png"))

    if not patch_files:
        print("❌ No patches found")
        return

    patch_path = patch_files[0]
    patch = Image.open(patch_path).convert('RGB')
    patch_cv = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)

    print(f"✓ Loaded patch: {patch_cv.shape} from {patch_path.name}")

    # Test 1: Original sizes
    print("\n📊 Test 1: Original image sizes")
    test_match(matcher, frame, patch_cv, "Original")

    # Test 2: Resize frame to match patch size
    print("\n📊 Test 2: Resize frame to match patch")
    frame_resized = cv2.resize(frame, (patch_cv.shape[1], patch_cv.shape[0]))
    test_match(matcher, frame_resized, patch_cv, "Same size")

    # Test 3: Both at 512x512
    print("\n📊 Test 3: Both at 512x512")
    frame_512 = cv2.resize(frame, (512, 512))
    patch_512 = cv2.resize(patch_cv, (512, 512))
    test_match(matcher, frame_512, patch_512, "512x512")

    # Test 4: Both at 256x256
    print("\n📊 Test 4: Both at 256x256")
    frame_256 = cv2.resize(frame, (256, 256))
    patch_256 = cv2.resize(patch_cv, (256, 256))
    test_match(matcher, frame_256, patch_256, "256x256")

    print("\n✅ Testing complete!")
    print("\n💡 Recommendations:")
    print("   - If matches are very low (<10), images may not overlap")
    print("   - EfficientLoFTR works best on images with significant visual overlap")
    print("   - Consider using different test frames/patches with more overlap")
    print("   - Outdoor weights are optimized for outdoor scenes with texture")

def test_match(matcher, img0, img1, label):
    """Run matching and report statistics."""
    # Convert to grayscale
    if len(img0.shape) == 3:
        img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    else:
        img0_gray = img0

    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1

    # Resize to multiples of 32
    h0, w0 = img0_gray.shape
    h1, w1 = img1_gray.shape
    img0_gray = cv2.resize(img0_gray, (w0//32*32, h0//32*32))
    img1_gray = cv2.resize(img1_gray, (w1//32*32, h1//32*32))

    # Convert to tensors
    img0_tensor = torch.from_numpy(img0_gray)[None][None].cuda() / 255.
    img1_tensor = torch.from_numpy(img1_gray)[None][None].cuda() / 255.

    batch = {'image0': img0_tensor, 'image1': img1_tensor}

    # Match
    with torch.no_grad():
        matcher(batch)

    # Extract results
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

    num_matches = len(mconf)

    print(f"   {label}: {num_matches} matches", end="")

    if num_matches > 0:
        avg_conf = mconf.mean()
        max_conf = mconf.max()
        min_conf = mconf.min()
        print(f" | Confidence: avg={avg_conf:.2f}, min={min_conf:.2f}, max={max_conf:.2f}")

        # Count high confidence matches
        high_conf = (mconf > 30.0).sum()
        med_conf = ((mconf > 20.0) & (mconf <= 30.0)).sum()
        low_conf = (mconf <= 20.0).sum()
        print(f"      High (>30): {high_conf}, Med (20-30): {med_conf}, Low (<20): {low_conf}")
    else:
        print(" | No matches found ❌")

if __name__ == "__main__":
    test_efficientloftr()

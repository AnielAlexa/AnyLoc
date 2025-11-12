#!/usr/bin/env python3
"""
Compare EfficientLoFTR variants: outdoor vs matchanything
Tests both models on sample drone frame and satellite patch.
"""

import sys
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image
import time

# Add EfficientLoFTR to path
sys.path.insert(0, str(Path(__file__).parent / "EfficientLoFTR"))
from src.loftr import LoFTR, opt_default_cfg, reparameter
from copy import deepcopy

def load_model(variant='outdoor', threshold=20):
    """Load EfficientLoFTR model with specified weights."""
    print(f"\n{'='*60}")
    print(f"Loading {variant.upper()} model...")
    print(f"{'='*60}")

    # Load config
    _default_cfg = deepcopy(opt_default_cfg)
    _default_cfg['match_coarse']['thr'] = threshold

    # Initialize matcher
    matcher = LoFTR(config=_default_cfg)

    # Load weights
    weights_map = {
        'outdoor': 'eloftr_outdoor.ckpt',
        'matchanything': 'matchanything_eloftr.ckpt'
    }

    weights_path = Path(__file__).parent / "EfficientLoFTR" / "weights" / weights_map[variant]

    if not weights_path.exists():
        print(f"❌ Weights not found: {weights_path}")
        return None

    state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
    matcher.load_state_dict(state_dict)
    matcher = reparameter(matcher)
    matcher = matcher.eval().cuda()

    print(f"✓ Loaded {variant} weights ({weights_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"   Matching threshold: {threshold}")

    return matcher

def match_images(matcher, img0, img1, variant_name):
    """Run matching and return statistics."""
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
    start_time = time.time()
    with torch.no_grad():
        matcher(batch)
    elapsed = (time.time() - start_time) * 1000  # ms

    # Extract results
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

    num_matches = len(mconf)

    results = {
        'variant': variant_name,
        'num_matches': num_matches,
        'time_ms': elapsed
    }

    if num_matches > 0:
        results['confidence'] = {
            'mean': float(mconf.mean()),
            'std': float(mconf.std()),
            'min': float(mconf.min()),
            'max': float(mconf.max())
        }

        # Count by confidence bins
        results['high_conf'] = int((mconf > 30.0).sum())
        results['med_conf'] = int(((mconf > 20.0) & (mconf <= 30.0)).sum())
        results['low_conf'] = int((mconf <= 20.0).sum())

    return results

def print_results(results):
    """Pretty print matching results."""
    print(f"\n📊 {results['variant'].upper()} Results:")
    print(f"   {'─'*50}")
    print(f"   Matches found: {results['num_matches']}")
    print(f"   Processing time: {results['time_ms']:.1f} ms")

    if results['num_matches'] > 0:
        conf = results['confidence']
        print(f"   Confidence: {conf['mean']:.2f} ± {conf['std']:.2f}")
        print(f"               (min: {conf['min']:.2f}, max: {conf['max']:.2f})")
        print(f"   Distribution:")
        print(f"      High (>30):  {results['high_conf']:4d} matches")
        print(f"      Med (20-30): {results['med_conf']:4d} matches")
        print(f"      Low (<20):   {results['low_conf']:4d} matches")
    else:
        print(f"   ❌ No matches found")

def main():
    print("🔬 EfficientLoFTR Variant Comparison")
    print("="*60)

    # Configuration
    THRESHOLD = 20  # Matching threshold

    # Load sample images
    video_path = Path(__file__).parent.parent / "flight_100m.mp4"
    patch_dir = Path(__file__).parent / "data" / "patches"

    # Load video frame
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)  # Frame 500
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Could not read video frame")
        return

    print(f"✓ Loaded video frame: {frame.shape}")

    # Load patch
    patch_files = sorted(patch_dir.glob("*.png"))
    if not patch_files:
        print(f"❌ No patches found in {patch_dir}")
        return

    patch_path = patch_files[len(patch_files)//2]  # Middle patch
    patch = Image.open(patch_path).convert('RGB')
    patch_cv = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)

    print(f"✓ Loaded patch: {patch_cv.shape} from {patch_path.name}")

    # Resize for matching
    frame_resized = cv2.resize(frame, (512, 512))
    patch_resized = cv2.resize(patch_cv, (512, 512))

    # Test both variants
    variants = ['outdoor', 'matchanything']
    results = []

    for variant in variants:
        matcher = load_model(variant, threshold=THRESHOLD)
        if matcher is None:
            continue

        result = match_images(matcher, frame_resized, patch_resized, variant)
        results.append(result)
        print_results(result)

        # Cleanup
        del matcher
        torch.cuda.empty_cache()

    # Comparison summary
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("📈 Comparison Summary")
        print(f"{'='*60}")

        outdoor = results[0]
        matchanything = results[1]

        print(f"\n   {'Metric':<20} {'Outdoor':<15} {'MatchAnything':<15} {'Diff'}")
        print(f"   {'-'*65}")

        # Matches
        diff_matches = matchanything['num_matches'] - outdoor['num_matches']
        pct_matches = (diff_matches / outdoor['num_matches'] * 100) if outdoor['num_matches'] > 0 else 0
        print(f"   {'Matches':<20} {outdoor['num_matches']:<15} {matchanything['num_matches']:<15} {diff_matches:+d} ({pct_matches:+.1f}%)")

        # Time
        diff_time = matchanything['time_ms'] - outdoor['time_ms']
        pct_time = (diff_time / outdoor['time_ms'] * 100) if outdoor['time_ms'] > 0 else 0
        print(f"   {'Time (ms)':<20} {outdoor['time_ms']:<15.1f} {matchanything['time_ms']:<15.1f} {diff_time:+.1f} ({pct_time:+.1f}%)")

        # Confidence (if both have matches)
        if outdoor['num_matches'] > 0 and matchanything['num_matches'] > 0:
            diff_conf = matchanything['confidence']['mean'] - outdoor['confidence']['mean']
            print(f"   {'Mean Confidence':<20} {outdoor['confidence']['mean']:<15.2f} {matchanything['confidence']['mean']:<15.2f} {diff_conf:+.2f}")

        print(f"\n{'='*60}")

        # Recommendation
        if matchanything['num_matches'] > outdoor['num_matches']:
            print("✅ MatchAnything found more matches - Recommended for drone-satellite!")
        elif matchanything['num_matches'] == outdoor['num_matches']:
            print("⚖️  Both models found similar matches - Either variant works well")
        else:
            print("⚠️  Outdoor found more matches - May work better for this specific case")

        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

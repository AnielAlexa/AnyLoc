#!/usr/bin/env python3
"""
Step 2: Compute DINOv2 ViT-S Descriptors and VLAD

Extracts DINOv2 features for all layer/facet combinations and computes VLAD descriptors.
Tests layers 8-11 with key/query/value facets.
"""

import os
import sys
from pathlib import Path
import yaml
import json
import time
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
import einops as ein

# Add parent directory to path for utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities import DinoV2ExtractFeatures, VLAD, seed_everything

try:
    from lightglue import SuperPoint
    SUPERPOINT_AVAILABLE = True
except ImportError:
    SUPERPOINT_AVAILABLE = False
    print("⚠️  LightGlue not available - SuperPoint extraction will be skipped")


def load_patches(patches_dir):
    """Load all patches from directory."""
    patch_files = sorted([f for f in os.listdir(patches_dir) if f.endswith('.png')])
    patches = []
    patch_names = []

    for patch_file in patch_files:
        patch_path = os.path.join(patches_dir, patch_file)
        patch = Image.open(patch_path).convert('RGB')
        patches.append(patch)
        patch_names.append(patch_file.replace('.png', ''))

    return patches, patch_names


def extract_features(patches, extractor, image_size=518):
    """Extract DINOv2 features from patches."""

    # Define transform
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    descriptors = []

    for patch in tqdm(patches, desc="   Extracting features", unit="img"):
        # Transform and add batch dimension
        patch_tensor = transform(patch).unsqueeze(0).to(extractor.device)

        # Extract features
        with torch.no_grad():
            desc = extractor(patch_tensor)  # Shape: (1, num_patches, desc_dim)

        descriptors.append(desc)

    # Concatenate all descriptors
    all_descriptors = torch.cat(descriptors, dim=0)  # (N, num_patches, desc_dim)

    return all_descriptors


def extract_superpoint_features(patches, device="cuda", max_keypoints=2048):
    """Extract SuperPoint keypoints and descriptors from patches."""
    if not SUPERPOINT_AVAILABLE:
        return None, None

    superpoint = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)

    # Define transform for SuperPoint (no normalization, just to tensor)
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])

    all_kpts = []
    all_descs = []

    for patch in tqdm(patches, desc="   Extracting SuperPoint", unit="img"):
        patch_tensor = transform(patch).unsqueeze(0).to(device)

        with torch.no_grad():
            features = superpoint({"image": patch_tensor})

        # Extract keypoints and descriptors
        kpts = features.get("keypoints", torch.tensor([]).to(device))  # (1, N, 2)
        desc = features.get("descriptors", torch.tensor([]).to(device))  # (1, N, 256)

        all_kpts.append(kpts)
        all_descs.append(desc)

    return all_kpts, all_descs


def compute_vlad(descriptors, num_clusters, output_dir, dist_mode="cosine", vlad_mode="hard", intra_norm=True):
    """Compute VLAD descriptors and save cluster centers."""

    N, num_patches, desc_dim = descriptors.shape

    print(f"   Descriptor shape: {tuple(descriptors.shape)}")
    print(f"   Fitting VLAD clusters (K={num_clusters})...")

    # Initialize VLAD
    vlad = VLAD(
        num_clusters=num_clusters,
        desc_dim=desc_dim,
        intra_norm=intra_norm,
        norm_descs=True,
        dist_mode=dist_mode,
        vlad_mode=vlad_mode,
        cache_dir=output_dir
    )

    # Fit VLAD on all descriptors (move to CPU to avoid CUDA tensor issues)
    all_descs = ein.rearrange(descriptors, "n k d -> (n k) d").cpu()
    vlad.fit(all_descs)

    print(f"   Generating VLAD descriptors...")

    # Generate VLAD global descriptors (move to CPU)
    vlad_descriptors = vlad.generate_multi(descriptors.cpu())  # (N, num_clusters * desc_dim)

    print(f"   Output shape: {tuple(vlad_descriptors.shape)}")

    return vlad_descriptors, vlad


def save_descriptors(vlad_descriptors, patch_names, vlad, output_dir, config_info,
                     superpoint_kpts=None, superpoint_descs=None):
    """Save VLAD descriptors, cluster centers, SuperPoint features, and metadata."""

    os.makedirs(output_dir, exist_ok=True)

    # Save VLAD descriptors
    torch.save(vlad_descriptors, os.path.join(output_dir, "vlad_descriptors.pt"))

    # Save VLAD cluster centers (required for loading later)
    torch.save(vlad.c_centers, os.path.join(output_dir, "c_centers.pt"))

    # Save SuperPoint features if available
    if superpoint_kpts is not None and superpoint_descs is not None:
        torch.save(superpoint_kpts, os.path.join(output_dir, "superpoint_kpts.pt"))
        torch.save(superpoint_descs, os.path.join(output_dir, "superpoint_descs.pt"))

    # Save patch names
    with open(os.path.join(output_dir, "patch_names.txt"), 'w') as f:
        f.write('\n'.join(patch_names))

    # Save metadata
    metadata = {
        "num_patches": len(patch_names),
        "vlad_dim": vlad_descriptors.shape[1],
        "num_clusters": vlad.num_clusters,
        "desc_dim": vlad.desc_dim,
        "has_superpoint": superpoint_kpts is not None,
        **config_info
    }

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✓ Saved to {output_dir}/")


def main():
    """Main execution function."""

    # Set seed for reproducibility
    seed_everything(42)

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("🔧 Computing DINOv2 ViT-S descriptors...")
    print(f"   Model: {config['dinov2']['model']}")
    print(f"   Layers: {config['dinov2']['layers_to_test']}")
    print(f"   Facets: {config['dinov2']['facets_to_test']}")

    # Load patches
    patches_dir = config['patches']['output_dir']
    print(f"\nLoading patches from {patches_dir}...")
    patches, patch_names = load_patches(patches_dir)
    print(f"✓ Loaded {len(patches)} patches")

    # Calculate total configurations
    layers = config['dinov2']['layers_to_test']
    facets = config['dinov2']['facets_to_test']
    total_configs = len(layers) * len(facets)

    print(f"   Patches: {len(patches)}")
    print(f"   Configurations: {total_configs}")

    # Extract SuperPoint features ONCE before processing all configurations
    superpoint_kpts = None
    superpoint_descs = None
    if config.get('geometric_verification', {}).get('enabled', True) and SUPERPOINT_AVAILABLE:
        print(f"\n🔍 Extracting SuperPoint features (one-time cost)...")
        superpoint_kpts, superpoint_descs = extract_superpoint_features(
            patches,
            device=config['dinov2']['device'],
            max_keypoints=config['geometric_verification']['max_keypoints']
        )

    # Process each configuration
    config_idx = 0
    total_start_time = time.time()

    for layer in layers:
        for facet in facets:
            config_idx += 1

            print(f"\nConfiguration {config_idx}/{total_configs}: Layer {layer}, Facet \"{facet}\"")

            # Initialize extractor
            extractor = DinoV2ExtractFeatures(
                dino_model=config['dinov2']['model'],
                layer=layer,
                facet=facet,
                use_cls=False,
                norm_descs=True,
                device=config['dinov2']['device']
            )

            config_start_time = time.time()

            # Extract features
            descriptors = extract_features(patches, extractor, config['dinov2']['image_size'])

            # Compute VLAD
            vlad_descriptors, vlad = compute_vlad(
                descriptors,
                config['vlad']['num_clusters'],
                None,  # We'll specify output_dir in save
                dist_mode=config['vlad']['dist_mode'],
                vlad_mode=config['vlad']['vlad_mode'],
                intra_norm=config['vlad']['intra_norm']
            )

            # Save descriptors
            output_dir = f"data/descriptors/layer_{layer}_{facet}"
            config_info = {
                "layer": layer,
                "facet": facet,
                "model": config['dinov2']['model'],
                "image_size": config['dinov2']['image_size']
            }

            save_descriptors(vlad_descriptors, patch_names, vlad, output_dir, config_info,
                           superpoint_kpts, superpoint_descs)

            config_time = time.time() - config_start_time
            print(f"   Time: {config_time:.1f}s")

    total_time = time.time() - total_start_time

    print(f"\nSummary:")
    print(f"   • Configurations processed: {total_configs}")
    print(f"   • Total time: {int(total_time // 60)}m {int(total_time % 60)}s")
    print(f"   • Avg time per config: {total_time / total_configs:.1f}s")

    # Calculate cache size
    cache_size_mb = 0
    for layer in layers:
        for facet in facets:
            output_dir = f"data/descriptors/layer_{layer}_{facet}"
            if os.path.exists(output_dir):
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        cache_size_mb += os.path.getsize(file_path) / (1024 * 1024)

    print(f"   • Cache size: {cache_size_mb:.1f} MB")
    print("\n✅ Descriptor computation complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 1: Download Satellite Imagery from Google Maps and Split into Patches

Downloads high-quality satellite imagery using Google Maps Static API.
Requires API key in config.yaml.

Alternative: Uses tile-based download from Google Maps tile server.
"""

import os
import sys
from pathlib import Path
import yaml
import json
import time
import math
import requests
from PIL import Image
import numpy as np
from io import BytesIO
from tqdm import tqdm

# Add parent directory to path for utilities
sys.path.insert(0, str(Path(__file__).parent.parent))


def latlon_to_tile(lat, lon, zoom):
    """Convert lat/lon to tile coordinates at given zoom level."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_latlon(x, y, zoom):
    """Convert tile coordinates to lat/lon (top-left corner)."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def download_google_tile(x, y, zoom, delay=0.15, api_key=None):
    """
    Download a single tile from Google Maps.

    If api_key is provided, uses Google Maps Static API.
    Otherwise, uses Google Maps tile server (may have usage restrictions).
    """

    if api_key:
        # Use Google Maps Static API (requires API key and billing)
        lat, lon = tile_to_latlon(x, y, zoom)
        url = (f"https://maps.googleapis.com/maps/api/staticmap?"
               f"center={lat},{lon}&zoom={zoom}&size=640x640&"
               f"maptype=satellite&key={api_key}&scale=2")
    else:
        # Use Google Maps tile server (unofficial, may be restricted)
        # Format: https://mt{0-3}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}
        server = x % 4  # Distribute across 4 servers
        url = f"https://mt{server}.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        time.sleep(delay)  # Rate limiting
        return img
    except Exception as e:
        print(f"⚠️  Failed to download tile ({x}, {y}): {e}")
        return None


def download_area(min_lat, max_lat, min_lon, max_lon, zoom, delay=0.15, api_key=None):
    """Download all tiles for the specified area and stitch them together."""

    # Get tile bounds
    min_x, max_y = latlon_to_tile(min_lat, min_lon, zoom)
    max_x, min_y = latlon_to_tile(max_lat, max_lon, zoom)

    # Ensure correct ordering
    if min_x > max_x:
        min_x, max_x = max_x, min_x
    if min_y > max_y:
        min_y, max_y = max_y, min_y

    num_tiles_x = max_x - min_x + 1
    num_tiles_y = max_y - min_y + 1
    total_tiles = num_tiles_x * num_tiles_y

    print(f"   Tiles needed: {num_tiles_x}×{num_tiles_y} = {total_tiles} tiles")

    if api_key:
        print("   Using Google Maps Static API (high quality)")
    else:
        print("   Using Google Maps tile server (free, may have restrictions)")

    # Download tiles
    tiles = {}
    pbar = tqdm(total=total_tiles, desc="Downloading tiles", unit="tiles")

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            tile = download_google_tile(x, y, zoom, delay, api_key)
            if tile is not None:
                tiles[(x, y)] = tile
            pbar.update(1)

    pbar.close()

    if not tiles:
        raise ValueError("Failed to download any tiles!")

    # Stitch tiles together
    tile_size = 256  # Standard tile size (or 640 if using Static API with scale=2)
    if api_key and tiles:
        # API returns different size
        sample_tile = next(iter(tiles.values()))
        tile_size = sample_tile.width

    img_width = num_tiles_x * tile_size
    img_height = num_tiles_y * tile_size

    stitched = Image.new('RGB', (img_width, img_height))

    for (x, y), tile in tiles.items():
        px = (x - min_x) * tile_size
        py = (y - min_y) * tile_size
        # Resize tile if needed
        if tile.size != (tile_size, tile_size):
            tile = tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        stitched.paste(tile, (px, py))

    print(f"✓ Stitched image: {img_width}×{img_height} pixels")

    return stitched, (min_x, min_y, max_x, max_y)


def create_patches(image, patch_size, overlap, output_dir, bounds, zoom):
    """
    Split image into overlapping patches and calculate GPS metadata.

    Args:
        image: PIL Image
        patch_size: Size of each patch (pixels)
        overlap: Overlap ratio (0.0 to 1.0)
        output_dir: Output directory
        bounds: Tile bounds (min_x, min_y, max_x, max_y)
        zoom: Zoom level
    """

    img_width, img_height = image.size
    stride = int(patch_size * (1 - overlap))

    num_patches_x = (img_width - patch_size) // stride + 1
    num_patches_y = (img_height - patch_size) // stride + 1

    print(f"   Grid: {num_patches_x}×{num_patches_y} = {num_patches_x * num_patches_y} patches")

    os.makedirs(output_dir, exist_ok=True)

    gps_metadata = {}
    patches_created = 0

    pbar = tqdm(total=num_patches_x * num_patches_y, desc="Extracting patches", unit="patches")

    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate pixel coordinates
            x = j * stride
            y = i * stride

            # Extract patch
            patch = image.crop((x, y, x + patch_size, y + patch_size))

            # Save patch
            patch_name = f"patch_{i}_{j}"
            patch_path = os.path.join(output_dir, f"{patch_name}.png")
            patch.save(patch_path, quality=95)  # High quality PNG

            # Calculate GPS coordinates for patch center
            min_x, min_y, max_x, max_y = bounds
            tile_size = 256

            # Pixel position in tile coordinates
            center_px = x + patch_size / 2
            center_py = y + patch_size / 2

            # Convert to tile coordinates
            tile_x = min_x + center_px / tile_size
            tile_y = min_y + center_py / tile_size

            # Convert to lat/lon
            n = 2.0 ** zoom
            lon = tile_x / n * 360.0 - 180.0
            lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
            lat = math.degrees(lat_rad)

            # Calculate bounds for the patch
            patch_min_tile_x = min_x + x / tile_size
            patch_max_tile_x = min_x + (x + patch_size) / tile_size
            patch_min_tile_y = min_y + y / tile_size
            patch_max_tile_y = min_y + (y + patch_size) / tile_size

            patch_min_lon = patch_min_tile_x / n * 360.0 - 180.0
            patch_max_lon = patch_max_tile_x / n * 360.0 - 180.0

            lat_rad_min = math.atan(math.sinh(math.pi * (1 - 2 * patch_min_tile_y / n)))
            lat_rad_max = math.atan(math.sinh(math.pi * (1 - 2 * patch_max_tile_y / n)))
            patch_max_lat = math.degrees(lat_rad_min)  # Note: inverted
            patch_min_lat = math.degrees(lat_rad_max)

            # Store metadata
            gps_metadata[patch_name] = {
                "lat": lat,
                "lon": lon,
                "bounds": {
                    "min_lat": patch_min_lat,
                    "max_lat": patch_max_lat,
                    "min_lon": patch_min_lon,
                    "max_lon": patch_max_lon
                }
            }

            patches_created += 1
            pbar.update(1)

    pbar.close()

    # Save GPS metadata
    metadata_path = os.path.join(output_dir, "gps_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(gps_metadata, f, indent=2)

    print(f"✓ Saved {patches_created} patches to {output_dir}/")
    print(f"✓ Saved GPS metadata to {metadata_path}")

    return patches_created, gps_metadata


def main():
    """Main execution function."""

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("🛰️  Downloading satellite imagery from Google Maps...")
    print(f"   Area: {config['area']['min_lat']:.3f}→{config['area']['max_lat']:.3f}°N, "
          f"{config['area']['min_lon']:.3f}→{config['area']['max_lon']:.3f}°E")
    print(f"   Zoom: {config['satellite']['zoom_level']} "
          f"(~{156543.03 * math.cos(math.radians((config['area']['min_lat'] + config['area']['max_lat']) / 2)) / (2 ** config['satellite']['zoom_level']):.2f} m/pixel)")

    # Get API key if available
    api_key = config['satellite'].get('google_api_key', None)

    if not api_key:
        print("\n⚠️  No Google API key found in config.yaml")
        print("   Using free tile server (may have lower quality/restrictions)")
        print("   To use high-quality imagery, add 'google_api_key' to config.yaml")

    # Download and stitch tiles
    stitched_image, tile_bounds = download_area(
        config['area']['min_lat'],
        config['area']['max_lat'],
        config['area']['min_lon'],
        config['area']['max_lon'],
        config['satellite']['zoom_level'],
        config['satellite']['delay_between_requests'],
        api_key
    )

    print(f"\nCreating patches ({config['patches']['patch_size']}×{config['patches']['patch_size']}, "
          f"{int(config['patches']['overlap'] * 100)}% overlap)...")

    # Create patches
    num_patches, gps_metadata = create_patches(
        stitched_image,
        config['patches']['patch_size'],
        config['patches']['overlap'],
        config['patches']['output_dir'],
        tile_bounds,
        config['satellite']['zoom_level']
    )

    # Calculate area covered
    lats = [m['lat'] for m in gps_metadata.values()]
    lons = [m['lon'] for m in gps_metadata.values()]
    lat_range = max(lats) - min(lats)
    lon_range = max(lons) - min(lons)

    # Approximate distances (rough estimate)
    lat_dist_m = lat_range * 111320  # meters per degree latitude
    lon_dist_m = lon_range * 111320 * math.cos(math.radians(np.mean(lats)))

    # Calculate total size
    total_size_mb = sum(
        os.path.getsize(os.path.join(config['patches']['output_dir'], f"patch_{i}_{j}.png"))
        for i in range(int(math.sqrt(num_patches)) + 1)
        for j in range(int(math.sqrt(num_patches)) + 1)
        if os.path.exists(os.path.join(config['patches']['output_dir'], f"patch_{i}_{j}.png"))
    ) / (1024 * 1024)

    print("\nSummary:")
    print(f"   • Area covered: ~{lat_dist_m:.0f}m × {lon_dist_m:.0f}m")
    print(f"   • Patches: {num_patches}")
    print(f"   • Overlap: {int(config['patches']['overlap'] * 100)}%")
    print(f"   • Format: {config['patches']['format'].upper()} ({config['patches']['patch_size']}×{config['patches']['patch_size']})")
    print(f"   • Total size: {total_size_mb:.1f} MB")
    print(f"   • Source: {'Google Maps API' if api_key else 'Google Maps tiles'}")
    print("\n✅ Satellite imagery download complete!")


if __name__ == "__main__":
    main()

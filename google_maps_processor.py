#!/usr/bin/env python3
"""
Google Maps Processor for Aerial Video Matching

This module handles:
1. Converting large satellite/aerial images into grid patches
2. Generating GPS coordinates for each patch
3. Creating metadata files for AnyLoc drone navigation system

Usage:
    # Process a large satellite image into grid patches
    python3 google_maps_processor.py --input large_satellite_map.png --bounds "lat1,lon1,lat2,lon2" --output ./grid_patches

    # Download tiles from open-source providers (alternative to Google Maps)
    python3 google_maps_processor.py --download-osm --bounds "40.75,-73.99,40.76,-73.98" --zoom 18 --output ./osm_tiles
"""

import os
import sys
from pathlib import Path
import argparse
import json
import math
import warnings
warnings.filterwarnings("ignore")

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

import numpy as np
from PIL import Image
import requests
import time
from typing import Tuple, List, Dict

class GoogleMapsProcessor:
    """Process satellite imagery for drone navigation database"""
    
    def __init__(self, output_dir="./satellite_patches"):
        self.output_dir = output_dir
        self.patches_dir = f"{output_dir}/patches"
        self.metadata_file = f"{output_dir}/gps_metadata.json"
        
        os.makedirs(self.patches_dir, exist_ok=True)
        
    def process_large_satellite_image(self, image_path: str, bounds: Tuple[float, float, float, float], 
                                    patch_size: int = 512, overlap: int = 64):
        """
        Convert a large satellite image into grid patches with GPS coordinates
        
        Args:
            image_path: Path to large satellite image
            bounds: (lat_min, lon_min, lat_max, lon_max) GPS bounds of the image
            patch_size: Size of each patch in pixels
            overlap: Overlap between patches in pixels
        
        Returns:
            dict: GPS metadata for all patches
        """
        print("🗺️  PROCESSING LARGE SATELLITE IMAGE")
        print("=" * 50)
        
        # Load the large image
        print(f"📂 Loading image: {image_path}")
        satellite_img = Image.open(image_path)
        img_width, img_height = satellite_img.size
        print(f"📐 Image dimensions: {img_width} x {img_height}")
        
        # Calculate GPS parameters
        lat_min, lon_min, lat_max, lon_max = bounds
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        
        print(f"🌍 GPS bounds: ({lat_min:.6f}, {lon_min:.6f}) to ({lat_max:.6f}, {lon_max:.6f})")
        
        # Calculate step size accounting for overlap
        step_size = patch_size - overlap
        
        # Calculate number of patches
        patches_x = (img_width - patch_size) // step_size + 1
        patches_y = (img_height - patch_size) // step_size + 1
        
        print(f"📊 Creating {patches_x} x {patches_y} = {patches_x * patches_y} patches")
        print(f"   Patch size: {patch_size}x{patch_size}, Overlap: {overlap}px")
        
        gps_metadata = {}
        patch_count = 0
        
        for y in range(patches_y):
            for x in range(patches_x):
                # Calculate pixel coordinates
                pixel_x = x * step_size
                pixel_y = y * step_size
                
                # Ensure we don't go beyond image bounds
                if pixel_x + patch_size > img_width:
                    pixel_x = img_width - patch_size
                if pixel_y + patch_size > img_height:
                    pixel_y = img_height - patch_size
                
                # Extract patch
                bbox = (pixel_x, pixel_y, pixel_x + patch_size, pixel_y + patch_size)
                patch = satellite_img.crop(bbox)
                
                # Calculate GPS coordinates (center of patch)
                # Note: y increases downward in images, but latitude increases upward
                center_x_ratio = (pixel_x + patch_size/2) / img_width
                center_y_ratio = (pixel_y + patch_size/2) / img_height
                
                patch_lon = lon_min + center_x_ratio * lon_range
                patch_lat = lat_max - center_y_ratio * lat_range  # Flip Y axis
                
                # Save patch
                patch_name = f"patch_{y:03d}_{x:03d}"
                patch_path = f"{self.patches_dir}/{patch_name}.png"
                patch.save(patch_path)
                
                # Store GPS metadata
                gps_metadata[patch_name] = {
                    "lat": round(patch_lat, 8),
                    "lon": round(patch_lon, 8),
                    "altitude": 100,  # Approximate satellite view altitude
                    "patch_bounds": {
                        "lat_min": round(lat_max - ((pixel_y + patch_size) / img_height) * lat_range, 8),
                        "lat_max": round(lat_max - (pixel_y / img_height) * lat_range, 8),
                        "lon_min": round(lon_min + (pixel_x / img_width) * lon_range, 8),
                        "lon_max": round(lon_min + ((pixel_x + patch_size) / img_width) * lon_range, 8)
                    },
                    "pixel_coords": {
                        "x": pixel_x,
                        "y": pixel_y,
                        "width": patch_size,
                        "height": patch_size
                    },
                    "source": "processed_satellite_image",
                    "patch_id": f"{y:03d}_{x:03d}"
                }
                
                patch_count += 1
                if patch_count % 100 == 0:
                    print(f"   ⏳ Processed {patch_count}/{patches_x * patches_y} patches...")
        
        # Save GPS metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(gps_metadata, f, indent=2)
        
        print(f"✅ Successfully created {patch_count} patches")
        print(f"📍 GPS metadata saved to: {self.metadata_file}")
        print(f"🖼️  Patches saved to: {self.patches_dir}")
        
        return gps_metadata
    
    def download_osm_tiles(self, bounds: Tuple[float, float, float, float], zoom: int = 18, delay: float = 0.1):
        """
        Download aerial tiles from OpenStreetMap (alternative to Google Maps)
        
        Args:
            bounds: (lat_min, lon_min, lat_max, lon_max) GPS bounds
            zoom: Zoom level (18 is good for drone altitude matching)
            delay: Delay between requests (be respectful to tile servers)
        
        Returns:
            dict: GPS metadata for downloaded tiles
        """
        print("🌍 DOWNLOADING OPENSTREETMAP TILES")
        print("=" * 50)
        
        lat_min, lon_min, lat_max, lon_max = bounds
        
        # Convert GPS bounds to tile coordinates
        def deg2num(lat_deg, lon_deg, zoom):
            lat_rad = math.radians(lat_deg)
            n = 2.0 ** zoom
            x = int((lon_deg + 180.0) / 360.0 * n)
            y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
            return (x, y)
        
        def num2deg(x, y, zoom):
            n = 2.0 ** zoom
            lon_deg = x / n * 360.0 - 180.0
            lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
            lat_deg = math.degrees(lat_rad)
            return (lat_deg, lon_deg)
        
        # Get tile bounds
        x_min, y_max = deg2num(lat_min, lon_min, zoom)  # Note: y is flipped
        x_max, y_min = deg2num(lat_max, lon_max, zoom)
        
        print(f"📊 Tile range: X({x_min}-{x_max}), Y({y_min}-{y_max}), Zoom: {zoom}")
        
        total_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
        print(f"📦 Total tiles to download: {total_tiles}")
        
        gps_metadata = {}
        tile_count = 0
        
        # OSM tile server (you can change this to other providers)
        tile_server = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                try:
                    # Download tile
                    url = tile_server.format(z=zoom, x=x, y=y)
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        # Save tile
                        tile_name = f"tile_{zoom}_{x}_{y}"
                        tile_path = f"{self.patches_dir}/{tile_name}.png"
                        
                        with open(tile_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Calculate GPS coordinates for tile center
                        center_lat, center_lon = num2deg(x + 0.5, y + 0.5, zoom)
                        corner_lat_1, corner_lon_1 = num2deg(x, y, zoom)
                        corner_lat_2, corner_lon_2 = num2deg(x + 1, y + 1, zoom)
                        
                        # Store GPS metadata
                        gps_metadata[tile_name] = {
                            "lat": round(center_lat, 8),
                            "lon": round(center_lon, 8),
                            "altitude": 100,
                            "tile_bounds": {
                                "lat_min": round(min(corner_lat_1, corner_lat_2), 8),
                                "lat_max": round(max(corner_lat_1, corner_lat_2), 8),
                                "lon_min": round(min(corner_lon_1, corner_lon_2), 8),
                                "lon_max": round(max(corner_lon_1, corner_lon_2), 8)
                            },
                            "tile_coords": {"x": x, "y": y, "z": zoom},
                            "source": "openstreetmap",
                            "url": url
                        }
                        
                        tile_count += 1
                        if tile_count % 10 == 0:
                            print(f"   ⏳ Downloaded {tile_count}/{total_tiles} tiles...")
                    
                    else:
                        print(f"   ⚠️  Failed to download tile {x},{y}: {response.status_code}")
                    
                    # Be respectful to tile servers
                    time.sleep(delay)
                    
                except Exception as e:
                    print(f"   ❌ Error downloading tile {x},{y}: {e}")
                    continue
        
        # Save GPS metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(gps_metadata, f, indent=2)
        
        print(f"✅ Successfully downloaded {tile_count} tiles")
        print(f"📍 GPS metadata saved to: {self.metadata_file}")
        print(f"🖼️  Tiles saved to: {self.patches_dir}")
        
        return gps_metadata
    
    def create_coordinate_grid_overlay(self, gps_metadata: dict, output_path: str = None):
        """Create a visualization showing the coordinate grid"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            if output_path is None:
                output_path = f"{self.output_dir}/coordinate_grid.png"
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract coordinates
            lats = [item['lat'] for item in gps_metadata.values()]
            lons = [item['lon'] for item in gps_metadata.values()]
            
            # Plot grid points
            ax.scatter(lons, lats, c='red', s=10, alpha=0.6)
            
            # Add grid lines if patches have bounds
            for patch_name, data in gps_metadata.items():
                if 'patch_bounds' in data or 'tile_bounds' in data:
                    bounds_key = 'patch_bounds' if 'patch_bounds' in data else 'tile_bounds'
                    bounds = data[bounds_key]
                    
                    rect = patches.Rectangle(
                        (bounds['lon_min'], bounds['lat_min']),
                        bounds['lon_max'] - bounds['lon_min'],
                        bounds['lat_max'] - bounds['lat_min'],
                        linewidth=0.5, edgecolor='blue', facecolor='none', alpha=0.3
                    )
                    ax.add_patch(rect)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'Satellite Patch Grid ({len(gps_metadata)} patches)')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Grid visualization saved to: {output_path}")
            
        except ImportError:
            print("⚠️  Matplotlib not available, skipping grid visualization")

def main():
    parser = argparse.ArgumentParser(description="Process satellite imagery for drone navigation")
    
    parser.add_argument('--input', type=str, help='Path to large satellite image')
    parser.add_argument('--bounds', type=str, required=True, 
                       help='GPS bounds as "lat_min,lon_min,lat_max,lon_max"')
    parser.add_argument('--output', type=str, default='./satellite_patches',
                       help='Output directory for patches')
    parser.add_argument('--patch-size', type=int, default=512,
                       help='Size of each patch in pixels')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Overlap between patches in pixels')
    parser.add_argument('--download-osm', action='store_true',
                       help='Download tiles from OpenStreetMap instead of processing image')
    parser.add_argument('--zoom', type=int, default=18,
                       help='Zoom level for tile download (18 is good for drone matching)')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between tile downloads (seconds)')
    
    args = parser.parse_args()
    
    # Parse bounds
    try:
        bounds = tuple(map(float, args.bounds.split(',')))
        if len(bounds) != 4:
            raise ValueError("Bounds must have 4 values")
    except:
        print("❌ Error: Bounds must be in format 'lat_min,lon_min,lat_max,lon_max'")
        return
    
    # Create processor
    processor = GoogleMapsProcessor(args.output)
    
    if args.download_osm:
        # Download OSM tiles
        gps_metadata = processor.download_osm_tiles(bounds, args.zoom, args.delay)
    else:
        # Process large satellite image
        if not args.input:
            print("❌ Error: --input required when not using --download-osm")
            return
        
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file not found: {args.input}")
            return
        
        gps_metadata = processor.process_large_satellite_image(
            args.input, bounds, args.patch_size, args.overlap
        )
    
    # Create visualization
    processor.create_coordinate_grid_overlay(gps_metadata)
    
    print("\n🎉 Processing complete!")
    print(f"📁 Output directory: {args.output}")
    print(f"📊 Total patches/tiles: {len(gps_metadata)}")
    print("\n📋 Next steps:")
    print(f"   1. Use patches directory: {processor.patches_dir}")
    print(f"   2. Use GPS metadata file: {processor.metadata_file}")
    print(f"   3. Build AnyLoc database with these files")

if __name__ == "__main__":
    main()
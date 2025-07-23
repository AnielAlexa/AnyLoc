#!/usr/bin/env python3
"""
Demo the composite satellite image functionality
"""

import os
import sys
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

def create_composite_demo():
    """Create composite satellite image with simulated highlighting"""
    
    print("🗺️  Creating composite satellite image demo...")
    
    # Load satellite patches and GPS data
    patches_dir = Path("./flight_results/satellite_patches/patches")
    gps_file = Path("./flight_results/satellite_patches/gps_metadata.json")
    
    if not patches_dir.exists() or not gps_file.exists():
        print("❌ Required files not found")
        return
    
    # Load GPS metadata
    with open(gps_file, 'r') as f:
        gps_metadata = json.load(f)
    
    # Get patch files
    patch_files = {}
    for img_file in patches_dir.glob("*.png"):
        patch_name = img_file.stem
        patch_files[patch_name] = str(img_file)
    
    print(f"📊 Found {len(patch_files)} patches with GPS data for {len(gps_metadata)} entries")
    
    # Extract coordinates and sort patches by GPS position
    patch_coords = []
    for patch_name, patch_path in patch_files.items():
        if patch_name in gps_metadata:
            gps_data = gps_metadata[patch_name]
            lat = gps_data.get('lat', 0)
            lon = gps_data.get('lon', 0)
            patch_coords.append({
                'name': patch_name,
                'path': patch_path,
                'lat': lat,
                'lon': lon
            })
    
    # Sort by lat/lon to determine grid layout
    patch_coords.sort(key=lambda x: (-x['lat'], x['lon']))  # Sort by lat desc, lon asc
    
    # Determine grid dimensions
    unique_lats = sorted(list(set([p['lat'] for p in patch_coords])), reverse=True)
    unique_lons = sorted(list(set([p['lon'] for p in patch_coords])))
    
    rows = len(unique_lats)
    cols = len(unique_lons)
    
    print(f"📐 Grid layout: {rows} x {cols}")
    
    # Load first patch to get dimensions
    first_patch = Image.open(patch_coords[0]['path'])
    patch_width, patch_height = first_patch.size
    
    print(f"🖼️  Patch size: {patch_width} x {patch_height}")
    
    # Create composite image
    composite_width = cols * patch_width
    composite_height = rows * patch_height
    composite = Image.new('RGB', (composite_width, composite_height), color='black')
    
    print(f"🎨 Composite size: {composite_width} x {composite_height}")
    
    # Place patches in composite and store coordinates
    patch_coordinates = {}
    
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
        patch_coordinates[patch_info['name']] = {
            'x': x,
            'y': y,
            'width': patch_width,
            'height': patch_height
        }
        
        print(f"   {patch_info['name']}: ({x}, {y})")
    
    # Save plain composite
    composite.save("satellite_composite_plain.png")
    print("✅ Plain composite saved: satellite_composite_plain.png")
    
    # Create highlighted versions with different confidence levels
    confidence_demos = [
        ("high", 0.65, 'lime'),
        ("medium", 0.42, 'orange'),
        ("low", 0.28, 'red')
    ]
    
    # Use middle patch for demonstration
    demo_patch = patch_coords[len(patch_coords)//2]['name']
    print(f"🎯 Using patch '{demo_patch}' for highlighting demo")
    
    for conf_name, confidence, color in confidence_demos:
        # Create working copy
        demo_image = composite.copy()
        draw = ImageDraw.Draw(demo_image)
        
        if demo_patch in patch_coordinates:
            coords = patch_coordinates[demo_patch]
            
            # Determine border width
            if confidence > 0.5:
                width = 4
            elif confidence > 0.3:
                width = 3
            else:
                width = 2
            
            # Draw highlight border
            x, y = coords['x'], coords['y']
            w, h = coords['width'], coords['height']
            
            # Draw thick border
            for i in range(width):
                draw.rectangle([x-i, y-i, x+w+i, y+h+i], outline=color, width=1)
            
            # Add confidence text
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
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
        
        # Save highlighted version
        output_file = f"satellite_composite_{conf_name}_confidence.png"
        demo_image.save(output_file)
        print(f"✅ {conf_name.title()} confidence demo saved: {output_file}")
    
    print(f"\n🎉 Composite demo completed!")
    print(f"📁 Generated files:")
    print(f"   • satellite_composite_plain.png - Base composite image")
    print(f"   • satellite_composite_high_confidence.png - Green highlight (>0.5)")
    print(f"   • satellite_composite_medium_confidence.png - Orange highlight (>0.3)")
    print(f"   • satellite_composite_low_confidence.png - Red highlight (<0.3)")

def main():
    create_composite_demo()

if __name__ == "__main__":
    main()
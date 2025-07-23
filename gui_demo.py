#!/usr/bin/env python3
"""
Simple demo of the GUI functionality without continuous playback
"""

import os
import sys
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

from drone_navigation_system import DroneNavigationSystem

def create_gui_mockup():
    """Create a mockup image showing what the GUI looks like"""
    
    print("🎨 Creating GUI mockup...")
    
    # Load satellite patches
    patches_dir = Path("./flight_results/satellite_patches/patches")
    patch_files = list(patches_dir.glob("*.png"))[:9]  # Take first 9 patches
    
    if len(patch_files) < 9:
        print("❌ Need at least 9 patches for demo")
        return
        
    # Load video frame
    video_path = "/home/aniel/Downloads/flight_100m.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, video_frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not load video frame")
        return
        
    # Create mockup canvas
    canvas_width = 1200
    canvas_height = 800
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    # Draw title
    draw.text((10, 10), "Aerial Video GPS Estimation - AnyLoc GUI", fill='black', font=title_font)
    
    # Left panel: Satellite patches grid (3x3)
    patch_size = 120
    start_x, start_y = 20, 50
    
    draw.text((start_x, start_y), "Satellite Patches:", fill='black', font=font)
    grid_start_y = start_y + 30
    
    for i, patch_file in enumerate(patch_files):
        row = i // 3
        col = i % 3
        
        x = start_x + col * (patch_size + 10)
        y = grid_start_y + row * (patch_size + 30)
        
        # Load and resize patch
        patch_img = Image.open(patch_file)
        patch_img = patch_img.resize((patch_size, patch_size))
        
        # Paste patch
        canvas.paste(patch_img, (x, y))
        
        # Add border (green for "matched" patch at position 4)
        border_color = 'green' if i == 4 else 'gray'
        border_width = 3 if i == 4 else 1
        draw.rectangle([x-border_width, y-border_width, 
                       x+patch_size+border_width, y+patch_size+border_width], 
                      outline=border_color, width=border_width)
        
        # Add patch name
        patch_name = patch_file.stem[-8:]  # Last 8 chars
        draw.text((x, y + patch_size + 5), patch_name, fill='black', font=font)
    
    # Right panel: Video frame
    video_x = 450
    video_y = 50
    video_width = 400
    video_height = 300
    
    draw.text((video_x, video_y), "Current Video Frame:", fill='black', font=font)
    
    # Resize video frame
    video_pil = Image.fromarray(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
    video_resized = video_pil.resize((video_width, video_height))
    canvas.paste(video_resized, (video_x, video_y + 30))
    
    # Draw border around video
    draw.rectangle([video_x-1, video_y+29, video_x+video_width+1, video_y+video_height+31], 
                  outline='black', width=1)
    
    # Control panel
    control_y = video_y + video_height + 60
    draw.text((video_x, control_y), "Controls:", fill='black', font=font)
    
    # Draw buttons
    button_y = control_y + 30
    buttons = ["Play", "Step", "Reset"]
    for i, button_text in enumerate(buttons):
        button_x = video_x + i * 80
        draw.rectangle([button_x, button_y, button_x + 70, button_y + 30], 
                      outline='gray', fill='lightgray')
        draw.text((button_x + 20, button_y + 8), button_text, fill='black', font=font)
    
    # Progress bar
    progress_y = button_y + 50
    draw.text((video_x, progress_y), "Progress:", fill='black', font=font)
    draw.rectangle([video_x, progress_y + 25, video_x + 300, progress_y + 35], 
                  outline='gray', fill='lightgray')
    # Fill 30% progress
    draw.rectangle([video_x, progress_y + 25, video_x + 90, progress_y + 35], 
                  fill='blue')
    
    # Info panel
    info_y = progress_y + 60
    draw.text((video_x, info_y), "Match Information:", fill='black', font=font)
    
    # Sample match info
    info_text = [
        "Frame: 150",
        "✅ MATCH FOUND",
        "Best Match: tile_18_146503_93481", 
        "Confidence: 0.421",
        "GPS: 45.790031, 21.193314",
        "Processing time: 0.034s"
    ]
    
    for i, line in enumerate(info_text):
        color = 'green' if '✅' in line else 'black'
        draw.text((video_x, info_y + 25 + i * 20), line, fill=color, font=font)
    
    # Add highlighting annotation
    draw.text((start_x, grid_start_y + 3 * (patch_size + 30) + 20), 
              "Green border = Current match", fill='green', font=font)
    
    # Save mockup
    output_path = "gui_mockup.png"
    canvas.save(output_path)
    print(f"✅ GUI mockup saved to: {output_path}")
    
    return output_path

def demo_matching_process():
    """Demonstrate the matching process"""
    
    print("\n🔍 Demonstrating matching process...")
    
    # Initialize navigation system
    nav_system = DroneNavigationSystem("./flight_results/anyloc_database")
    nav_system.load_satellite_database()
    
    # Load video
    video_path = "/home/aniel/Downloads/flight_100m.mp4"
    cap = cv2.VideoCapture(video_path)
    
    print(f"📹 Video loaded: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
    
    # Test several frames
    test_frames = [100, 200, 300, 400, 500]
    
    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            result = nav_system.match_drone_frame(frame)
            
            status = "✅ MATCH" if result.get('is_valid_match', False) else "❌ NO MATCH"
            similarity = result.get('similarity', 0)
            best_match = result.get('best_match_image', 'None')
            
            print(f"Frame {frame_num:3d}: {status} | Similarity: {similarity:.3f} | Best: {best_match}")
    
    cap.release()

def main():
    """Run GUI demo"""
    print("🎮 GUI Demo - Aerial Video GPS Estimation\n")
    
    # Create visual mockup
    mockup_path = create_gui_mockup()
    
    # Demonstrate matching
    demo_matching_process()
    
    print(f"\n✅ Demo completed!")
    print(f"📷 GUI mockup saved as: {mockup_path}")
    print("\n💡 To run the actual GUI:")
    print("   python3 aerial_video_gui.py --video /home/aniel/Downloads/flight_100m.mp4 --database ./flight_results/anyloc_database")

if __name__ == "__main__":
    main()
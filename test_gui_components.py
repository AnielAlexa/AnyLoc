#!/usr/bin/env python3
"""
Test GUI components for aerial video matching
"""

import os
import sys
from pathlib import Path
import json

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

from drone_navigation_system import DroneNavigationSystem
import cv2

def test_navigation_system():
    """Test the navigation system components"""
    
    print("Testing DroneNavigationSystem...")
    
    # Initialize system
    database_path = "./flight_results/anyloc_database"
    nav_system = DroneNavigationSystem(database_path)
    
    try:
        # Load database
        nav_system.load_satellite_database()
        print(f"✅ Database loaded: {len(nav_system.database)} images")
        
        # Test with a video frame
        video_path = "/home/aniel/Downloads/flight_100m.mp4"
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                print("✅ Video frame loaded successfully")
                
                # Test matching
                result = nav_system.match_drone_frame(frame)
                print(f"✅ Frame matching completed")
                print(f"   Best match: {result.get('best_match_image', 'None')}")
                print(f"   Similarity: {result.get('similarity', 0):.3f}")
                print(f"   Valid match: {result.get('is_valid_match', False)}")
                
                cap.release()
            else:
                print("❌ Could not read video frame")
        else:
            print("❌ Video file not found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_satellite_patches():
    """Test satellite patch loading"""
    
    print("\nTesting satellite patch loading...")
    
    satellite_dir = Path("./flight_results/satellite_patches")
    patches_dir = satellite_dir / "patches"
    gps_file = satellite_dir / "gps_metadata.json"
    
    if not patches_dir.exists():
        print("❌ Patches directory not found")
        return
        
    # Count patches
    patch_files = list(patches_dir.glob("*.png"))
    print(f"✅ Found {len(patch_files)} satellite patches")
    
    # Load GPS metadata
    if gps_file.exists():
        with open(gps_file, 'r') as f:
            gps_data = json.load(f)
        print(f"✅ GPS metadata loaded: {len(gps_data)} entries")
        
        # Show sample GPS data
        for i, (name, data) in enumerate(gps_data.items()):
            if i < 3:  # Show first 3
                print(f"   {name}: lat={data.get('lat', 'N/A')}, lon={data.get('lon', 'N/A')}")
    else:
        print("❌ GPS metadata file not found")

def main():
    """Run all tests"""
    print("🧪 Testing GUI Components\n")
    
    test_satellite_patches()
    test_navigation_system()
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()
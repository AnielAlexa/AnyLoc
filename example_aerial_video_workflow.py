#!/usr/bin/env python3
"""
Example Workflow for Aerial Video GPS Matching

This script demonstrates the complete workflow using the existing mock drone test data
as a starting point, then shows how to process a real aerial video.

Usage:
    python3 example_aerial_video_workflow.py
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

import cv2
import numpy as np
from PIL import Image
import json

def create_sample_aerial_video(output_path="./sample_aerial_video.mp4", duration_seconds=10, fps=30):
    """
    Create a sample aerial video using synthetic satellite images
    This simulates drone footage by adding transformations to satellite imagery
    """
    print("🎥 Creating sample aerial video...")
    
    # Use existing mock satellite images if available
    satellite_dir = "./mock_drone_test/satellite_images"
    if not os.path.exists(satellite_dir):
        print("   ⚠️  Mock satellite images not found. Creating synthetic video...")
        return create_synthetic_aerial_video(output_path, duration_seconds, fps)
    
    # Get satellite images
    satellite_files = sorted([f for f in os.listdir(satellite_dir) if f.endswith('.png')])
    if not satellite_files:
        print("   ⚠️  No satellite images found. Creating synthetic video...")
        return create_synthetic_aerial_video(output_path, duration_seconds, fps)
    
    # Video parameters
    total_frames = duration_seconds * fps
    frame_size = (640, 480)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    print(f"   📹 Creating {total_frames} frames at {fps} FPS...")
    
    for frame_num in range(total_frames):
        # Cycle through satellite images
        sat_idx = (frame_num // 30) % len(satellite_files)  # Change image every second
        sat_path = os.path.join(satellite_dir, satellite_files[sat_idx])
        
        # Load satellite image
        sat_img = cv2.imread(sat_path)
        if sat_img is None:
            continue
        
        # Simulate drone movement by cropping different regions
        h, w = sat_img.shape[:2]
        
        # Add some movement/drift
        t = frame_num / total_frames
        center_x = int(w/2 + 50 * np.sin(t * 4))  # Oscillating movement
        center_y = int(h/2 + 30 * np.cos(t * 3))
        
        # Extract crop region
        crop_size = 400
        x1 = max(0, min(w - crop_size, center_x - crop_size//2))
        y1 = max(0, min(h - crop_size, center_y - crop_size//2))
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        cropped = sat_img[y1:y2, x1:x2]
        
        # Add some noise and blur to simulate drone footage
        if frame_num % 3 == 0:  # Add noise every few frames
            noise = np.random.normal(0, 10, cropped.shape).astype(np.uint8)
            cropped = cv2.addWeighted(cropped, 0.9, noise, 0.1, 0)
        
        # Add slight blur occasionally
        if frame_num % 7 == 0:
            cropped = cv2.GaussianBlur(cropped, (3, 3), 0)
        
        # Resize to target frame size
        frame = cv2.resize(cropped, frame_size)
        
        # Write frame
        out.write(frame)
        
        if frame_num % 60 == 0:
            print(f"   ⏳ Progress: {frame_num}/{total_frames} frames...")
    
    out.release()
    print(f"✅ Sample video created: {output_path}")
    return output_path

def create_synthetic_aerial_video(output_path, duration_seconds=10, fps=30):
    """Create a completely synthetic aerial video for demonstration"""
    
    frame_size = (640, 480)
    total_frames = duration_seconds * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    print(f"   🎨 Creating synthetic aerial video with {total_frames} frames...")
    
    for frame_num in range(total_frames):
        # Create synthetic aerial scene
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        
        # Add ground texture (grass/fields)
        frame[:, :, 1] = 100 + 30 * np.random.random((frame_size[1], frame_size[0]))  # Green base
        
        # Add some "roads" or features
        t = frame_num / total_frames
        road_y = int(frame_size[1] * (0.3 + 0.4 * t))  # Moving road
        cv2.rectangle(frame, (0, road_y-5), (frame_size[0], road_y+5), (128, 128, 128), -1)
        
        # Add some "buildings" or structures
        for i in range(3):
            bldg_x = int(frame_size[0] * (0.2 + 0.3 * i))
            bldg_y = int(frame_size[1] * (0.6 + 0.1 * np.sin(t * 2 + i)))
            cv2.rectangle(frame, (bldg_x-20, bldg_y-20), (bldg_x+20, bldg_y+20), (80, 80, 150), -1)
        
        # Add movement by shifting colors slightly
        frame = cv2.addWeighted(frame, 0.9, 
                               np.random.randint(0, 20, frame.shape, dtype=np.uint8), 0.1, 0)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Synthetic video created: {output_path}")
    return output_path

def demonstrate_complete_workflow():
    """Demonstrate the complete aerial video matching workflow"""
    print("🚁 AERIAL VIDEO GPS MATCHING DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Create sample data if needed
    sample_video = "./example_aerial_video.mp4"
    if not os.path.exists(sample_video):
        create_sample_aerial_video(sample_video, duration_seconds=15, fps=30)
    
    # Step 2: Use existing mock satellite database or create new one
    satellite_db_dir = "./mock_drone_test/database"
    if os.path.exists(satellite_db_dir):
        print("📊 Using existing mock satellite database...")
        database_path = satellite_db_dir
    else:
        print("📊 Creating new satellite database from OpenStreetMap...")
        # Create a small area around Central Park for demonstration
        bounds = "40.764,-73.982,40.800,-73.949"  # Central Park area
        
        # Use the complete pipeline to download tiles and build database
        os.system(f"""python3 aerial_video_pipeline.py \
            --video {sample_video} \
            --download-osm \
            --bounds "{bounds}" \
            --zoom 17 \
            --frame-interval 45 \
            --output ./example_results \
            --confidence-threshold 0.1""")
        
        print("✅ Complete pipeline demonstration finished!")
        print("📁 Check ./example_results/ for all outputs")
        return
    
    # Step 3: Process the video against the existing database
    print("🎬 Processing aerial video...")
    
    from aerial_video_processor import AerialVideoProcessor
    
    processor = AerialVideoProcessor(database_path, "./example_video_results")
    
    results = processor.process_video(
        sample_video,
        frame_interval=60,  # Process every 2 seconds 
        confidence_threshold=0.1,  # Lower threshold for demo
        max_frames=20,  # Limit for quick demo
        save_processed_frames=True
    )
    
    # Create visualizations
    processor.create_trajectory_visualization(results)
    processor.export_kml(results)
    
    # Step 4: Show results summary
    valid_results = [r for r in results if r.is_valid_match]
    
    print(f"\n📊 DEMONSTRATION RESULTS:")
    print(f"   Total frames processed: {len(results)}")
    print(f"   Valid GPS estimates: {len(valid_results)}")
    print(f"   Success rate: {len(valid_results)/len(results)*100:.1f}%")
    
    if valid_results:
        avg_confidence = sum(r.confidence for r in valid_results) / len(valid_results)
        print(f"   Average confidence: {avg_confidence:.3f}")
        
        # Show a few sample matches
        print(f"\n📍 Sample GPS Estimates:")
        for i, result in enumerate(valid_results[:3]):
            if result.gps_estimate:
                print(f"   Frame {result.frame_number}: "
                      f"({result.gps_estimate['lat']:.6f}, {result.gps_estimate['lon']:.6f}) "
                      f"confidence: {result.confidence:.3f}")
    
    print(f"\n📁 Results saved to: ./example_video_results/")
    print(f"   📊 Trajectory plot: ./example_video_results/trajectory_plot.png")
    print(f"   🌍 KML file: ./example_video_results/flight_trajectory.kml")
    
    return results

def show_usage_examples():
    """Show practical usage examples"""
    print("\n💡 PRACTICAL USAGE EXAMPLES")
    print("=" * 40)
    
    examples = [
        {
            "name": "Property Survey",
            "description": "High-accuracy mapping for real estate",
            "command": """python3 aerial_video_pipeline.py \\
    --video property_drone_footage.mp4 \\
    --satellite-image high_res_satellite_map.png \\
    --bounds "40.7580,-73.9855,40.7600,-73.9830" \\
    --patch-size 1024 \\
    --frame-interval 15 \\
    --confidence-threshold 0.4 \\
    --output ./property_survey_results"""
        },
        {
            "name": "Search & Rescue",
            "description": "Fast processing for emergency response",
            "command": """python3 aerial_video_pipeline.py \\
    --video search_footage.mp4 \\
    --download-osm \\
    --bounds "search_area_bounds" \\
    --zoom 18 \\
    --frame-interval 60 \\
    --confidence-threshold 0.2 \\
    --max-frames 300 \\
    --output ./search_results"""
        },
        {
            "name": "Agricultural Monitoring", 
            "description": "Field boundary detection and crop monitoring",
            "command": """python3 aerial_video_pipeline.py \\
    --video crop_inspection.mp4 \\
    --satellite-image farm_satellite_imagery.tif \\
    --bounds "farm_coordinates" \\
    --patch-size 512 \\
    --overlap 128 \\
    --export-kml \\
    --output ./farm_analysis"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   📝 {example['description']}")
        print(f"   💻 Command:")
        for line in example['command'].split('\n'):
            if line.strip():
                print(f"      {line}")

def main():
    """Main demonstration function"""
    
    print("🎯 This script demonstrates the aerial video GPS matching system")
    print("   It will create sample data and show the complete workflow\n")
    
    try:
        # Run the demonstration
        results = demonstrate_complete_workflow()
        
        # Show usage examples
        show_usage_examples()
        
        print(f"\n🎉 Demonstration complete!")
        print(f"📚 See AERIAL_VIDEO_MATCHING_README.md for full documentation")
        print(f"🔧 The system is ready for your own aerial video processing!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        print(f"💡 Make sure you're in the AnyLoc directory with the conda environment activated")
        print(f"📋 Required: conda activate anyloc")
        raise

if __name__ == "__main__":
    main()
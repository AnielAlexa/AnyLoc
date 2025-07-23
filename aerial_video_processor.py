#!/usr/bin/env python3
"""
Aerial Video Processor for GPS Estimation

This module processes aerial video files to estimate GPS coordinates using
satellite image matching with the AnyLoc drone navigation system.

Usage:
    # Process aerial video against satellite database
    python3 aerial_video_processor.py --video aerial_flight.mp4 --database ./satellite_patches --output ./flight_results

    # Process with custom frame sampling
    python3 aerial_video_processor.py --video flight.mp4 --database ./satellite_db --frame-interval 30 --confidence-threshold 0.4
"""

import os
import sys
from pathlib import Path
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

import cv2
import numpy as np
import torch
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from drone_navigation_system import DroneNavigationSystem

@dataclass
class FrameResult:
    """Result for a single video frame"""
    frame_number: int
    timestamp: float
    gps_estimate: Optional[Dict[str, float]]
    confidence: float
    match_name: str
    processing_time: float
    is_valid_match: bool

class AerialVideoProcessor:
    """Process aerial video files for GPS estimation"""
    
    def __init__(self, database_path: str, output_dir: str = "./flight_results"):
        self.database_path = database_path
        self.output_dir = output_dir
        self.results_file = f"{output_dir}/flight_trajectory.json"
        self.frames_dir = f"{output_dir}/processed_frames"
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Initialize drone navigation system
        print("🔧 Initializing drone navigation system...")
        self.nav_system = DroneNavigationSystem(database_path)
        self.nav_system.load_satellite_database()
        
        print(f"✅ Loaded satellite database with {len(self.nav_system.database)} images")
    
    def process_video(self, video_path: str, frame_interval: int = 30, 
                     confidence_threshold: float = 0.3, max_frames: Optional[int] = None,
                     save_processed_frames: bool = False) -> List[FrameResult]:
        """
        Process aerial video to estimate GPS trajectory
        
        Args:
            video_path: Path to aerial video file
            frame_interval: Process every Nth frame (30 = once per second at 30fps)
            confidence_threshold: Minimum confidence for valid GPS estimates
            max_frames: Maximum number of frames to process (None = all)
            save_processed_frames: Save processed frames to disk
        
        Returns:
            List of FrameResult objects with GPS estimates
        """
        print("🎥 PROCESSING AERIAL VIDEO")
        print("=" * 50)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📹 Video info:")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Frame interval: {frame_interval} (processing every {frame_interval/fps:.1f} seconds)")
        
        # Calculate frames to process
        frames_to_process = list(range(0, total_frames, frame_interval))
        if max_frames:
            frames_to_process = frames_to_process[:max_frames]
        
        print(f"📊 Will process {len(frames_to_process)} frames")
        
        results = []
        start_time = time.time()
        
        for i, frame_num in enumerate(frames_to_process):
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                print(f"   ⚠️  Could not read frame {frame_num}")
                continue
            
            frame_timestamp = frame_num / fps if fps > 0 else 0
            frame_start_time = time.time()
            
            # Process frame with navigation system
            try:
                match_result = self.nav_system.match_drone_frame(frame)
                
                # Create result object
                result = FrameResult(
                    frame_number=frame_num,
                    timestamp=frame_timestamp,
                    gps_estimate=match_result.get('gps_estimate'),
                    confidence=match_result.get('similarity', 0.0),
                    match_name=match_result.get('best_match_name', ''),
                    processing_time=time.time() - frame_start_time,
                    is_valid_match=match_result.get('similarity', 0.0) >= confidence_threshold
                )
                
                results.append(result)
                
                # Save frame if requested and match is valid
                if save_processed_frames and result.is_valid_match:
                    frame_filename = f"frame_{frame_num:06d}_conf{result.confidence:.3f}.jpg"
                    frame_path = f"{self.frames_dir}/{frame_filename}"
                    cv2.imwrite(frame_path, frame)
                
                # Progress update
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    eta = avg_time * (len(frames_to_process) - i - 1)
                    
                    valid_matches = sum(1 for r in results if r.is_valid_match)
                    print(f"   ⏳ Processed {i+1}/{len(frames_to_process)} frames "
                          f"({valid_matches} valid matches, ETA: {eta:.1f}s)")
                
            except Exception as e:
                print(f"   ❌ Error processing frame {frame_num}: {e}")
                continue
        
        cap.release()
        
        # Summary statistics
        total_time = time.time() - start_time
        valid_matches = sum(1 for r in results if r.is_valid_match)
        avg_confidence = np.mean([r.confidence for r in results]) if results else 0
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total frames processed: {len(results)}")
        print(f"   Valid matches: {valid_matches} ({valid_matches/len(results)*100:.1f}%)")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Total processing time: {total_time:.1f}s")
        print(f"   Average time per frame: {total_time/len(results):.3f}s")
        
        # Save results
        self._save_results(results, video_path)
        
        return results
    
    def _save_results(self, results: List[FrameResult], video_path: str):
        """Save processing results to JSON file"""
        
        # Convert results to JSON-serializable format
        results_data = {
            "video_info": {
                "video_path": video_path,
                "processing_timestamp": datetime.now().isoformat(),
                "total_frames_processed": len(results),
                "valid_matches": sum(1 for r in results if r.is_valid_match)
            },
            "trajectory": []
        }
        
        for result in results:
            frame_data = {
                "frame_number": result.frame_number,
                "timestamp": result.timestamp,
                "confidence": result.confidence,
                "match_name": result.match_name,
                "processing_time": result.processing_time,
                "is_valid_match": result.is_valid_match
            }
            
            # Add GPS data if available
            if result.gps_estimate:
                frame_data["gps"] = result.gps_estimate
            
            results_data["trajectory"].append(frame_data)
        
        # Save to file
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to: {self.results_file}")
    
    def create_trajectory_visualization(self, results: List[FrameResult], output_path: str = None):
        """Create visualization of estimated flight trajectory"""
        
        if output_path is None:
            output_path = f"{self.output_dir}/trajectory_plot.png"
        
        try:
            import matplotlib.pyplot as plt
            
            # Extract valid GPS coordinates
            valid_results = [r for r in results if r.is_valid_match and r.gps_estimate]
            
            if not valid_results:
                print("⚠️  No valid GPS estimates for trajectory visualization")
                return
            
            lats = [r.gps_estimate['lat'] for r in valid_results]
            lons = [r.gps_estimate['lon'] for r in valid_results]
            confidences = [r.confidence for r in valid_results]
            timestamps = [r.timestamp for r in valid_results]
            
            # Create trajectory plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # GPS trajectory plot
            scatter = ax1.scatter(lons, lats, c=confidences, cmap='viridis', 
                                s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax1.plot(lons, lats, 'r-', alpha=0.3, linewidth=1)
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_title(f'Estimated Flight Trajectory ({len(valid_results)} points)')
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Match Confidence')
            
            # Confidence over time
            ax2.plot(timestamps, confidences, 'b-', linewidth=2, alpha=0.7)
            ax2.axhline(y=self.nav_system.confidence_threshold, color='r', 
                       linestyle='--', alpha=0.7, label='Confidence Threshold')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Match Confidence')
            ax2.set_title('Match Confidence Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Trajectory visualization saved to: {output_path}")
            
            # Create summary statistics
            self._create_statistics_report(valid_results)
            
        except ImportError:
            print("⚠️  Matplotlib not available, skipping trajectory visualization")
    
    def _create_statistics_report(self, valid_results: List[FrameResult]):
        """Create detailed statistics report"""
        
        if not valid_results:
            return
        
        report_path = f"{self.output_dir}/statistics_report.txt"
        
        lats = [r.gps_estimate['lat'] for r in valid_results]
        lons = [r.gps_estimate['lon'] for r in valid_results]
        confidences = [r.confidence for r in valid_results]
        
        # Calculate statistics
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        
        # Rough distance calculation (not accounting for Earth curvature)
        lat_km = lat_range * 111  # 1 degree lat ≈ 111 km
        lon_km = lon_range * 111 * np.cos(np.radians(np.mean(lats)))  # longitude varies with latitude
        
        with open(report_path, 'w') as f:
            f.write("AERIAL VIDEO PROCESSING STATISTICS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Flight Area Coverage:\n")
            f.write(f"  Latitude range: {min(lats):.6f} to {max(lats):.6f} ({lat_range:.6f}°, ~{lat_km:.1f} km)\n")
            f.write(f"  Longitude range: {min(lons):.6f} to {max(lons):.6f} ({lon_range:.6f}°, ~{lon_km:.1f} km)\n\n")
            
            f.write(f"Match Quality:\n")
            f.write(f"  Valid GPS estimates: {len(valid_results)}\n")
            f.write(f"  Average confidence: {np.mean(confidences):.3f}\n")
            f.write(f"  Confidence std dev: {np.std(confidences):.3f}\n")
            f.write(f"  Min confidence: {min(confidences):.3f}\n")
            f.write(f"  Max confidence: {max(confidences):.3f}\n\n")
            
            f.write(f"Processing Performance:\n")
            f.write(f"  Average processing time per frame: {np.mean([r.processing_time for r in valid_results]):.3f}s\n")
            f.write(f"  Total processing time: {sum(r.processing_time for r in valid_results):.1f}s\n")
        
        print(f"📊 Statistics report saved to: {report_path}")
    
    def export_kml(self, results: List[FrameResult], output_path: str = None):
        """Export trajectory as KML file for Google Earth"""
        
        if output_path is None:
            output_path = f"{self.output_dir}/flight_trajectory.kml"
        
        valid_results = [r for r in results if r.is_valid_match and r.gps_estimate]
        
        if not valid_results:
            print("⚠️  No valid GPS estimates for KML export")
            return
        
        kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Aerial Video Flight Trajectory</name>
    <description>Estimated GPS trajectory from aerial video matching</description>
    
    <Style id="flightPath">
      <LineStyle>
        <color>ff0000ff</color>
        <width>3</width>
      </LineStyle>
    </Style>
    
    <Style id="waypoint">
      <IconStyle>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/pal4/icon57.png</href>
        </Icon>
      </IconStyle>
    </Style>
    
    <Placemark>
      <name>Flight Path</name>
      <styleUrl>#flightPath</styleUrl>
      <LineString>
        <coordinates>
'''
        
        # Add coordinates
        for result in valid_results:
            lon = result.gps_estimate['lon']
            lat = result.gps_estimate['lat']
            alt = result.gps_estimate.get('altitude', 100)
            kml_content += f"          {lon},{lat},{alt}\n"
        
        kml_content += '''        </coordinates>
      </LineString>
    </Placemark>
'''
        
        # Add waypoints for high-confidence matches
        high_conf_results = [r for r in valid_results if r.confidence > 0.7]
        for i, result in enumerate(high_conf_results[::10]):  # Every 10th high-confidence point
            lon = result.gps_estimate['lon']
            lat = result.gps_estimate['lat']
            alt = result.gps_estimate.get('altitude', 100)
            
            kml_content += f'''    <Placemark>
      <name>Frame {result.frame_number}</name>
      <description>
        Time: {result.timestamp:.1f}s
        Confidence: {result.confidence:.3f}
        Match: {result.match_name}
      </description>
      <styleUrl>#waypoint</styleUrl>
      <Point>
        <coordinates>{lon},{lat},{alt}</coordinates>
      </Point>
    </Placemark>
'''
        
        kml_content += '''  </Document>
</kml>'''
        
        with open(output_path, 'w') as f:
            f.write(kml_content)
        
        print(f"🌍 KML file saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process aerial video for GPS estimation")
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to aerial video file')
    parser.add_argument('--database', type=str, required=True,
                       help='Path to satellite database directory')
    parser.add_argument('--output', type=str, default='./flight_results',
                       help='Output directory for results')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Process every Nth frame (30 = once per second at 30fps)')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                       help='Minimum confidence for valid GPS estimates')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save processed frames to disk')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip creating trajectory visualization')
    parser.add_argument('--export-kml', action='store_true',
                       help='Export trajectory as KML file for Google Earth')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.database):
        print(f"❌ Error: Database directory not found: {args.database}")
        return
    
    # Create processor
    processor = AerialVideoProcessor(args.database, args.output)
    
    # Process video
    results = processor.process_video(
        args.video, 
        frame_interval=args.frame_interval,
        confidence_threshold=args.confidence_threshold,
        max_frames=args.max_frames,
        save_processed_frames=args.save_frames
    )
    
    # Create visualizations
    if not args.no_visualization:
        processor.create_trajectory_visualization(results)
    
    if args.export_kml:
        processor.export_kml(results)
    
    print(f"\n🎉 Video processing complete!")
    print(f"📁 Results directory: {args.output}")
    print(f"📊 Valid GPS estimates: {sum(1 for r in results if r.is_valid_match)}/{len(results)}")

if __name__ == "__main__":
    main()
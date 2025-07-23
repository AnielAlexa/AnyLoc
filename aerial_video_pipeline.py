#!/usr/bin/env python3
"""
Complete Aerial Video GPS Estimation Pipeline

This script provides an end-to-end workflow for estimating GPS trajectories
from aerial video using satellite image matching.

Workflow:
1. Process satellite imagery (from large image or download tiles)
2. Build AnyLoc feature database
3. Process aerial video against database
4. Generate comprehensive results and visualizations

Usage Examples:

    # Complete pipeline with large satellite image
    python3 aerial_video_pipeline.py \
        --video flight.mp4 \
        --satellite-image large_map.png \
        --bounds "40.750,-73.990,40.760,-73.980" \
        --output ./flight_analysis

    # Complete pipeline with OSM tile download
    python3 aerial_video_pipeline.py \
        --video aerial_video.mp4 \
        --download-osm \
        --bounds "40.750,-73.990,40.760,-73.980" \
        --zoom 18 \
        --output ./results

    # Quick analysis with existing database
    python3 aerial_video_pipeline.py \
        --video flight.mp4 \
        --existing-database ./satellite_db \
        --output ./quick_results
"""

import os
import sys
from pathlib import Path
import argparse
import json
import time
import warnings
warnings.filterwarnings("ignore")

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

from google_maps_processor import GoogleMapsProcessor
from aerial_video_processor import AerialVideoProcessor
from drone_navigation_system import DroneNavigationSystem

class AerialVideoPipeline:
    """Complete pipeline for aerial video GPS estimation"""
    
    def __init__(self, output_dir: str = "./pipeline_results"):
        self.output_dir = output_dir
        self.satellite_dir = f"{output_dir}/satellite_patches"
        self.database_dir = f"{output_dir}/anyloc_database"
        self.video_results_dir = f"{output_dir}/video_analysis"
        
        # Create directories
        for dir_path in [self.output_dir, self.satellite_dir, self.database_dir, self.video_results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.pipeline_log = []
    
    def log_step(self, step: str, status: str = "started", details: str = ""):
        """Log pipeline step"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "status": status,
            "details": details
        }
        self.pipeline_log.append(log_entry)
        
        status_emoji = "🔄" if status == "started" else "✅" if status == "completed" else "❌"
        print(f"{status_emoji} [{timestamp}] {step}: {status}")
        if details:
            print(f"   {details}")
    
    def run_complete_pipeline(self, video_path: str, satellite_image: str = None, 
                            bounds: tuple = None, download_osm: bool = False,
                            existing_database: str = None, **kwargs):
        """
        Run the complete aerial video GPS estimation pipeline
        
        Args:
            video_path: Path to aerial video file
            satellite_image: Path to large satellite image (if processing from image)
            bounds: GPS bounds tuple (lat_min, lon_min, lat_max, lon_max)
            download_osm: Download OSM tiles instead of processing image
            existing_database: Use existing AnyLoc database
            **kwargs: Additional parameters for processing steps
        """
        
        print("🚁 AERIAL VIDEO GPS ESTIMATION PIPELINE")
        print("=" * 60)
        
        pipeline_start_time = time.time()
        
        # Step 1: Prepare satellite database
        if existing_database and os.path.exists(existing_database):
            self.log_step("Using existing database", "completed", f"Database: {existing_database}")
            database_path = existing_database
        else:
            database_path = self._prepare_satellite_database(
                satellite_image, bounds, download_osm, **kwargs
            )
        
        # Step 2: Process aerial video
        video_results = self._process_aerial_video(video_path, database_path, **kwargs)
        
        # Step 3: Generate comprehensive analysis
        self._generate_comprehensive_analysis(video_results, video_path, **kwargs)
        
        # Step 4: Create final report
        self._create_final_report(video_results, pipeline_start_time)
        
        total_time = time.time() - pipeline_start_time
        self.log_step("Complete pipeline", "completed", f"Total time: {total_time:.1f}s")
        
        return video_results
    
    def _prepare_satellite_database(self, satellite_image: str, bounds: tuple, 
                                  download_osm: bool, **kwargs) -> str:
        """Prepare satellite database for matching"""
        
        self.log_step("Preparing satellite database", "started")
        
        # Step 1a: Process satellite imagery
        maps_processor = GoogleMapsProcessor(self.satellite_dir)
        
        if download_osm:
            self.log_step("Downloading OSM tiles", "started")
            gps_metadata = maps_processor.download_osm_tiles(
                bounds, 
                zoom=kwargs.get('zoom', 18),
                delay=kwargs.get('tile_delay', 0.1)
            )
            self.log_step("Downloading OSM tiles", "completed", f"Downloaded {len(gps_metadata)} tiles")
        else:
            if not satellite_image or not os.path.exists(satellite_image):
                raise ValueError("Satellite image required when not downloading OSM tiles")
            
            self.log_step("Processing satellite image", "started")
            gps_metadata = maps_processor.process_large_satellite_image(
                satellite_image,
                bounds,
                patch_size=kwargs.get('patch_size', 512),
                overlap=kwargs.get('overlap', 64)
            )
            self.log_step("Processing satellite image", "completed", f"Created {len(gps_metadata)} patches")
        
        # Create grid visualization
        maps_processor.create_coordinate_grid_overlay(gps_metadata)
        
        # Step 1b: Build AnyLoc database
        self.log_step("Building AnyLoc database", "started")
        
        nav_system = DroneNavigationSystem(self.database_dir)
        nav_system.build_satellite_database(
            maps_processor.patches_dir,
            maps_processor.metadata_file
        )
        
        self.log_step("Building AnyLoc database", "completed")
        
        return self.database_dir
    
    def _process_aerial_video(self, video_path: str, database_path: str, **kwargs):
        """Process aerial video against satellite database"""
        
        self.log_step("Processing aerial video", "started")
        
        video_processor = AerialVideoProcessor(database_path, self.video_results_dir)
        
        results = video_processor.process_video(
            video_path,
            frame_interval=kwargs.get('frame_interval', 30),
            confidence_threshold=kwargs.get('confidence_threshold', 0.3),
            max_frames=kwargs.get('max_frames', None),
            save_processed_frames=kwargs.get('save_frames', False)
        )
        
        self.log_step("Processing aerial video", "completed", f"Processed {len(results)} frames")
        
        # Generate visualizations
        if not kwargs.get('no_visualization', False):
            self.log_step("Creating visualizations", "started")
            video_processor.create_trajectory_visualization(results)
            self.log_step("Creating visualizations", "completed")
        
        if kwargs.get('export_kml', True):
            self.log_step("Exporting KML", "started")
            video_processor.export_kml(results)
            self.log_step("Exporting KML", "completed")
        
        return results
    
    def _generate_comprehensive_analysis(self, video_results, video_path: str, **kwargs):
        """Generate comprehensive analysis of results"""
        
        self.log_step("Generating comprehensive analysis", "started")
        
        valid_results = [r for r in video_results if r.is_valid_match]
        
        if not valid_results:
            self.log_step("Generating comprehensive analysis", "completed", "No valid matches found")
            return
        
        # Create comprehensive analysis
        analysis = {
            "video_info": {
                "path": video_path,
                "total_frames_processed": len(video_results),
                "valid_gps_estimates": len(valid_results),
                "success_rate": len(valid_results) / len(video_results) * 100
            },
            "spatial_analysis": self._analyze_spatial_coverage(valid_results),
            "temporal_analysis": self._analyze_temporal_consistency(valid_results),
            "quality_metrics": self._analyze_match_quality(valid_results)
        }
        
        # Save comprehensive analysis
        analysis_file = f"{self.output_dir}/comprehensive_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.log_step("Generating comprehensive analysis", "completed", f"Saved to {analysis_file}")
    
    def _analyze_spatial_coverage(self, valid_results):
        """Analyze spatial coverage of the flight"""
        if not valid_results:
            return {}
        
        lats = [r.gps_estimate['lat'] for r in valid_results]
        lons = [r.gps_estimate['lon'] for r in valid_results]
        
        import numpy as np
        
        return {
            "flight_bounds": {
                "lat_min": min(lats),
                "lat_max": max(lats),
                "lon_min": min(lons),
                "lon_max": max(lons)
            },
            "coverage_area": {
                "lat_range_deg": max(lats) - min(lats),
                "lon_range_deg": max(lons) - min(lons),
                "approximate_area_km2": ((max(lats) - min(lats)) * 111) * 
                                       ((max(lons) - min(lons)) * 111 * np.cos(np.radians(np.mean(lats))))
            },
            "centroid": {
                "lat": np.mean(lats),
                "lon": np.mean(lons)
            }
        }
    
    def _analyze_temporal_consistency(self, valid_results):
        """Analyze temporal consistency of matches"""
        if len(valid_results) < 2:
            return {}
        
        import numpy as np
        
        # Calculate time gaps between valid matches
        timestamps = [r.timestamp for r in valid_results]
        time_gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        return {
            "flight_duration": max(timestamps) - min(timestamps),
            "average_time_between_matches": np.mean(time_gaps),
            "max_time_gap": max(time_gaps),
            "temporal_density": len(valid_results) / (max(timestamps) - min(timestamps))
        }
    
    def _analyze_match_quality(self, valid_results):
        """Analyze quality metrics of matches"""
        if not valid_results:
            return {}
        
        import numpy as np
        
        confidences = [r.confidence for r in valid_results]
        processing_times = [r.processing_time for r in valid_results]
        
        return {
            "confidence_stats": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "high_confidence_ratio": sum(1 for c in confidences if c > 0.7) / len(confidences)
            },
            "performance_stats": {
                "mean_processing_time": np.mean(processing_times),
                "total_processing_time": sum(processing_times),
                "fps_equivalent": 1.0 / np.mean(processing_times)
            }
        }
    
    def _create_final_report(self, video_results, pipeline_start_time):
        """Create final comprehensive report"""
        
        self.log_step("Creating final report", "started")
        
        report_path = f"{self.output_dir}/FINAL_REPORT.md"
        total_time = time.time() - pipeline_start_time
        
        valid_results = [r for r in video_results if r.is_valid_match]
        success_rate = len(valid_results) / len(video_results) * 100 if video_results else 0
        
        with open(report_path, 'w') as f:
            f.write("# Aerial Video GPS Estimation Report\n\n")
            
            f.write("## Pipeline Summary\n")
            f.write(f"- **Total Processing Time**: {total_time:.1f} seconds\n")
            f.write(f"- **Frames Processed**: {len(video_results)}\n")
            f.write(f"- **Valid GPS Estimates**: {len(valid_results)}\n")
            f.write(f"- **Success Rate**: {success_rate:.1f}%\n\n")
            
            f.write("## Output Files\n")
            f.write(f"- **Satellite Patches**: `{self.satellite_dir}/`\n")
            f.write(f"- **AnyLoc Database**: `{self.database_dir}/`\n")
            f.write(f"- **Video Analysis**: `{self.video_results_dir}/`\n")
            f.write(f"- **Flight Trajectory**: `{self.video_results_dir}/flight_trajectory.json`\n")
            f.write(f"- **Trajectory Plot**: `{self.video_results_dir}/trajectory_plot.png`\n")
            f.write(f"- **KML Export**: `{self.video_results_dir}/flight_trajectory.kml`\n\n")
            
            if valid_results:
                import numpy as np
                lats = [r.gps_estimate['lat'] for r in valid_results]
                lons = [r.gps_estimate['lon'] for r in valid_results]
                confidences = [r.confidence for r in valid_results]
                
                f.write("## Flight Statistics\n")
                f.write(f"- **Flight Area**: {min(lats):.6f}°N to {max(lats):.6f}°N, {min(lons):.6f}°E to {max(lons):.6f}°E\n")
                f.write(f"- **Average Confidence**: {np.mean(confidences):.3f}\n")
                f.write(f"- **High Confidence Matches**: {sum(1 for c in confidences if c > 0.7)} ({sum(1 for c in confidences if c > 0.7)/len(confidences)*100:.1f}%)\n\n")
            
            f.write("## Usage Instructions\n")
            f.write("1. **View Trajectory**: Open `trajectory_plot.png` for flight path visualization\n")
            f.write("2. **Google Earth**: Import `flight_trajectory.kml` into Google Earth\n")
            f.write("3. **Detailed Data**: Check `flight_trajectory.json` for frame-by-frame results\n")
            f.write("4. **Analysis**: See `comprehensive_analysis.json` for detailed statistics\n\n")
            
            f.write("## Pipeline Log\n")
            for log_entry in self.pipeline_log:
                f.write(f"- **{log_entry['timestamp']}**: {log_entry['step']} - {log_entry['status']}\n")
                if log_entry['details']:
                    f.write(f"  {log_entry['details']}\n")
        
        self.log_step("Creating final report", "completed", f"Report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Complete aerial video GPS estimation pipeline")
    
    # Required arguments
    parser.add_argument('--video', type=str, required=True,
                       help='Path to aerial video file')
    parser.add_argument('--bounds', type=str, required=True,
                       help='GPS bounds as "lat_min,lon_min,lat_max,lon_max"')
    parser.add_argument('--output', type=str, default='./pipeline_results',
                       help='Output directory for all results')
    
    # Satellite imagery options (mutually exclusive)
    satellite_group = parser.add_mutually_exclusive_group()
    satellite_group.add_argument('--satellite-image', type=str,
                                help='Path to large satellite image to process')
    satellite_group.add_argument('--download-osm', action='store_true',
                                help='Download tiles from OpenStreetMap')
    satellite_group.add_argument('--existing-database', type=str,
                                help='Use existing AnyLoc database')
    
    # Satellite processing options
    parser.add_argument('--patch-size', type=int, default=512,
                       help='Size of satellite patches in pixels')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Overlap between patches in pixels')
    parser.add_argument('--zoom', type=int, default=18,
                       help='Zoom level for OSM tile download')
    parser.add_argument('--tile-delay', type=float, default=0.1,
                       help='Delay between tile downloads (seconds)')
    
    # Video processing options
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Process every Nth frame')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                       help='Minimum confidence for valid GPS estimates')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save processed video frames')
    
    # Output options
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip creating trajectory visualization')
    parser.add_argument('--no-kml', action='store_false', dest='export_kml',
                       help='Skip KML export')
    
    args = parser.parse_args()
    
    # Parse bounds
    try:
        bounds = tuple(map(float, args.bounds.split(',')))
        if len(bounds) != 4:
            raise ValueError("Bounds must have 4 values")
    except:
        print("❌ Error: Bounds must be in format 'lat_min,lon_min,lat_max,lon_max'")
        return
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return
    
    if not args.existing_database and not args.download_osm and not args.satellite_image:
        print("❌ Error: Must specify --satellite-image, --download-osm, or --existing-database")
        return
    
    if args.satellite_image and not os.path.exists(args.satellite_image):
        print(f"❌ Error: Satellite image not found: {args.satellite_image}")
        return
    
    # Create pipeline
    pipeline = AerialVideoPipeline(args.output)
    
    # Run complete pipeline
    try:
        # Create kwargs excluding parameters already passed explicitly
        kwargs = {k: v for k, v in vars(args).items() 
                 if k not in ['video', 'satellite_image', 'download_osm', 'existing_database', 'bounds', 'output']}
        
        results = pipeline.run_complete_pipeline(
            video_path=args.video,
            satellite_image=args.satellite_image,
            bounds=bounds,
            download_osm=args.download_osm,
            existing_database=args.existing_database,
            **kwargs
        )
        
        print(f"\n🎉 PIPELINE COMPLETE!")
        print(f"📁 All results saved to: {args.output}")
        print(f"📊 GPS estimates: {sum(1 for r in results if r.is_valid_match)}/{len(results)}")
        print(f"📄 See FINAL_REPORT.md for detailed summary")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
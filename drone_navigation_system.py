#!/usr/bin/env python3
"""
Practical Drone Navigation System using Satellite-Drone Matching

This system demonstrates the complete workflow for drone navigation:
1. Offline: Cache satellite database with GPS coordinates
2. Runtime: Real-time drone video matching for localization

Usage:
    # 1. Build satellite database
    python3 drone_navigation_system.py --mode build-database --area "lat1,lon1,lat2,lon2"
    
    # 2. Run real-time drone matching
    python3 drone_navigation_system.py --mode drone-navigation --camera 0
"""

import os
import sys
from pathlib import Path
import time
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from utilities import DinoV2ExtractFeatures, VLAD, seed_everything
from configs import device

class DroneNavigationSystem:
    """Complete drone navigation system using satellite-drone matching"""
    
    def __init__(self, database_path="./drone_satellite_db"):
        self.database_path = database_path
        self.database_file = f"{database_path}/satellite_database.json"
        self.features_file = f"{database_path}/satellite_features.pt"
        
        # Optimized settings for real-time performance
        self.model_name = "dinov2_vits14"  # Fast model
        self.resolution = (224, 224)       # Optimal resolution
        self.vlad_clusters = 16            # Good balance
        self.layer = 11                    # vits14 layer
        self.facet = "value"
        
        # GPS and navigation
        self.current_gps_estimate = None
        self.confidence_threshold = 0.3    # Minimum similarity for valid match
        
        os.makedirs(database_path, exist_ok=True)
        
    def build_satellite_database(self, satellite_images_dir, gps_metadata_file=None):
        """
        Build satellite database with GPS coordinates
        
        Args:
            satellite_images_dir: Directory with satellite images
            gps_metadata_file: JSON file with image->GPS mapping
        """
        print("🛰️  BUILDING SATELLITE DATABASE")
        print("=" * 50)
        
        # Load models
        self._load_feature_extractor()
        
        # Find satellite images
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            image_files.extend(Path(satellite_images_dir).glob(f"*{ext}"))
        
        if not image_files:
            raise ValueError(f"No images found in {satellite_images_dir}")
        
        print(f"📊 Found {len(image_files)} satellite images")
        
        # Load GPS metadata if provided
        gps_data = {}
        if gps_metadata_file and os.path.exists(gps_metadata_file):
            with open(gps_metadata_file, 'r') as f:
                gps_data = json.load(f)
            print(f"📍 Loaded GPS data for {len(gps_data)} images")
        
        # Extract features from all satellite images
        database = {}
        all_features = []
        
        for i, img_path in enumerate(image_files):
            img_name = img_path.stem
            print(f"   🔄 Processing {img_name} ({i+1}/{len(image_files)})")
            
            try:
                # Load and process image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Extract features
                img_tensor = self._preprocess_image(image)
                with torch.no_grad():
                    features = self.extractor(img_tensor)
                    features = features.squeeze(0).cpu()
                
                # Get GPS coordinates
                gps_coords = gps_data.get(img_name, {
                    'lat': None, 
                    'lon': None,
                    'altitude': None
                })
                
                # Store in database
                database[img_name] = {
                    'image_path': str(img_path),
                    'gps': gps_coords,
                    'feature_idx': len(all_features)
                }
                
                all_features.append(features)
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        if not all_features:
            raise ValueError("No features extracted from satellite images")
        
        # Create VLAD vocabulary
        print(f"🧠 Creating VLAD vocabulary from {len(all_features)} images...")
        all_features_combined = torch.cat(all_features, dim=0)
        
        vlad = VLAD(num_clusters=self.vlad_clusters, desc_dim=384, cache_dir=None)
        vlad.fit(all_features_combined)
        
        # Generate VLAD descriptors for database
        print("📊 Generating VLAD descriptors...")
        satellite_vlads = []
        
        for features in all_features:
            vlad_desc = vlad.generate(features)
            satellite_vlads.append(vlad_desc)
        
        # Save database
        print("💾 Saving satellite database...")
        
        # Save metadata
        with open(self.database_file, 'w') as f:
            json.dump(database, f, indent=2)
        
        # Save features and VLAD
        torch.save({
            'vlad_centers': vlad.c_centers,
            'vlad_clusters': vlad.num_clusters,
            'vlad_dim': vlad.desc_dim,
            'satellite_vlads': torch.stack(satellite_vlads),
            'kmeans_centroids': vlad.c_centers,  # Save for reconstruction
            'feature_extraction_settings': {
                'model_name': self.model_name,
                'resolution': self.resolution,
                'layer': self.layer,
                'facet': self.facet
            }
        }, self.features_file)
        
        print(f"✅ Database built successfully!")
        print(f"   📊 {len(database)} satellite images processed")
        print(f"   💾 Saved to: {self.database_path}")
        print(f"   🎯 VLAD descriptors: {len(satellite_vlads)} x {satellite_vlads[0].shape[0]}D")
        
        return database
    
    def load_satellite_database(self):
        """Load pre-built satellite database"""
        
        if not os.path.exists(self.database_file) or not os.path.exists(self.features_file):
            raise FileNotFoundError(f"Database not found. Run build-database first.")
        
        print("📂 Loading satellite database...")
        
        # Load metadata
        with open(self.database_file, 'r') as f:
            self.database = json.load(f)
        
        # Load features
        data = torch.load(self.features_file, map_location='cpu')
        
        # Setup VLAD
        self.vlad = VLAD(
            num_clusters=data['vlad_clusters'], 
            desc_dim=data['vlad_dim'], 
            cache_dir=None
        )
        self.vlad.c_centers = data['vlad_centers']
        self.vlad.is_fitted = True
        
        # Initialize kmeans object for VLAD
        from fast_pytorch_kmeans import KMeans
        self.vlad.kmeans = KMeans(n_clusters=data['vlad_clusters'], mode='cosine')
        if 'kmeans_centroids' in data:
            self.vlad.kmeans.centroids = data['kmeans_centroids']
        else:
            self.vlad.kmeans.centroids = data['vlad_centers']
        
        # Load satellite VLAD descriptors
        self.satellite_vlads = data['satellite_vlads']
        self.satellite_names = list(self.database.keys())
        
        print(f"✅ Database loaded: {len(self.database)} satellite images")
        
        # Load feature extractor
        self._load_feature_extractor()
    
    def _load_feature_extractor(self):
        """Load DINOv2 feature extractor"""
        print(f"🔧 Loading {self.model_name}...")
        
        self.extractor = DinoV2ExtractFeatures(
            self.model_name, 
            layer=self.layer, 
            facet=self.facet, 
            device=device
        )
        
        # Jetson optimizations
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if 'Orin' in gpu_name or 'Jetson' in gpu_name:
                print("   🔧 Applying Jetson optimizations...")
                try:
                    self.extractor.dino_model = self.extractor.dino_model.half()
                    print("   ✅ FP16 precision enabled")
                except:
                    print("   ⚠️  FP16 not supported")
    
    def _preprocess_image(self, image):
        """Preprocess image for DINOv2"""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize
        image_resized = cv2.resize(image_rgb, self.resolution)
        
        # Convert to tensor and normalize
        img_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0).to(device, dtype=torch.float32)
    
    def match_drone_frame(self, drone_image):
        """
        Match drone frame to satellite database
        
        Returns:
            dict: Match result with GPS estimate and confidence
        """
        start_time = time.time()
        
        # Extract features from drone image
        img_tensor = self._preprocess_image(drone_image)
        
        with torch.no_grad():
            features = self.extractor(img_tensor)
            features = features.squeeze(0).cpu()
        
        extraction_time = time.time()
        
        # Generate VLAD descriptor
        vlad_desc = self.vlad.generate(features)
        vlad_time = time.time()
        
        # Find best match in satellite database
        similarities = F.cosine_similarity(
            vlad_desc.unsqueeze(0), 
            self.satellite_vlads, 
            dim=1
        )
        
        best_idx = similarities.argmax().item()
        best_similarity = similarities[best_idx].item()
        best_match_name = self.satellite_names[best_idx]
        
        search_time = time.time()
        
        # Get GPS coordinates
        match_data = self.database[best_match_name]
        gps_coords = match_data['gps']
        
        # Create result
        result = {
            'matched_satellite': best_match_name,
            'satellite_image_path': match_data['image_path'],
            'similarity': best_similarity,
            'confidence': 'HIGH' if best_similarity > 0.5 else 'MEDIUM' if best_similarity > self.confidence_threshold else 'LOW',
            'gps_estimate': gps_coords,
            'is_valid_match': best_similarity > self.confidence_threshold,
            'timing': {
                'extraction_ms': (extraction_time - start_time) * 1000,
                'vlad_ms': (vlad_time - extraction_time) * 1000,
                'search_ms': (search_time - vlad_time) * 1000,
                'total_ms': (search_time - start_time) * 1000
            }
        }
        
        # Update current GPS estimate if valid
        if result['is_valid_match'] and gps_coords['lat'] and gps_coords['lon']:
            self.current_gps_estimate = gps_coords
        
        return result
    
    def run_drone_navigation(self, camera_id=0, save_results=True):
        """
        Run real-time drone navigation with live video
        
        Args:
            camera_id: Camera ID for video capture
            save_results: Save navigation log
        """
        print("🚁 STARTING DRONE NAVIGATION SYSTEM")
        print("=" * 50)
        print("Press 'q' to quit, 's' to save frame, 'g' to show GPS")
        
        # Load database
        self.load_satellite_database()
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Navigation log
        navigation_log = []
        frame_count = 0
        
        print("🎬 Navigation started - showing live drone feed")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                try:
                    # Match frame to satellite database
                    result = self.match_drone_frame(frame)
                    
                    # Add to navigation log
                    if save_results:
                        log_entry = {
                            'timestamp': time.time(),
                            'frame_number': frame_count,
                            'result': result
                        }
                        navigation_log.append(log_entry)
                    
                    # Display results on frame
                    self._draw_navigation_info(frame, result)
                    
                    # Print status
                    if frame_count % 30 == 0:  # Every 10 processed frames
                        gps = result['gps_estimate']
                        gps_str = f"({gps['lat']:.6f}, {gps['lon']:.6f})" if gps['lat'] else "Unknown"
                        print(f"Frame {frame_count}: {result['confidence']} match to {result['matched_satellite']} "
                              f"GPS: {gps_str} ({result['timing']['total_ms']:.1f}ms)")
                
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
            
            # Show frame
            cv2.imshow('Drone Navigation', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                cv2.imwrite(f"drone_frame_{timestamp}.jpg", frame)
                print(f"Saved frame: drone_frame_{timestamp}.jpg")
            elif key == ord('g'):
                # Print current GPS estimate
                if self.current_gps_estimate and self.current_gps_estimate['lat']:
                    gps = self.current_gps_estimate
                    print(f"Current GPS estimate: {gps['lat']:.6f}, {gps['lon']:.6f}")
                else:
                    print("No valid GPS estimate available")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save navigation log
        if save_results and navigation_log:
            log_file = f"{self.database_path}/navigation_log_{int(time.time())}.json"
            with open(log_file, 'w') as f:
                json.dump(navigation_log, f, indent=2)
            print(f"💾 Navigation log saved: {log_file}")
        
        print(f"✅ Navigation session completed ({len(navigation_log)} matches logged)")
    
    def _draw_navigation_info(self, frame, result):
        """Draw navigation information on frame"""
        # Prepare info text
        info_lines = [
            f"Match: {result['matched_satellite']}",
            f"Confidence: {result['confidence']} ({result['similarity']:.3f})",
            f"Time: {result['timing']['total_ms']:.0f}ms"
        ]
        
        # Add GPS if available
        gps = result['gps_estimate']
        if gps['lat'] and gps['lon']:
            info_lines.append(f"GPS: {gps['lat']:.6f}, {gps['lon']:.6f}")
        else:
            info_lines.append("GPS: Unknown")
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        y_offset = 35
        for line in info_lines:
            color = (0, 255, 0) if result['is_valid_match'] else (0, 255, 255)
            cv2.putText(frame, line, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Drone Navigation System')
    
    parser.add_argument('--mode', 
                       choices=['build-database', 'drone-navigation', 'test-matching'], 
                       required=True,
                       help='Operation mode')
    
    parser.add_argument('--satellite-dir', type=str,
                       help='Directory with satellite images (for build-database)')
    
    parser.add_argument('--gps-file', type=str,
                       help='JSON file with GPS metadata (for build-database)')
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID for drone navigation')
    
    parser.add_argument('--database-path', type=str, default='./drone_satellite_db',
                       help='Path to satellite database')
    
    args = parser.parse_args()
    
    print("🚁 Drone Navigation System")
    print("=" * 60)
    
    # Setup
    seed_everything(42)
    system = DroneNavigationSystem(args.database_path)
    
    try:
        if args.mode == 'build-database':
            if not args.satellite_dir:
                print("❌ --satellite-dir required for build-database mode")
                return
            
            system.build_satellite_database(args.satellite_dir, args.gps_file)
            
        elif args.mode == 'drone-navigation':
            system.run_drone_navigation(args.camera)
            
        elif args.mode == 'test-matching':
            # Test matching with sample images
            system.load_satellite_database()
            print("🧪 Test matching mode - place test images in ./test_images/")
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
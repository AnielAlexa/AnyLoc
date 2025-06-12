#!/usr/bin/env python3
"""
Real-time Drone-Satellite Matching with Optimizations

Uses optimal settings discovered from benchmarks:
- dinov2_vits14 (fastest model)
- 224x224 resolution (42+ FPS)
- 16 VLAD clusters (20ms aggregation)
- GPU acceleration
- Cached satellite database

Usage:
    python3 realtime_drone_matching.py --mode demo
    python3 realtime_drone_matching.py --mode webcam
"""

import os
import sys
from pathlib import Path
import time
import argparse
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

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from utilities import DinoV2ExtractFeatures, VLAD, seed_everything
from configs import device

class OptimizedDroneMatching:
    """Real-time drone-satellite matching with optimizations"""
    
    def __init__(self, satellite_cache_dir=None, use_onnx=False, use_torch_compile=False):
        self.satellite_cache_dir = satellite_cache_dir or "./AnyLoc2023-Public-Data/Public/Colab1/cache"
        self.use_onnx = use_onnx and HAS_ONNX
        self.use_torch_compile = use_torch_compile
        
        # Use vits14 for speed (4-5x faster than vitg14)
        self.model_name = "dinov2_vits14"  # Fast model
        self.resolution = (224, 224)       # Optimal for speed
        self.vlad_clusters = 16            # Optimal balance
        self.layer = 11                    # vits14 has 12 layers (0-11)
        self.facet = "value"               # Value facet
        
        # Performance tracking
        self.frame_times = []
        self.match_history = []
        
        self._load_models()
        self._load_satellite_database()
    
    def _load_models(self):
        """Load DINOv2 and VLAD models"""
        print("🔧 Loading optimized models...")
        
        # Load DINOv2 extractor
        print(f"   📥 Loading {self.model_name}...")
        self.extractor = DinoV2ExtractFeatures(
            self.model_name, 
            layer=self.layer, 
            facet=self.facet, 
            device=device
        )
        
        # Load VLAD aggregator
        print(f"   📥 Creating VLAD vocabulary for vits14 ({self.vlad_clusters} clusters)...")
        
        # vits14 has 384 dimensions, need to create vocabulary on-the-fly
        # or use cached features to generate vocabulary
        self.vlad = VLAD(num_clusters=self.vlad_clusters, desc_dim=384, cache_dir=None)
        
        # We'll fit the vocabulary using cached database features
        print(f"   🔧 Will fit vocabulary using database features...")
        
        # Apply torch.compile optimization if requested
        if self.use_torch_compile:
            print("   🚀 Applying torch.compile optimization...")
            try:
                self.extractor.dino_model = torch.compile(self.extractor.dino_model, mode='reduce-overhead')
                print("   ✅ torch.compile applied successfully!")
            except Exception as e:
                print(f"   ⚠️  torch.compile failed: {e}")
        
        print("✅ Models loaded successfully!")
        print(f"   🎯 Target resolution: {self.resolution}")
        print(f"   💾 VLAD descriptor size: {self.vlad_clusters * self.vlad.desc_dim}")
    
    def _load_satellite_database(self):
        """Load satellite database and create VLAD vocabulary from vits14 features"""
        print("🛰️  Loading satellite database...")
        
        # Load satellite images and extract features with vits14
        images_dir = f"{self.satellite_cache_dir}/imgs/aerial"
        
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Find database images
        db_image_files = [f for f in os.listdir(images_dir) if '_db-' in f and f.endswith('.png')]
        
        if not db_image_files:
            raise ValueError("No database images found!")
        
        print(f"   🖼️  Found {len(db_image_files)} database images")
        
        # Extract features for vocabulary creation and database
        all_db_features = []
        self.satellite_data = {}
        
        for img_file in db_image_files:
            base_name = img_file.replace('.png', '')
            img_path = f"{images_dir}/{img_file}"
            
            print(f"   🔄 Processing {base_name}...")
            
            try:
                # Load and preprocess image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Extract features with vits14
                img_tensor = self.preprocess_frame(image)
                with torch.no_grad():
                    features = self.extractor(img_tensor)
                    features = features.squeeze(0).cpu()
                
                # Store for vocabulary creation
                all_db_features.append(features)
                
                self.satellite_data[base_name] = {
                    'features': features,
                    'image_path': img_path
                }
                
            except Exception as e:
                print(f"⚠️  Error processing {img_file}: {e}")
        
        if not all_db_features:
            raise ValueError("No database features extracted!")
        
        # Create VLAD vocabulary from all database features
        print(f"   🧠 Creating VLAD vocabulary from {len(all_db_features)} images...")
        all_features_combined = torch.cat(all_db_features, dim=0)
        self.vlad.fit(all_features_combined)
        
        # Generate VLAD descriptors for database
        print(f"   📊 Generating VLAD descriptors...")
        self.satellite_vlads = []
        self.satellite_names = []
        
        for base_name, data in self.satellite_data.items():
            vlad_desc = self.vlad.generate(data['features'])
            data['vlad'] = vlad_desc
            
            self.satellite_vlads.append(vlad_desc)
            self.satellite_names.append(base_name)
        
        # Stack VLAD descriptors for fast similarity search
        self.satellite_vlads_tensor = torch.stack(self.satellite_vlads)
        print(f"✅ Loaded {len(self.satellite_vlads)} satellite images")
        print(f"   📊 Database shape: {self.satellite_vlads_tensor.shape}")
        print(f"   🎯 VLAD vocabulary: {self.vlad_clusters} clusters, {self.vlad.desc_dim}D features")
    
    def preprocess_frame(self, frame):
        """Preprocess video frame for DINOv2"""
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Resize to optimal resolution
        frame_resized = cv2.resize(frame_rgb, self.resolution)
        
        # Convert to PIL for consistency
        pil_image = Image.fromarray(frame_resized)
        
        # Convert to tensor
        img_tensor = torch.tensor(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0).to(device)
    
    def process_frame(self, frame):
        """Process single frame and find best satellite match"""
        
        start_time = time.time()
        
        # 1. Preprocess frame
        img_tensor = self.preprocess_frame(frame)
        preprocess_time = time.time()
        
        # 2. Extract DINOv2 features
        with torch.no_grad():
            features = self.extractor(img_tensor)
            features = features.squeeze(0).cpu()
        extraction_time = time.time()
        
        # 3. Generate VLAD descriptor
        vlad_desc = self.vlad.generate(features)
        vlad_time = time.time()
        
        # 4. Find best match in satellite database
        similarities = F.cosine_similarity(
            vlad_desc.unsqueeze(0), 
            self.satellite_vlads_tensor, 
            dim=1
        )
        
        best_idx = similarities.argmax().item()
        best_similarity = similarities[best_idx].item()
        best_match = self.satellite_names[best_idx]
        
        search_time = time.time()
        total_time = search_time - start_time
        
        # Track performance
        self.frame_times.append(total_time)
        if len(self.frame_times) > 100:  # Keep last 100 frames
            self.frame_times = self.frame_times[-100:]
        
        # Create result
        result = {
            'best_match': best_match,
            'similarity': best_similarity,
            'satellite_image_path': self.satellite_data[best_match]['image_path'],
            'timing': {
                'total': total_time * 1000,  # ms
                'preprocess': (preprocess_time - start_time) * 1000,
                'extraction': (extraction_time - preprocess_time) * 1000,
                'vlad': (vlad_time - extraction_time) * 1000,
                'search': (search_time - vlad_time) * 1000,
            },
            'performance': {
                'fps': 1.0 / total_time,
                'avg_fps': len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
            }
        }
        
        return result
    
    def run_benchmark_mode(self):
        """Benchmark different inference optimizations"""
        print("\n⚡ BENCHMARK MODE - Testing Inference Speed")
        print("=" * 50)
        
        # Load one test image
        query_dir = f"{self.satellite_cache_dir}/imgs/aerial"
        query_files = [f for f in os.listdir(query_dir) if '_qu-' in f and f.endswith('.png')]
        
        if not query_files:
            print("❌ No query images found for benchmark")
            return
        
        test_image_path = f"{query_dir}/{query_files[0]}"
        test_image = cv2.imread(test_image_path)
        
        print(f"🧪 Test image: {query_files[0]}")
        print(f"📊 Original resolution: {test_image.shape[:2]}")
        
        # Test different resolutions for vits14
        optimizations = [
            ("vits14 @ 224x224", (224, 224), False),
            ("vits14 @ 280x280", (280, 280), False),
            ("vits14 @ 336x336", (336, 336), False),
            ("vits14 @ 392x392", (392, 392), False),
            ("vits14 @ 448x448", (448, 448), False),
            ("vits14 @ 518x518", (518, 518), False),
        ]
        
        for opt_name, res, use_compile in optimizations:
            print(f"\n🔍 Testing: {opt_name}")
            
            # Apply torch.compile if needed for this test
            if use_compile and not hasattr(self.extractor.dino_model, '_orig_mod'):
                print("   🚀 Applying torch.compile...")
                self.extractor.dino_model = torch.compile(self.extractor.dino_model, mode='reduce-overhead')
            elif not use_compile and hasattr(self.extractor.dino_model, '_orig_mod'):
                print("   🔄 Reverting torch.compile...")
                # Can't easily revert torch.compile, so we'll just note it
            
            # Temporarily change resolution
            orig_res = self.resolution
            self.resolution = res
            
            # Run multiple times for average (more runs for compile optimization)
            num_runs = 10 if use_compile else 5
            times = []
            
            # Warmup for compiled model
            if use_compile:
                for _ in range(3):
                    img_tensor = self.preprocess_frame(test_image)
                    with torch.no_grad():
                        _ = self.extractor(img_tensor)
            
            for i in range(num_runs):
                start_time = time.time()
                
                # Just test feature extraction (not full pipeline)
                img_tensor = self.preprocess_frame(test_image)
                with torch.no_grad():
                    features = self.extractor(img_tensor)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            fps = 1.0 / avg_time
            
            print(f"   ⏱️  Average extraction time: {avg_time*1000:.1f}ms")
            print(f"   🚀 FPS: {fps:.1f}")
            
            # Restore resolution
            self.resolution = orig_res

    def run_demo_mode(self):
        """Run demo with sample images"""
        print("\n🎬 DEMO MODE - Processing Sample Images")
        print("=" * 50)
        
        # Load sample query images
        query_dir = f"{self.satellite_cache_dir}/imgs/aerial"
        query_files = [f for f in os.listdir(query_dir) if '_qu-' in f and f.endswith('.png')]
        
        if not query_files:
            print("❌ No query images found for demo")
            return
        
        for i, query_file in enumerate(query_files):
            print(f"\n📸 Processing query {i+1}/{len(query_files)}: {query_file}")
            
            # Load query image
            query_path = f"{query_dir}/{query_file}"
            query_image = cv2.imread(query_path)
            
            if query_image is None:
                print(f"   ❌ Could not load {query_file}")
                continue
            
            # Process frame
            result = self.process_frame(query_image)
            
            # Display results
            print(f"   🎯 Best match: {result['best_match']}")
            print(f"   📊 Similarity: {result['similarity']:.3f}")
            print(f"   ⏱️  Processing time: {result['timing']['total']:.1f}ms")
            print(f"   🚀 FPS: {result['performance']['fps']:.1f}")
            
            # Show timing breakdown
            timing = result['timing']
            print(f"   📋 Breakdown: Preprocess({timing['preprocess']:.1f}ms) + "
                  f"Extract({timing['extraction']:.1f}ms) + "
                  f"VLAD({timing['vlad']:.1f}ms) + "
                  f"Search({timing['search']:.1f}ms)")
        
        # Performance summary
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            avg_fps = 1.0 / avg_time
            print(f"\n📊 DEMO PERFORMANCE SUMMARY:")
            print(f"   Average processing time: {avg_time*1000:.1f}ms")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Frames processed: {len(self.frame_times)}")
    
    def run_webcam_mode(self, camera_id=0):
        """Run real-time webcam processing"""
        print(f"\n📹 WEBCAM MODE - Camera {camera_id}")
        print("=" * 50)
        print("Press 'q' to quit, 's' to save current match")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Could not open camera {camera_id}")
            return
        
        # Set camera properties for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Process every nth frame for performance
            skip_frames = 3  # Process every 3rd frame
            if frame_count % skip_frames != 0:
                cv2.imshow('Drone View', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Process frame
            try:
                result = self.process_frame(frame)
                
                # Create display frame with info
                display_frame = frame.copy()
                
                # Add text overlay
                info_text = [
                    f"Best Match: {result['best_match']}",
                    f"Similarity: {result['similarity']:.3f}",
                    f"FPS: {result['performance']['avg_fps']:.1f}",
                    f"Time: {result['timing']['total']:.0f}ms"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(display_frame, text, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
                
                # Show frames
                cv2.imshow('Drone View', display_frame)
                
                # Optionally show satellite match
                if os.path.exists(result['satellite_image_path']):
                    sat_img = cv2.imread(result['satellite_image_path'])
                    if sat_img is not None:
                        sat_img_resized = cv2.resize(sat_img, (400, 300))
                        cv2.imshow('Best Satellite Match', sat_img_resized)
                
                # Print performance stats
                if frame_count % (skip_frames * 10) == 0:  # Every 10 processed frames
                    print(f"Frame {frame_count//skip_frames}: {result['best_match']} "
                          f"(sim: {result['similarity']:.3f}, "
                          f"fps: {result['performance']['avg_fps']:.1f})")
                
            except Exception as e:
                print(f"Error processing frame: {e}")
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current match
                timestamp = int(time.time())
                cv2.imwrite(f"drone_frame_{timestamp}.jpg", frame)
                print(f"Saved frame: drone_frame_{timestamp}.jpg")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final performance summary
        if self.frame_times:
            avg_fps = len(self.frame_times) / sum(self.frame_times)
            print(f"\n📊 WEBCAM SESSION SUMMARY:")
            print(f"   Frames processed: {len(self.frame_times)}")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Total time: {sum(self.frame_times):.1f}s")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Real-time Drone-Satellite Matching')
    parser.add_argument('--mode', choices=['demo', 'webcam', 'benchmark'], default='demo',
                       help='Run mode: demo (sample images), webcam (real-time), or benchmark (speed test)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID for webcam mode')
    parser.add_argument('--cache-dir', type=str, 
                       default='./AnyLoc2023-Public-Data/Public/Colab1/cache',
                       help='Cache directory path')
    
    args = parser.parse_args()
    
    print("🚀 Real-time Drone-Satellite Matching System")
    print("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - running on CPU (slower)")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🖥️  GPU: {gpu_name}")
    
    seed_everything(42)
    
    try:
        # Initialize matching system
        print("\n🔧 Initializing optimized matching system...")
        matcher = OptimizedDroneMatching(
            satellite_cache_dir=args.cache_dir,
            use_onnx=HAS_ONNX
        )
        
        # Run selected mode
        if args.mode == 'demo':
            matcher.run_demo_mode()
        elif args.mode == 'webcam':
            matcher.run_webcam_mode(args.camera)
        elif args.mode == 'benchmark':
            matcher.run_benchmark_mode()
        
        print("\n✅ Session completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
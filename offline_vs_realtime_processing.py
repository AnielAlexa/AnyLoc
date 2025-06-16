#!/usr/bin/env python3
"""
Offline vs Real-time Processing Comparison for Drone Deployment

Compares two deployment strategies:
1. Real-time: Create VLAD vocabulary on drone startup
2. Offline: Pre-process database images, upload pre-computed descriptors

Shows speed improvements and practical deployment considerations.
"""

import os
import sys
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pickle
from datetime import datetime

from utilities import DinoV2ExtractFeatures, VLAD, seed_everything
from configs import device

class OfflineVsRealtimeComparison:
    """Compare offline preprocessing vs real-time processing for drone deployment"""
    
    def __init__(self):
        self.cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
        self.images_dir = f"{self.cache_dir}/imgs/aerial"
        self.output_dir = "./offline_processing_results"
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Drone configuration (optimized vitl14)
        self.drone_config = {
            'model': 'dinov2_vitl14',
            'layer': 23,
            'facet': 'value',
            'resolution': (224, 224),
            'feature_dim': 1024,
            'vlad_clusters': 16,
            'quantization': True  # FP16
        }
        
    def setup_feature_extractor(self):
        """Setup optimized feature extractor"""
        
        print("🔧 Setting up vitl14 feature extractor...")
        
        self.extractor = DinoV2ExtractFeatures(
            self.drone_config['model'], 
            self.drone_config['layer'], 
            self.drone_config['facet'], 
            device
        )
        
        # Apply FP16 optimization
        self.extractor.dino_model = self.extractor.dino_model.to(device).half()
        
        print("✅ Feature extractor ready (vitl14 + FP16)")
        
    def preprocess_image(self, image):
        """Preprocess image for vitl14"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        image_resized = cv2.resize(image_rgb, self.drone_config['resolution'])
        
        # Convert to tensor
        img_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Apply half precision for FP16
        return img_tensor.unsqueeze(0).to(device).half()
    
    def offline_preprocessing_approach(self):
        """Demonstrate offline preprocessing approach"""
        
        print("\n📦 OFFLINE PREPROCESSING APPROACH")
        print("=" * 60)
        print("Pre-process database images offline, upload descriptors to drone")
        
        # Find database images
        db_files = [f for f in os.listdir(self.images_dir) if '_db-' in f and f.endswith('.png')][:10]
        
        if not db_files:
            print("❌ No database images found!")
            return None
        
        print(f"📸 Processing {len(db_files)} database images offline...")
        
        # Step 1: Extract features from all database images
        print("   🔧 Step 1: Extracting features...")
        start_time = time.time()
        
        all_features = []
        db_image_names = []
        
        for db_file in db_files:
            img_path = f"{self.images_dir}/{db_file}"
            image = cv2.imread(img_path)
            
            if image is not None:
                img_tensor = self.preprocess_image(image)
                
                with torch.no_grad():
                    features = self.extractor(img_tensor)
                    features = features.squeeze(0).cpu().float()
                    all_features.append(features)
                    db_image_names.append(db_file.replace('.png', ''))
        
        feature_extraction_time = time.time() - start_time
        
        # Step 2: Create VLAD vocabulary
        print("   🧠 Step 2: Creating VLAD vocabulary...")
        vocab_start = time.time()
        
        vlad = VLAD(num_clusters=self.drone_config['vlad_clusters'], 
                   desc_dim=self.drone_config['feature_dim'], 
                   cache_dir=None)
        
        combined_features = torch.cat(all_features, dim=0)
        vlad.fit(combined_features)
        
        vocab_creation_time = time.time() - vocab_start
        
        # Step 3: Generate VLAD descriptors for all database images  
        print("   📊 Step 3: Generating VLAD descriptors...")
        vlad_start = time.time()
        
        db_descriptors = []
        for features in all_features:
            vlad_desc = vlad.generate(features)
            db_descriptors.append(vlad_desc)
        
        vlad_generation_time = time.time() - vlad_start
        
        # Step 4: Package for drone deployment
        print("   📦 Step 4: Packaging for drone...")
        package_start = time.time()
        
        # Create deployment package
        deployment_package = {
            'vlad_vocabulary': {
                'c_centers': vlad.c_centers,
                'num_clusters': vlad.num_clusters,
                'desc_dim': vlad.desc_dim
            },
            'database_descriptors': torch.stack(db_descriptors),
            'database_names': db_image_names,
            'config': self.drone_config,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'num_database_images': len(db_image_names),
                'processing_time_seconds': time.time() - start_time
            }
        }
        
        # Save deployment package
        package_path = f"{self.output_dir}/drone_deployment_package.pkl"
        with open(package_path, 'wb') as f:
            pickle.dump(deployment_package, f)
        
        package_time = time.time() - package_start
        total_offline_time = time.time() - start_time
        
        # Calculate package size
        package_size_mb = os.path.getsize(package_path) / (1024 * 1024)
        
        print(f"\n📊 Offline Processing Results:")
        print(f"   Feature extraction: {feature_extraction_time:.2f}s")
        print(f"   VLAD vocabulary:    {vocab_creation_time:.2f}s") 
        print(f"   VLAD descriptors:   {vlad_generation_time:.2f}s")
        print(f"   Packaging:          {package_time:.2f}s")
        print(f"   Total offline time: {total_offline_time:.2f}s")
        print(f"   Package size:       {package_size_mb:.1f}MB")
        print(f"   Package saved:      {package_path}")
        
        return {
            'package_path': package_path,
            'package_size_mb': package_size_mb,
            'total_time': total_offline_time,
            'num_images': len(db_image_names),
            'deployment_package': deployment_package
        }
    
    def simulate_drone_startup_offline(self, offline_results):
        """Simulate drone startup with offline-processed data"""
        
        print(f"\n🚁 DRONE STARTUP - OFFLINE PROCESSED DATA")
        print("=" * 60)
        print("Drone loads pre-computed descriptors (no feature extraction needed)")
        
        package_path = offline_results['package_path']
        
        # Simulate drone startup
        startup_start = time.time()
        
        print("   📁 Loading deployment package...")
        with open(package_path, 'rb') as f:
            deployment_package = pickle.load(f)
        
        # Reconstruct VLAD from package
        print("   🧠 Loading VLAD vocabulary...")
        vlad = VLAD(num_clusters=deployment_package['vlad_vocabulary']['num_clusters'],
                   desc_dim=deployment_package['vlad_vocabulary']['desc_dim'],
                   cache_dir=None)
        
        # Set clusters and fit status
        vlad.c_centers = deployment_package['vlad_vocabulary']['c_centers']
        vlad.is_fitted = True
        
        # Create a minimal kmeans mock for compatibility
        class KMeansMock:
            def __init__(self, cluster_centers):
                self.cluster_centers_ = cluster_centers.cpu().numpy()
                
            def predict(self, X):
                # Simple nearest cluster assignment
                import torch.nn.functional as F
                X_tensor = torch.tensor(X).to(self.cluster_centers_.device if hasattr(self.cluster_centers_, 'device') else 'cpu')
                centers_tensor = torch.tensor(self.cluster_centers_)
                distances = torch.cdist(X_tensor, centers_tensor)
                return distances.argmin(dim=1).cpu().numpy()
        
        vlad.kmeans = KMeansMock(vlad.c_centers)
        
        # Load database descriptors
        print("   📊 Loading database descriptors...")
        db_descriptors = deployment_package['database_descriptors']
        db_names = deployment_package['database_names']
        
        startup_time = time.time() - startup_start
        
        print(f"\n✅ Drone ready in {startup_time:.3f}s!")
        print(f"   Database: {len(db_names)} images loaded")
        print(f"   VLAD vocabulary: {vlad.num_clusters} clusters, {vlad.desc_dim}D")
        
        return {
            'startup_time': startup_time,
            'vlad': vlad,
            'db_descriptors': db_descriptors,
            'db_names': db_names
        }
    
    def simulate_drone_startup_realtime(self):
        """Simulate drone startup with real-time processing"""
        
        print(f"\n🚁 DRONE STARTUP - REAL-TIME PROCESSING")
        print("=" * 60)
        print("Drone processes database images on startup")
        
        # Find database images
        db_files = [f for f in os.listdir(self.images_dir) if '_db-' in f and f.endswith('.png')][:10]
        
        startup_start = time.time()
        
        print("   🔧 Extracting features from database images...")
        all_features = []
        db_names = []
        
        for db_file in db_files:
            img_path = f"{self.images_dir}/{db_file}"
            image = cv2.imread(img_path)
            
            if image is not None:
                img_tensor = self.preprocess_image(image)
                
                with torch.no_grad():
                    features = self.extractor(img_tensor)
                    features = features.squeeze(0).cpu().float()
                    all_features.append(features)
                    db_names.append(db_file.replace('.png', ''))
        
        print("   🧠 Creating VLAD vocabulary...")
        vlad = VLAD(num_clusters=self.drone_config['vlad_clusters'], 
                   desc_dim=self.drone_config['feature_dim'], 
                   cache_dir=None)
        
        combined_features = torch.cat(all_features, dim=0)
        vlad.fit(combined_features)
        
        print("   📊 Generating database descriptors...")
        db_descriptors = []
        for features in all_features:
            vlad_desc = vlad.generate(features)
            db_descriptors.append(vlad_desc)
        
        db_descriptors = torch.stack(db_descriptors)
        
        startup_time = time.time() - startup_start
        
        print(f"\n✅ Drone ready in {startup_time:.3f}s!")
        print(f"   Database: {len(db_names)} images processed")
        print(f"   VLAD vocabulary: {vlad.num_clusters} clusters, {vlad.desc_dim}D")
        
        return {
            'startup_time': startup_time,
            'vlad': vlad,
            'db_descriptors': db_descriptors,
            'db_names': db_names
        }
    
    def benchmark_query_processing(self, offline_data, realtime_data):
        """Benchmark query processing speed for both approaches"""
        
        print(f"\n⚡ QUERY PROCESSING BENCHMARK")
        print("=" * 60)
        print("Compare speed of processing live drone images")
        
        # Create test query image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Benchmark offline approach (only feature extraction + VLAD needed)
        print("🔄 Benchmarking offline approach...")
        offline_times = []
        
        vlad_offline = offline_data['vlad']
        
        # Warmup
        for _ in range(3):
            img_tensor = self.preprocess_image(test_image)
            with torch.no_grad():
                features = self.extractor(img_tensor)
                features = features.squeeze(0).cpu().float()
                _ = vlad_offline.generate(features)
        
        # Benchmark
        for _ in range(20):
            torch.cuda.synchronize()
            start_time = time.time()
            
            img_tensor = self.preprocess_image(test_image)
            with torch.no_grad():
                features = self.extractor(img_tensor)
                features = features.squeeze(0).cpu().float()
                query_desc = vlad_offline.generate(features)
            
            # Similarity search
            similarities = F.cosine_similarity(query_desc.unsqueeze(0), offline_data['db_descriptors'], dim=1)
            best_match_idx = similarities.argmax().item()
            
            torch.cuda.synchronize()
            end_time = time.time()
            offline_times.append((end_time - start_time) * 1000)
        
        # Benchmark real-time approach (same - no difference in query processing)
        print("🔄 Benchmarking real-time approach...")
        realtime_times = []
        
        vlad_realtime = realtime_data['vlad']
        
        # Warmup
        for _ in range(3):
            img_tensor = self.preprocess_image(test_image)
            with torch.no_grad():
                features = self.extractor(img_tensor)
                features = features.squeeze(0).cpu().float()
                _ = vlad_realtime.generate(features)
        
        # Benchmark
        for _ in range(20):
            torch.cuda.synchronize()
            start_time = time.time()
            
            img_tensor = self.preprocess_image(test_image)
            with torch.no_grad():
                features = self.extractor(img_tensor)
                features = features.squeeze(0).cpu().float()
                query_desc = vlad_realtime.generate(features)
            
            # Similarity search
            similarities = F.cosine_similarity(query_desc.unsqueeze(0), realtime_data['db_descriptors'], dim=1)
            best_match_idx = similarities.argmax().item()
            
            torch.cuda.synchronize()
            end_time = time.time()
            realtime_times.append((end_time - start_time) * 1000)
        
        offline_avg = np.mean(offline_times)
        realtime_avg = np.mean(realtime_times)
        
        offline_fps = 1000 / offline_avg
        realtime_fps = 1000 / realtime_avg
        
        print(f"\n📊 Query Processing Results:")
        print(f"   Offline approach:  {offline_avg:.1f}ms ({offline_fps:.1f} FPS)")
        print(f"   Real-time approach: {realtime_avg:.1f}ms ({realtime_fps:.1f} FPS)")
        print(f"   Speed difference: {abs(offline_avg - realtime_avg):.1f}ms ({abs(offline_fps - realtime_fps):.1f} FPS)")
        
        return {
            'offline': {'avg_time_ms': offline_avg, 'fps': offline_fps},
            'realtime': {'avg_time_ms': realtime_avg, 'fps': realtime_fps}
        }
    
    def comprehensive_comparison(self):
        """Run comprehensive comparison between approaches"""
        
        print("🏆 OFFLINE vs REAL-TIME PROCESSING COMPARISON")
        print("=" * 80)
        print("Comprehensive analysis for drone deployment strategies")
        print("=" * 80)
        
        self.setup_feature_extractor()
        
        # Run offline preprocessing
        offline_results = self.offline_preprocessing_approach()
        if not offline_results:
            return
        
        # Simulate drone startups
        offline_drone = self.simulate_drone_startup_offline(offline_results)
        realtime_drone = self.simulate_drone_startup_realtime()
        
        # Benchmark query processing
        query_benchmark = self.benchmark_query_processing(offline_drone, realtime_drone)
        
        # Print comprehensive analysis
        self._print_deployment_analysis(offline_results, offline_drone, realtime_drone, query_benchmark)
    
    def _print_deployment_analysis(self, offline_results, offline_drone, realtime_drone, query_benchmark):
        """Print comprehensive deployment analysis"""
        
        print(f"\n🏆 DEPLOYMENT STRATEGY ANALYSIS")
        print("=" * 80)
        
        print(f"📋 COMPARISON SUMMARY:")
        print(f"{'Aspect':<25} {'Offline Processing':<25} {'Real-time Processing':<25} {'Winner'}") 
        print("-" * 85)
        
        # Startup time comparison
        startup_winner = "Offline" if offline_drone['startup_time'] < realtime_drone['startup_time'] else "Real-time"
        startup_speedup = realtime_drone['startup_time'] / offline_drone['startup_time']
        
        print(f"{'Drone startup time':<25} {offline_drone['startup_time']:.3f}s{'':<16} {realtime_drone['startup_time']:.3f}s{'':<16} {'🚀 ' + startup_winner + f' ({startup_speedup:.1f}x)'}")
        
        # Query processing (should be identical)
        query_diff = abs(query_benchmark['offline']['fps'] - query_benchmark['realtime']['fps'])
        query_winner = "🤝 Identical" if query_diff < 1 else ("Offline" if query_benchmark['offline']['fps'] > query_benchmark['realtime']['fps'] else "Real-time")
        
        print(f"{'Query processing':<25} {query_benchmark['offline']['fps']:.1f} FPS{'':<15} {query_benchmark['realtime']['fps']:.1f} FPS{'':<15} {query_winner}")
        
        # Storage requirements
        storage_winner = "Real-time" if offline_results['package_size_mb'] > 10 else "Offline"
        print(f"{'Storage required':<25} {offline_results['package_size_mb']:.1f}MB{'':<19} ~0MB (just code){'':<11} {'💾 ' + storage_winner}")
        
        # Setup complexity
        print(f"{'Setup complexity':<25} {'High (offline pipeline)':<25} {'Low (just run)':<25} {'⚡ Real-time'}")
        
        # Flexibility
        print(f"{'Environment change':<25} {'Reprocess offline':<25} {'Adapt immediately':<25} {'🔄 Real-time'}")
        
        # Internet dependency
        print(f"{'Internet dependency':<25} {'Upload package':<25} {'None':<25} {'🌐 Real-time'}")
        
        print(f"\n💡 KEY INSIGHTS:")
        
        if startup_speedup > 2:
            print(f"   🚀 Offline startup is {startup_speedup:.1f}x faster ({offline_drone['startup_time']:.3f}s vs {realtime_drone['startup_time']:.3f}s)")
        
        print(f"   ⚡ Query processing is identical (~{query_benchmark['offline']['fps']:.1f} FPS)")
        print(f"   📦 Offline requires {offline_results['package_size_mb']:.1f}MB deployment package")
        print(f"   🔄 Real-time adapts to environment changes instantly")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        
        # Different scenarios
        print(f"\n📋 Use OFFLINE processing when:")
        print(f"   🏭 Fixed flight routes (same area repeatedly)")
        print(f"   ⚡ Critical startup time (<{offline_drone['startup_time']:.3f}s required)")
        print(f"   📊 Large database (50+ reference images)")
        print(f"   🔋 Want to minimize onboard computation")
        print(f"   🌐 Reliable upload/download infrastructure")
        
        print(f"\n📋 Use REAL-TIME processing when:")
        print(f"   🗺️ Changing flight areas frequently")
        print(f"   🚁 Startup time <{realtime_drone['startup_time']:.3f}s is acceptable")
        print(f"   📦 Want self-contained system")
        print(f"   🌍 No reliable internet for package uploads")
        print(f"   🔧 Rapid prototyping and testing")
        
        print(f"\n🏆 OPTIMAL HYBRID APPROACH:")
        print(f"   💡 Use offline processing for production flights")
        print(f"   💡 Use real-time processing for development/testing")
        print(f"   💡 Implement both modes in your drone software")
        print(f"   💡 Switch based on mission requirements")
        
        # Jetson considerations
        print(f"\n🤖 JETSON ORIN NANO CONSIDERATIONS:")
        jetson_offline_startup = offline_drone['startup_time'] * 2.5  # Jetson is slower
        jetson_realtime_startup = realtime_drone['startup_time'] * 2.5
        
        print(f"   Projected offline startup: ~{jetson_offline_startup:.2f}s")
        print(f"   Projected real-time startup: ~{jetson_realtime_startup:.2f}s")
        print(f"   Both are practical for drone operations!")

def main():
    """Main comparison function"""
    
    print("🔬 Offline vs Real-time Processing Analysis")
    print("=" * 60)
    
    seed_everything(42)
    
    # Check data availability
    cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
    if not os.path.exists(cache_dir):
        print("❌ AnyLoc2023-Public-Data not found!")
        return
    
    try:
        comparator = OfflineVsRealtimeComparison()
        comparator.comprehensive_comparison()
        
        print(f"\n✅ Comprehensive comparison completed!")
        print(f"🎉 Both approaches have their advantages - choose based on your needs!")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
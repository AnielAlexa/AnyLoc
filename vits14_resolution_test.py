#!/usr/bin/env python3
"""
Test vits14 accuracy and performance at different resolutions
"""

import os
import sys
from pathlib import Path
import time

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

import torch
import torch.nn.functional as F
import cv2
import numpy as np

from utilities import DinoV2ExtractFeatures, VLAD, seed_everything
from configs import device

class VitS14ResolutionTest:
    """Test vits14 at different resolutions"""
    
    def __init__(self, cache_dir="./AnyLoc2023-Public-Data/Public/Colab1/cache"):
        self.cache_dir = cache_dir
        self.images_dir = f"{cache_dir}/imgs/aerial"
        
        self.ground_truth = {
            'nardo-air_qu-42': 'nardo-air_db-42',
            'nardo-air-r_qu-70': 'nardo-air-r_db-45',
            'vpair_qu-122': 'vpair_db-122'
        }
        
        self.results = []
    
    def preprocess_image(self, image_path, resolution):
        """Preprocess image for DINOv2"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(image_rgb, resolution)
        
        # Convert to tensor
        img_tensor = torch.tensor(image_resized).permute(2, 0, 1).float() / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0).to(device)
    
    def test_resolution(self, resolution, vlad_clusters=16):
        """Test vits14 at a specific resolution"""
        
        print(f"\n{'='*60}")
        print(f"🔍 TESTING: DINOv2-vits14 @ {resolution[0]}x{resolution[1]}")
        print(f"{'='*60}")
        
        try:
            # Load model
            print(f"   📥 Loading dinov2_vits14...")
            extractor = DinoV2ExtractFeatures("dinov2_vits14", 11, "value", device)
            
            # Load database images
            db_image_files = [f for f in os.listdir(self.images_dir) 
                             if '_db-' in f and f.endswith('.png')]
            
            if not db_image_files:
                raise ValueError("No database images found!")
            
            print(f"   🖼️  Found {len(db_image_files)} database images")
            
            # Extract database features
            print(f"   🗃️  Extracting database features...")
            db_features = []
            db_names = []
            db_extraction_times = []
            
            for img_file in db_image_files:
                img_path = f"{self.images_dir}/{img_file}"
                base_name = img_file.replace('.png', '')
                
                # Extract features
                img_tensor = self.preprocess_image(img_path, resolution)
                
                start_time = time.time()
                with torch.no_grad():
                    features = extractor(img_tensor)
                    features = features.squeeze(0).cpu()
                torch.cuda.synchronize()
                end_time = time.time()
                
                db_features.append(features)
                db_names.append(base_name)
                db_extraction_times.append((end_time - start_time) * 1000)
            
            # Create VLAD vocabulary
            print(f"   🧠 Creating VLAD vocabulary ({vlad_clusters} clusters)...")
            vlad = VLAD(num_clusters=vlad_clusters, desc_dim=384, cache_dir=None)
            
            all_db_features = torch.cat(db_features, dim=0)
            vlad.fit(all_db_features)
            
            # Generate database VLAD descriptors
            print(f"   📊 Generating database VLAD descriptors...")
            db_vlads = []
            db_vlad_times = []
            
            for features in db_features:
                start_time = time.time()
                vlad_desc = vlad.generate(features)
                end_time = time.time()
                
                db_vlads.append(vlad_desc)
                db_vlad_times.append((end_time - start_time) * 1000)
            
            db_vlads_tensor = torch.stack(db_vlads)
            
            # Test queries
            print(f"   🎯 Testing queries...")
            correct = 0
            total = 0
            qu_extraction_times = []
            qu_vlad_times = []
            similarities = []
            
            for query_name, expected_match in self.ground_truth.items():
                query_path = f"{self.images_dir}/{query_name}.png"
                
                if not os.path.exists(query_path):
                    continue
                
                # Extract query features
                img_tensor = self.preprocess_image(query_path, resolution)
                
                start_time = time.time()
                with torch.no_grad():
                    qu_features = extractor(img_tensor)
                    qu_features = qu_features.squeeze(0).cpu()
                torch.cuda.synchronize()
                extraction_end = time.time()
                
                # Generate VLAD descriptor
                qu_vlad = vlad.generate(qu_features)
                vlad_end = time.time()
                
                # Find best match
                sims = F.cosine_similarity(qu_vlad.unsqueeze(0), db_vlads_tensor, dim=1)
                best_idx = sims.argmax().item()
                best_similarity = sims[best_idx].item()
                best_match = db_names[best_idx]
                
                is_correct = (best_match == expected_match)
                
                print(f"      🔍 {query_name} → {best_match} (sim: {best_similarity:.3f}) {'✅' if is_correct else '❌'}")
                
                if is_correct:
                    correct += 1
                total += 1
                
                qu_extraction_times.append((extraction_end - start_time) * 1000)
                qu_vlad_times.append((vlad_end - extraction_end) * 1000)
                similarities.append(best_similarity)
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            
            avg_extraction_time = np.mean(db_extraction_times + qu_extraction_times)
            avg_vlad_time = np.mean(db_vlad_times + qu_vlad_times)
            total_time = avg_extraction_time + avg_vlad_time
            
            avg_similarity = np.mean(similarities)
            fps = 1000 / total_time if total_time > 0 else 0
            
            # Calculate speedup vs baseline (224x224)
            baseline_time = 26.4  # From previous test
            speedup = baseline_time / total_time if total_time > 0 else 0
            
            result = {
                'resolution': resolution,
                'accuracy': accuracy,
                'avg_extraction_time_ms': avg_extraction_time,
                'avg_vlad_time_ms': avg_vlad_time,
                'total_time_ms': total_time,
                'fps': fps,
                'avg_similarity': avg_similarity,
                'correct': correct,
                'total': total,
                'speedup_vs_224': speedup,
                'pixels': resolution[0] * resolution[1]
            }
            
            self.results.append(result)
            
            # Print summary
            print(f"\n   📊 RESULTS:")
            print(f"      Accuracy: {accuracy:.1%} ({correct}/{total})")
            print(f"      Avg extraction time: {avg_extraction_time:.1f}ms")
            print(f"      Avg VLAD time: {avg_vlad_time:.1f}ms")
            print(f"      Total time: {total_time:.1f}ms")
            print(f"      FPS: {fps:.1f}")
            print(f"      Avg similarity: {avg_similarity:.3f}")
            print(f"      Speedup vs 224x224: {speedup:.2f}x")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Resolution {resolution} failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_resolution_sweep(self):
        """Test multiple resolutions"""
        
        print("🚀 DINOv2-vits14 Resolution Sweep")
        print("=" * 70)
        
        # Test different resolutions
        resolutions = [
            (224, 224),   # Baseline - fast
            (280, 280),   # Slightly higher
            (336, 336),   # Medium
            (392, 392),   # Higher
            (448, 448),   # High
            (518, 518),   # vitg14 default
        ]
        
        for resolution in resolutions:
            self.test_resolution(resolution)
        
        # Analysis
        self.print_analysis()
    
    def print_analysis(self):
        """Print comprehensive analysis"""
        
        if not self.results:
            print("❌ No results to analyze")
            return
        
        print(f"\n{'='*100}")
        print("📊 RESOLUTION ANALYSIS - DINOv2-vits14")
        print(f"{'='*100}")
        
        # Sort by resolution
        sorted_results = sorted(self.results, key=lambda x: x['pixels'])
        
        # Header
        print(f"{'Resolution':<12} {'Pixels':<8} {'Accuracy':<10} {'Extract(ms)':<12} {'VLAD(ms)':<10} {'Total(ms)':<10} {'FPS':<6} {'Similarity':<10} {'Quality'}")
        print("-" * 100)
        
        for result in sorted_results:
            res_str = f"{result['resolution'][0]}x{result['resolution'][1]}"
            pixels_k = result['pixels'] // 1000
            
            # Quality assessment
            if result['accuracy'] >= 0.8 and result['fps'] >= 15:
                quality = "🚀 Excellent"
            elif result['accuracy'] >= 0.6 and result['fps'] >= 10:
                quality = "⚡ Good"
            elif result['accuracy'] >= 0.4 and result['fps'] >= 5:
                quality = "📊 Fair"
            else:
                quality = "⚠️  Poor"
            
            print(f"{res_str:<12} {pixels_k:<8}K {result['accuracy']:<10.1%} {result['avg_extraction_time_ms']:<12.1f} {result['avg_vlad_time_ms']:<10.1f} {result['total_time_ms']:<10.1f} {result['fps']:<6.1f} {result['avg_similarity']:<10.3f} {quality}")
        
        # Find optimal points
        print(f"\n💡 KEY INSIGHTS:")
        
        # Best accuracy
        best_accuracy = max(self.results, key=lambda x: x['accuracy'])
        print(f"   🎯 Highest Accuracy: {best_accuracy['resolution'][0]}x{best_accuracy['resolution'][1]} ({best_accuracy['accuracy']:.1%})")
        
        # Best speed (real-time)
        realtime_candidates = [r for r in self.results if r['fps'] >= 15]
        if realtime_candidates:
            fastest_realtime = max(realtime_candidates, key=lambda x: x['fps'])
            print(f"   🚀 Fastest Real-time: {fastest_realtime['resolution'][0]}x{fastest_realtime['resolution'][1]} ({fastest_realtime['fps']:.1f} FPS)")
        
        # Best balanced (accuracy * fps score)
        best_balanced = max(self.results, key=lambda x: x['accuracy'] * (x['fps'] / 10))
        print(f"   ⚖️  Best Balanced: {best_balanced['resolution'][0]}x{best_balanced['resolution'][1]} (Score: {best_balanced['accuracy'] * (best_balanced['fps'] / 10):.2f})")
        
        # Sweet spot analysis
        print(f"\n🎯 SWEET SPOT ANALYSIS:")
        
        # Find resolution that gives good accuracy (>= 80%) with decent speed (>= 10 FPS)
        sweet_spots = [r for r in self.results if r['accuracy'] >= 0.8 and r['fps'] >= 10]
        if sweet_spots:
            best_sweet_spot = max(sweet_spots, key=lambda x: x['fps'])
            print(f"   🍯 Optimal: {best_sweet_spot['resolution'][0]}x{best_sweet_spot['resolution'][1]} - {best_sweet_spot['accuracy']:.1%} accuracy @ {best_sweet_spot['fps']:.1f} FPS")
        else:
            # Fallback: best accuracy among real-time candidates
            if realtime_candidates:
                fallback = max(realtime_candidates, key=lambda x: x['accuracy'])
                print(f"   📊 Compromise: {fallback['resolution'][0]}x{fallback['resolution'][1]} - {fallback['accuracy']:.1%} accuracy @ {fallback['fps']:.1f} FPS")
        
        # Accuracy vs resolution trend
        print(f"\n📈 ACCURACY vs RESOLUTION TREND:")
        baseline_224 = next((r for r in self.results if r['resolution'] == (224, 224)), None)
        
        if baseline_224:
            for result in sorted_results:
                if result != baseline_224:
                    acc_improvement = result['accuracy'] - baseline_224['accuracy']
                    speed_ratio = baseline_224['fps'] / result['fps']
                    print(f"   {result['resolution'][0]}x{result['resolution'][1]}: {acc_improvement:+.1%} accuracy, {speed_ratio:.1f}x slower")
        
        # Recommendations
        print(f"\n🏆 RECOMMENDATIONS:")
        
        print(f"   💨 For Maximum Speed: 224x224 (baseline)")
        
        if sweet_spots:
            best_sweet_spot = max(sweet_spots, key=lambda x: x['fps'])
            print(f"   🎯 For Best Balance: {best_sweet_spot['resolution'][0]}x{best_sweet_spot['resolution'][1]} - excellent accuracy + real-time speed")
        
        highest_acc = max(self.results, key=lambda x: x['accuracy'])
        if highest_acc['fps'] >= 5:
            print(f"   🌟 For Maximum Accuracy: {highest_acc['resolution'][0]}x{highest_acc['resolution'][1]} - {highest_acc['accuracy']:.1%} accuracy @ {highest_acc['fps']:.1f} FPS")
        
        # Compare to vitg14 baseline
        print(f"\n⚖️  vs vitg14 @ 518x518 (1.4 FPS, ~100% accuracy):")
        for result in self.results:
            if result['accuracy'] >= 0.8:  # Close to vitg14 accuracy
                fps_improvement = result['fps'] / 1.4
                print(f"   vits14 @ {result['resolution'][0]}x{result['resolution'][1]}: {result['accuracy']:.1%} accuracy, {fps_improvement:.1f}x faster!")

def main():
    """Main function"""
    seed_everything(42)
    
    print("🚀 DINOv2-vits14 Resolution Impact Analysis")
    print("=" * 70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - running on CPU (slower)")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🖥️  GPU: {gpu_name}")
    
    try:
        tester = VitS14ResolutionTest()
        tester.run_resolution_sweep()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
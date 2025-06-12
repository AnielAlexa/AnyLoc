#!/usr/bin/env python3
"""
Simple accuracy test using the working realtime matching approach
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
from realtime_drone_matching import OptimizedDroneMatching

def test_accuracy_different_resolutions():
    """Test accuracy with different resolutions using working realtime matcher"""
    
    print("🚀 VPAir Accuracy Test - Different Resolutions")
    print("=" * 60)
    
    # Ground truth for simple evaluation
    ground_truth = {
        'nardo-air_qu-42': 'nardo-air_db-42',
        'nardo-air-r_qu-70': 'nardo-air-r_db-45',  # Cross-viewpoint
        'vpair_qu-122': 'vpair_db-122'
    }
    
    # Test different configurations
    configs = [
        ("vits14 @ 224x224", "dinov2_vits14", (224, 224), 11),
        ("vitg14 @ 224x224", "dinov2_vitg14", (224, 224), 39), 
        ("vitg14 @ 518x518", "dinov2_vitg14", (518, 518), 39),
    ]
    
    results = []
    
    for config_name, model_name, resolution, layer in configs:
        print(f"\n{'='*50}")
        print(f"🔍 TESTING: {config_name}")
        print(f"{'='*50}")
        
        try:
            # Create matcher with this configuration
            # Temporarily modify the class to use our config
            class TestMatcher(OptimizedDroneMatching):
                def __init__(self, model_name, resolution, layer):
                    self.satellite_cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
                    self.use_onnx = False
                    self.use_torch_compile = False
                    
                    self.model_name = model_name
                    self.resolution = resolution
                    self.vlad_clusters = 16
                    self.layer = layer
                    self.facet = "value"
                    
                    self.frame_times = []
                    self.match_history = []
                    
                    self._load_models()
                    self._load_satellite_database()
            
            matcher = TestMatcher(model_name, resolution, layer)
            
            # Test queries
            cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
            query_dir = f"{cache_dir}/imgs/aerial"
            
            correct = 0
            total = 0
            timings = []
            similarities = []
            
            for query_name, expected_match in ground_truth.items():
                query_path = f"{query_dir}/{query_name}.png"
                
                if not os.path.exists(query_path):
                    continue
                
                print(f"   🔍 Testing: {query_name}")
                
                # Load and process
                query_image = cv2.imread(query_path)
                
                start_time = time.time()
                result = matcher.process_frame(query_image)
                end_time = time.time()
                
                best_match = result['best_match']
                similarity = result['similarity']
                processing_time = (end_time - start_time) * 1000  # ms
                
                is_correct = (best_match == expected_match)
                
                print(f"      → {best_match} (sim: {similarity:.3f}, {processing_time:.1f}ms) {'✅' if is_correct else '❌'}")
                
                if is_correct:
                    correct += 1
                total += 1
                
                timings.append(processing_time)
                similarities.append(similarity)
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            avg_time = sum(timings) / len(timings) if timings else 0
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            fps = 1000 / avg_time if avg_time > 0 else 0
            
            result_summary = {
                'config': config_name,
                'accuracy': accuracy,
                'avg_time_ms': avg_time,
                'fps': fps,
                'avg_similarity': avg_similarity,
                'correct': correct,
                'total': total
            }
            
            results.append(result_summary)
            
            print(f"\n   📊 SUMMARY:")
            print(f"      Accuracy: {accuracy:.1%} ({correct}/{total})")
            print(f"      Avg time: {avg_time:.1f}ms")
            print(f"      FPS: {fps:.1f}")
            print(f"      Avg similarity: {avg_similarity:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON")
    print(f"{'='*80}")
    
    if results:
        print(f"{'Configuration':<20} {'Accuracy':<10} {'Time(ms)':<10} {'FPS':<8} {'Similarity':<12} {'Quality'}")
        print("-" * 70)
        
        # Sort by a combined score (accuracy weighted by speed)
        sorted_results = sorted(results, key=lambda x: x['accuracy'] * (x['fps'] / 10), reverse=True)
        
        for result in sorted_results:
            # Quality assessment
            if result['accuracy'] >= 0.8 and result['fps'] >= 15:
                quality = "🚀 Excellent"
            elif result['accuracy'] >= 0.6 and result['fps'] >= 5:
                quality = "⚡ Good"  
            elif result['accuracy'] >= 0.3:
                quality = "📊 Fair"
            else:
                quality = "⚠️  Poor"
            
            print(f"{result['config']:<20} {result['accuracy']:<10.1%} {result['avg_time_ms']:<10.1f} {result['fps']:<8.1f} {result['avg_similarity']:<12.3f} {quality}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        best_accuracy = max(results, key=lambda x: x['accuracy'])
        best_speed = max(results, key=lambda x: x['fps'])
        best_balanced = max(results, key=lambda x: x['accuracy'] * (x['fps'] / 10))
        
        print(f"   🎯 Most Accurate: {best_accuracy['config']} ({best_accuracy['accuracy']:.1%})")
        print(f"   🚀 Fastest: {best_speed['config']} ({best_speed['fps']:.1f} FPS)")
        print(f"   ⚖️  Best Balanced: {best_balanced['config']} (Score: {best_balanced['accuracy'] * (best_balanced['fps'] / 10):.2f})")
        
        # Speed comparison
        baseline = next((r for r in results if 'vitg14 @ 518x518' in r['config']), results[-1])
        print(f"\n⚡ SPEEDUP ANALYSIS (vs {baseline['config']}):")
        for result in sorted_results:
            if result != baseline:
                speedup = result['fps'] / baseline['fps']
                print(f"   {result['config']}: {speedup:.1f}x faster")

def main():
    test_accuracy_different_resolutions()

if __name__ == "__main__":
    main()
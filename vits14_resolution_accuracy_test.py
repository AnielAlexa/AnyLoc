#!/usr/bin/env python3
"""
Test vits14 accuracy and performance at different resolutions using working approach
"""

import os
import sys
from pathlib import Path
import time

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

import cv2
import numpy as np
from realtime_drone_matching import OptimizedDroneMatching

def test_vits14_resolution_accuracy():
    """Test vits14 accuracy at different resolutions"""
    
    print("🚀 vits14 Resolution vs Accuracy Analysis")
    print("=" * 60)
    
    # Ground truth
    ground_truth = {
        'nardo-air_qu-42': 'nardo-air_db-42',
        'nardo-air-r_qu-70': 'nardo-air-r_db-45',
        'vpair_qu-122': 'vpair_db-122'
    }
    
    # Test resolutions
    resolutions = [
        (224, 224),   # Fast
        (280, 280),   # Slightly higher
        (336, 336),   # Medium
        (392, 392),   # Higher
        (448, 448),   # High
        (518, 518),   # vitg14 default
    ]
    
    results = []
    
    for resolution in resolutions:
        print(f"\n{'='*50}")
        print(f"🔍 TESTING: vits14 @ {resolution[0]}x{resolution[1]}")
        print(f"{'='*50}")
        
        try:
            # Create matcher with this resolution
            class TestMatcher(OptimizedDroneMatching):
                def __init__(self, test_resolution):
                    self.satellite_cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
                    self.use_onnx = False
                    self.use_torch_compile = False
                    
                    self.model_name = "dinov2_vits14"
                    self.resolution = test_resolution
                    self.vlad_clusters = 16
                    self.layer = 11
                    self.facet = "value"
                    
                    self.frame_times = []
                    self.match_history = []
                    
                    self._load_models()
                    self._load_satellite_database()
            
            matcher = TestMatcher(resolution)
            
            # Test queries
            cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
            query_dir = f"{cache_dir}/imgs/aerial"
            
            correct = 0
            total = 0
            timings = []
            similarities = []
            extraction_times = []
            vlad_times = []
            
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
                processing_time = (end_time - start_time) * 1000
                extraction_time = result['timing']['extraction']
                vlad_time = result['timing']['vlad']
                
                is_correct = (best_match == expected_match)
                
                print(f"      → {best_match} (sim: {similarity:.3f}) {'✅' if is_correct else '❌'}")
                print(f"      ⏱️  Extract: {extraction_time:.1f}ms, VLAD: {vlad_time:.1f}ms, Total: {processing_time:.1f}ms")
                
                if is_correct:
                    correct += 1
                total += 1
                
                timings.append(processing_time)
                similarities.append(similarity)
                extraction_times.append(extraction_time)
                vlad_times.append(vlad_time)
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            avg_time = np.mean(timings)
            avg_extraction = np.mean(extraction_times)
            avg_vlad = np.mean(vlad_times)
            avg_similarity = np.mean(similarities)
            fps = 1000 / avg_time if avg_time > 0 else 0
            
            # Calculate relative to baseline (224x224)
            baseline_time = 26.4  # From previous tests
            speedup = baseline_time / avg_time if avg_time > 0 else 0
            
            result_data = {
                'resolution': resolution,
                'resolution_str': f"{resolution[0]}x{resolution[1]}",
                'pixels': resolution[0] * resolution[1],
                'accuracy': accuracy,
                'avg_extraction_time_ms': avg_extraction,
                'avg_vlad_time_ms': avg_vlad,
                'avg_total_time_ms': avg_time,
                'fps': fps,
                'avg_similarity': avg_similarity,
                'correct': correct,
                'total': total,
                'speedup_vs_224': speedup
            }
            
            results.append(result_data)
            
            print(f"\n   📊 SUMMARY:")
            print(f"      Accuracy: {accuracy:.1%} ({correct}/{total})")
            print(f"      Extraction: {avg_extraction:.1f}ms")
            print(f"      VLAD: {avg_vlad:.1f}ms") 
            print(f"      Total: {avg_time:.1f}ms")
            print(f"      FPS: {fps:.1f}")
            print(f"      Avg similarity: {avg_similarity:.3f}")
            print(f"      Speedup vs 224x224: {speedup:.2f}x")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Analysis and comparison
    print_resolution_analysis(results)

def print_resolution_analysis(results):
    """Print comprehensive resolution analysis"""
    
    if not results:
        print("❌ No results to analyze")
        return
    
    print(f"\n{'='*100}")
    print("📊 VITS14 RESOLUTION ANALYSIS - Accuracy vs Performance Trade-off")
    print(f"{'='*100}")
    
    # Sort by resolution
    sorted_results = sorted(results, key=lambda x: x['pixels'])
    
    # Print table
    print(f"{'Resolution':<12} {'Pixels':<8} {'Accuracy':<10} {'Extract(ms)':<12} {'Total(ms)':<10} {'FPS':<6} {'Similarity':<10} {'vs vitg14':<10} {'Quality'}")
    print("-" * 100)
    
    for result in sorted_results:
        pixels_k = result['pixels'] // 1000
        
        # Compare to vitg14 baseline (100% accuracy, 1.4 FPS)
        vitg14_comparison = "Better" if result['accuracy'] >= 1.0 and result['fps'] > 1.4 else f"{result['accuracy']:.0%}/{result['fps']:.0f}FPS"
        
        # Quality assessment  
        if result['accuracy'] >= 0.8 and result['fps'] >= 15:
            quality = "🚀 Excellent"
        elif result['accuracy'] >= 0.6 and result['fps'] >= 10:
            quality = "⚡ Good"
        elif result['accuracy'] >= 0.4 and result['fps'] >= 5:
            quality = "📊 Fair"
        else:
            quality = "⚠️  Poor"
        
        print(f"{result['resolution_str']:<12} {pixels_k:<8}K {result['accuracy']:<10.1%} {result['avg_extraction_time_ms']:<12.1f} {result['avg_total_time_ms']:<10.1f} {result['fps']:<6.1f} {result['avg_similarity']:<10.3f} {vitg14_comparison:<10} {quality}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    # Best accuracy
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    print(f"   🎯 Highest Accuracy: {best_accuracy['resolution_str']} ({best_accuracy['accuracy']:.1%}, {best_accuracy['fps']:.1f} FPS)")
    
    # Best real-time performance
    realtime_candidates = [r for r in results if r['fps'] >= 15]
    if realtime_candidates:
        best_realtime = max(realtime_candidates, key=lambda x: x['accuracy'])
        print(f"   🚀 Best Real-time: {best_realtime['resolution_str']} ({best_realtime['accuracy']:.1%}, {best_realtime['fps']:.1f} FPS)")
    
    # Best balance (weighted score)
    best_balanced = max(results, key=lambda x: x['accuracy'] * (x['fps'] / 10))
    print(f"   ⚖️  Best Balance: {best_balanced['resolution_str']} (Score: {best_balanced['accuracy'] * (best_balanced['fps'] / 10):.2f})")
    
    # Accuracy trend analysis
    print(f"\n📈 ACCURACY vs RESOLUTION TREND:")
    baseline_224 = next((r for r in results if r['resolution'] == (224, 224)), None)
    
    if baseline_224:
        for result in sorted_results:
            if result != baseline_224:
                acc_change = result['accuracy'] - baseline_224['accuracy']
                speed_change = baseline_224['fps'] / result['fps']
                
                acc_symbol = "📈" if acc_change > 0 else "📉" if acc_change < 0 else "➡️"
                print(f"   {result['resolution_str']}: {acc_symbol} {acc_change:+.1%} accuracy, {speed_change:.1f}x slower")
    
    # Performance scaling analysis
    print(f"\n⚡ PERFORMANCE SCALING:")
    if len(results) >= 2:
        smallest = min(results, key=lambda x: x['pixels'])
        largest = max(results, key=lambda x: x['pixels'])
        
        pixel_ratio = largest['pixels'] / smallest['pixels']
        time_ratio = largest['avg_extraction_time_ms'] / smallest['avg_extraction_time_ms']
        
        print(f"   📊 {largest['resolution_str']} vs {smallest['resolution_str']}:")
        print(f"      Pixels: {pixel_ratio:.1f}x more")
        print(f"      Time: {time_ratio:.1f}x slower")
        print(f"      Efficiency: {pixel_ratio/time_ratio:.2f} (>1 = super-linear scaling)")
    
    # Recommendations
    print(f"\n🏆 RECOMMENDATIONS:")
    
    # For different use cases
    speed_focused = min(results, key=lambda x: x['avg_total_time_ms'])
    print(f"   💨 Maximum Speed: {speed_focused['resolution_str']} ({speed_focused['fps']:.1f} FPS)")
    
    if realtime_candidates:
        best_realtime_acc = max(realtime_candidates, key=lambda x: x['accuracy'])
        print(f"   🎯 Real-time + Accuracy: {best_realtime_acc['resolution_str']} ({best_realtime_acc['accuracy']:.1%} @ {best_realtime_acc['fps']:.1f} FPS)")
    
    # Find sweet spot (good accuracy + reasonable speed)
    sweet_spots = [r for r in results if r['accuracy'] >= 0.8 and r['fps'] >= 10]
    if sweet_spots:
        sweet_spot = max(sweet_spots, key=lambda x: x['fps'])
        print(f"   🍯 Sweet Spot: {sweet_spot['resolution_str']} - excellent balance of accuracy & speed")
    
    # Compare to vitg14
    print(f"\n⚖️  COMPARISON vs vitg14 @ 518x518 (100% accuracy, 1.4 FPS):")
    
    competitive_results = [r for r in results if r['accuracy'] >= 0.8]
    if competitive_results:
        for result in competitive_results:
            fps_improvement = result['fps'] / 1.4
            print(f"   vits14 @ {result['resolution_str']}: {result['accuracy']:.1%} accuracy, {fps_improvement:.1f}x faster")
    
    # Final recommendation
    print(f"\n🎉 FINAL RECOMMENDATION:")
    
    if best_balanced['accuracy'] >= 0.8 and best_balanced['fps'] >= 10:
        print(f"   🚀 Use vits14 @ {best_balanced['resolution_str']} for optimal balance!")
        print(f"      Achieves {best_balanced['accuracy']:.1%} accuracy at {best_balanced['fps']:.1f} FPS")
        print(f"      Excellent for real-time drone-satellite matching")
    else:
        print(f"   📊 For real-time: Use {speed_focused['resolution_str']} (fastest)")
        if best_accuracy['fps'] >= 5:
            print(f"   🎯 For accuracy: Use {best_accuracy['resolution_str']} (most accurate)")

def main():
    """Main function"""
    test_vits14_resolution_accuracy()

if __name__ == "__main__":
    main()
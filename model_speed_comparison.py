#!/usr/bin/env python3
"""
Quick comparison of DINOv2 model speeds
"""

import torch
import time
import numpy as np
from utilities import DinoV2ExtractFeatures
from configs import device

def benchmark_model(model_name, layer, resolution=(224, 224)):
    """Benchmark a specific DINOv2 model"""
    print(f"\n🔧 Testing {model_name}...")
    
    try:
        # Load model
        extractor = DinoV2ExtractFeatures(model_name, layer, "value", device)
        
        # Create test input
        test_input = torch.randn(1, 3, *resolution).to(device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = extractor(test_input)
        
        # Benchmark
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                features = extractor(test_input)
            
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        print(f"   ✅ {model_name}: {avg_time*1000:.1f}ms, {fps:.1f} FPS")
        print(f"   📊 Feature shape: {features.shape}")
        print(f"   📏 Feature dim: {features.shape[-1]}")
        
        return avg_time, fps, features.shape[-1]
        
    except Exception as e:
        print(f"   ❌ {model_name} failed: {e}")
        return None, None, None

def main():
    print("🚀 DINOv2 Model Speed Comparison")
    print("=" * 50)
    
    # Test different models
    models_to_test = [
        ("dinov2_vits14", 11),  # Small model, last layer
        ("dinov2_vitb14", 11),  # Base model
        ("dinov2_vitl14", 23),  # Large model
        ("dinov2_vitg14", 39),  # Giant model, last layer
    ]
    
    results = {}
    
    for model_name, layer in models_to_test:
        avg_time, fps, feat_dim = benchmark_model(model_name, layer)
        if avg_time is not None:
            results[model_name] = {
                'time': avg_time,
                'fps': fps,
                'feat_dim': feat_dim
            }
    
    # Print comparison
    print(f"\n📊 SPEED COMPARISON (224x224 resolution)")
    print("-" * 60)
    print(f"{'Model':<15} {'Time (ms)':<12} {'FPS':<8} {'Feat Dim':<10} {'Speedup'}")
    print("-" * 60)
    
    baseline_time = results.get('dinov2_vitg14', {}).get('time', 1.0)
    
    for model_name, result in results.items():
        time_ms = result['time'] * 1000
        fps = result['fps']
        feat_dim = result['feat_dim']
        speedup = baseline_time / result['time']
        
        print(f"{model_name:<15} {time_ms:<12.1f} {fps:<8.1f} {feat_dim:<10} {speedup:.1f}x")
    
    print(f"\n💡 KEY INSIGHTS:")
    if 'dinov2_vits14' in results and 'dinov2_vitg14' in results:
        vits_fps = results['dinov2_vits14']['fps']
        vitg_fps = results['dinov2_vitg14']['fps']
        speedup = vits_fps / vitg_fps
        print(f"   🚀 vits14 is {speedup:.1f}x faster than vitg14")
        print(f"   📏 vits14 uses {results['dinov2_vits14']['feat_dim']}D vs vitg14's {results['dinov2_vitg14']['feat_dim']}D")
        print(f"   ⚡ For real-time: vits14 @ 224x224 could achieve {vits_fps:.1f} FPS")

if __name__ == "__main__":
    main()
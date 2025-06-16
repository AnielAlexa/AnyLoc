#!/usr/bin/env python3
"""
Raspberry Pi 4/5 Optimization Analysis for AnyLoc Drone Navigation

Tests different optimization strategies for CPU-only inference:
- Model selection (vits14 vs vitl14 vs vitg14)
- Resolution optimization (speed vs accuracy)
- CPU threading optimization
- Memory usage optimization
- VLAD clustering optimization
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
import psutil
from threading import Thread
import multiprocessing as mp

from utilities import DinoV2ExtractFeatures, VLAD, seed_everything

class RaspberryPiOptimizer:
    """Optimize AnyLoc for Raspberry Pi 4/5 constraints"""
    
    def __init__(self):
        self.device = "cpu"  # Force CPU-only
        
        # RPi-specific configurations
        self.rpi_configs = {
            'ultra_light': {
                'model': 'dinov2_vits14',
                'layer': 11,  # Earlier layer for speed
                'resolution': (112, 112),  # Quarter resolution
                'vlad_clusters': 8,  # Fewer clusters
                'batch_size': 1,
                'threads': 2
            },
            'light': {
                'model': 'dinov2_vits14', 
                'layer': 17,  # Mid layer
                'resolution': (168, 168),  # Half+ resolution
                'vlad_clusters': 12,
                'batch_size': 1,
                'threads': 4
            },
            'balanced': {
                'model': 'dinov2_vits14',
                'layer': 23,  # Full layer
                'resolution': (224, 224),  # Full resolution
                'vlad_clusters': 16,
                'batch_size': 1, 
                'threads': 4
            },
            'quality': {
                'model': 'dinov2_vitl14',
                'layer': 17,  # Earlier for speed
                'resolution': (224, 224),
                'vlad_clusters': 16,
                'batch_size': 1,
                'threads': 4
            }
        }
        
    def get_system_info(self):
        """Get Raspberry Pi system information"""
        
        print("🍓 RASPBERRY PI SYSTEM ANALYSIS")
        print("=" * 50)
        
        # CPU info
        cpu_count = mp.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        print(f"CPU Cores: {cpu_count}")
        if cpu_freq:
            print(f"CPU Frequency: {cpu_freq.current:.0f} MHz (max: {cpu_freq.max:.0f} MHz)")
        
        # Memory info
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        print(f"Memory Usage: {memory.percent:.1f}%")
        
        # Check for RPi-specific files
        rpi_model = "Unknown"
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", 'r') as f:
                rpi_model = f.read().strip().replace('\x00', '')
        
        print(f"Device Model: {rpi_model}")
        
        # Temperature (if available)
        try:
            if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
                with open("/sys/class/thermal/thermal_zone0/temp", 'r') as f:
                    temp = int(f.read().strip()) / 1000
                    print(f"CPU Temperature: {temp:.1f}°C")
        except:
            pass
        
        # Storage
        disk = psutil.disk_usage('/')
        print(f"Storage: {disk.used / (1024**3):.1f}GB / {disk.total / (1024**3):.1f}GB ({disk.percent:.1f}% used)")
        
        print("=" * 50)
        
        return {
            'cpu_cores': cpu_count,
            'ram_gb': memory.total / (1024**3),
            'model': rpi_model
        }
    
    def optimize_cpu_settings(self):
        """Optimize CPU settings for inference"""
        
        print("⚡ Optimizing CPU settings...")
        
        # Set CPU affinity to performance cores (if available)
        cpu_count = mp.cpu_count()
        
        # Set thread counts
        torch.set_num_threads(cpu_count)
        torch.set_num_interop_threads(cpu_count)
        
        # Disable GPU operations
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Memory optimization
        torch.backends.cudnn.enabled = False
        
        print(f"   Threads: {cpu_count}")
        print(f"   CPU-only mode enabled")
        
    def test_model_performance(self, config_name, config):
        """Test performance of specific configuration"""
        
        print(f"\n🔧 Testing {config_name} configuration...")
        print(f"   Model: {config['model']}")
        print(f"   Resolution: {config['resolution']}")
        print(f"   Layer: {config['layer']}")
        print(f"   VLAD clusters: {config['vlad_clusters']}")
        
        try:
            # Setup model
            extractor = DinoV2ExtractFeatures(
                config['model'], config['layer'], "value", self.device
            )
            extractor.dino_model = extractor.dino_model.to(self.device)
            
            # Setup VLAD
            vlad = VLAD(
                num_clusters=config['vlad_clusters'],
                desc_dim=384 if 'vits14' in config['model'] else 1024,
                cache_dir=None
            )
            
            # Create dummy features for VLAD training
            dummy_features = torch.randn(100, vlad.desc_dim)
            vlad.fit(dummy_features)
            
            # Create test image
            test_image = np.random.randint(0, 255, (*config['resolution'], 3), dtype=np.uint8)
            
            # Preprocess
            img_tensor = self._preprocess_image(test_image, config['resolution'])
            
            # Warmup
            print("   Warming up...")
            for _ in range(3):
                with torch.no_grad():
                    features = extractor(img_tensor)
                    _ = vlad.generate(features.squeeze(0))
            
            # Benchmark
            print("   Benchmarking...")
            times = []
            memory_usage = []
            
            for i in range(10):
                # Monitor memory
                memory_before = psutil.virtual_memory().used / (1024**2)
                
                start_time = time.time()
                
                with torch.no_grad():
                    features = extractor(img_tensor)
                    vlad_desc = vlad.generate(features.squeeze(0))
                
                end_time = time.time()
                
                memory_after = psutil.virtual_memory().used / (1024**2)
                
                times.append((end_time - start_time) * 1000)
                memory_usage.append(memory_after - memory_before)
            
            # Calculate statistics
            avg_time = np.mean(times)
            fps = 1000 / avg_time
            avg_memory = np.mean(memory_usage)
            
            results = {
                'config': config_name,
                'avg_time_ms': avg_time,
                'fps': fps,
                'min_time_ms': np.min(times),
                'max_time_ms': np.max(times),
                'memory_mb': avg_memory,
                'model_size': self._estimate_model_size(config['model'])
            }
            
            print(f"   Results: {avg_time:.1f}ms ({fps:.2f} FPS), {avg_memory:.1f}MB memory")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return None
    
    def _preprocess_image(self, image, resolution):
        """Preprocess image for model"""
        
        # Resize
        image_resized = cv2.resize(image, resolution)
        
        # Convert to tensor
        img_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0).to(self.device)
    
    def _estimate_model_size(self, model_name):
        """Estimate model size in memory"""
        
        sizes = {
            'dinov2_vits14': 85,   # MB
            'dinov2_vitl14': 581,  # MB  
            'dinov2_vitg14': 1100  # MB
        }
        
        return sizes.get(model_name, 0)
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive RPi benchmark"""
        
        print("🍓 RASPBERRY PI OPTIMIZATION BENCHMARK")
        print("=" * 80)
        print("Testing different configurations for optimal performance")
        print("=" * 80)
        
        # Get system info
        system_info = self.get_system_info()
        
        # Optimize CPU
        self.optimize_cpu_settings()
        
        # Test all configurations
        results = []
        
        for config_name, config in self.rpi_configs.items():
            result = self.test_model_performance(config_name, config)
            if result:
                results.append(result)
        
        # Analyze results
        self._analyze_results(results, system_info)
        
        return results
    
    def _analyze_results(self, results, system_info):
        """Analyze benchmark results and provide recommendations"""
        
        print(f"\n🏆 RASPBERRY PI OPTIMIZATION ANALYSIS")
        print("=" * 70)
        
        if not results:
            print("❌ No successful benchmarks")
            return
        
        # Sort by FPS
        results_sorted = sorted(results, key=lambda x: x['fps'], reverse=True)
        
        print(f"📊 Performance Comparison:")
        print(f"{'Config':<12} {'FPS':<8} {'Time(ms)':<10} {'Memory(MB)':<12} {'Model(MB)':<11} {'Usable'}")
        print("-" * 70)
        
        for result in results_sorted:
            usable = "✅ Good" if result['fps'] >= 2.0 else "⚠️ Slow" if result['fps'] >= 0.5 else "❌ Too slow"
            
            print(f"{result['config']:<12} {result['fps']:<8.2f} {result['avg_time_ms']:<10.1f} "
                  f"{result['memory_mb']:<12.1f} {result['model_size']:<11} {usable}")
        
        # Memory analysis
        total_ram_gb = system_info['ram_gb']
        print(f"\n💾 Memory Analysis (Total: {total_ram_gb:.1f}GB):")
        
        for result in results_sorted:
            total_memory = result['model_size'] + result['memory_mb']
            memory_percent = (total_memory / 1024) / total_ram_gb * 100
            
            if memory_percent < 25:
                status = "✅ Safe"
            elif memory_percent < 50:
                status = "⚠️ Moderate" 
            else:
                status = "❌ High"
            
            print(f"   {result['config']}: {total_memory:.0f}MB ({memory_percent:.1f}%) {status}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS FOR RASPBERRY PI:")
        
        # Find best options
        fast_enough = [r for r in results_sorted if r['fps'] >= 1.0]
        memory_safe = [r for r in results_sorted if (r['model_size'] + r['memory_mb']) / 1024 / total_ram_gb < 0.5]
        
        if fast_enough and memory_safe:
            best_overall = None
            for result in fast_enough:
                if result in memory_safe:
                    best_overall = result
                    break
            
            if best_overall:
                print(f"🎯 BEST OPTION: {best_overall['config']}")
                print(f"   Performance: {best_overall['fps']:.1f} FPS")
                print(f"   Memory usage: {(best_overall['model_size'] + best_overall['memory_mb']):.0f}MB")
                print(f"   Suitable for: Real-time navigation")
        
        # Specific recommendations
        print(f"\n📋 Use Case Recommendations:")
        
        if any(r['fps'] >= 2.0 for r in results):
            fast_config = max(results, key=lambda x: x['fps'])
            print(f"   🚀 Fast navigation: {fast_config['config']} ({fast_config['fps']:.1f} FPS)")
        
        if any(r['fps'] >= 0.5 for r in results):
            balanced = min([r for r in results if r['fps'] >= 0.5], key=lambda x: x['model_size'])
            print(f"   ⚖️ Balanced: {balanced['config']} ({balanced['fps']:.1f} FPS)")
        
        print(f"\n⚠️ IMPORTANT LIMITATIONS:")
        print(f"   • RPi performance is 10-20x slower than Jetson Orin")
        print(f"   • Consider shorter routes (1-5km max)")
        print(f"   • Use fewer reference images (50-200 max)")
        print(f"   • Pre-process everything offline")
        print(f"   • Monitor CPU temperature during flight")
        
        # Jetson comparison
        best_rpi_fps = max(r['fps'] for r in results)
        jetson_equivalent_fps = 12.6  # vitl14 on Jetson
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"   Raspberry Pi (best): {best_rpi_fps:.1f} FPS")
        print(f"   Jetson Orin Nano:    {jetson_equivalent_fps:.1f} FPS")
        print(f"   Performance gap:     {jetson_equivalent_fps/best_rpi_fps:.1f}x slower")

def main():
    """Main benchmark function"""
    
    print("🔬 Raspberry Pi 4/5 Optimization Analysis")
    print("=" * 60)
    
    seed_everything(42)
    
    try:
        optimizer = RaspberryPiOptimizer()
        results = optimizer.run_comprehensive_benchmark()
        
        print(f"\n✅ Benchmark completed!")
        print(f"🍓 Check results above for optimal RPi configuration")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
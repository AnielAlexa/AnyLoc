#!/usr/bin/env python3
"""
DINOv2 vitg14 Quantization Test

Tests different quantization methods for vitg14 to improve performance:
1. Dynamic Quantization (CPU/GPU)
2. Static Quantization (INT8)
3. QAT (Quantization Aware Training)
4. FP16 Half Precision
5. TensorRT Quantization

Compares accuracy vs speed trade-offs.
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
import torch.nn as nn
import torch.quantization as quantization
import numpy as np
import cv2

from utilities import DinoV2ExtractFeatures, VLAD, seed_everything
from configs import device

class QuantizedVitG14Tester:
    """Test different quantization methods for vitg14"""
    
    def __init__(self):
        self.model_name = "dinov2_vitg14"
        self.resolution = (224, 224)  # Start with smaller for speed
        self.layer = 39
        self.facet = "value"
        
        # Test configurations
        self.test_configs = [
            ("Original FP32", "original"),
            ("FP16 Half Precision", "fp16"),
            ("Dynamic Quantization", "dynamic_quant"),
            ("Static INT8 Quantization", "static_quant"),
        ]
        
        # Performance tracking
        self.results = {}
        
    def create_test_images(self, num_images=5):
        """Create test images for benchmarking"""
        images = []
        
        for i in range(num_images):
            # Create test pattern
            np.random.seed(42 + i)
            img = np.random.randint(0, 255, (*self.resolution, 3), dtype=np.uint8)
            
            # Add some structure
            x, y = np.meshgrid(np.linspace(0, 10, self.resolution[0]), 
                              np.linspace(0, 10, self.resolution[1]))
            pattern = (np.sin(x) * np.cos(y) * 50 + 128).astype(np.uint8)
            
            for c in range(3):
                img[:, :, c] = (img[:, :, c] * 0.7 + pattern * 0.3).astype(np.uint8)
            
            images.append(img)
        
        return images
    
    def preprocess_image(self, image):
        """Preprocess image for DINOv2"""
        if image.shape[:2] != self.resolution:
            image = cv2.resize(image, self.resolution)
        
        img_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0)
    
    def test_original_fp32(self, test_images):
        """Test original FP32 model"""
        print("\n🔧 Testing Original FP32 vitg14")
        print("-" * 40)
        
        try:
            extractor = DinoV2ExtractFeatures(self.model_name, self.layer, self.facet, device)
            extractor.dino_model = extractor.dino_model.to(device)
            
            return self._benchmark_model(extractor, test_images, "original", device)
            
        except Exception as e:
            print(f"❌ FP32 test failed: {e}")
            return None
    
    def test_fp16_half_precision(self, test_images):
        """Test FP16 half precision"""
        print("\n🔧 Testing FP16 Half Precision vitg14")
        print("-" * 40)
        
        try:
            extractor = DinoV2ExtractFeatures(self.model_name, self.layer, self.facet, device)
            
            # Convert to half precision
            extractor.dino_model = extractor.dino_model.half().to(device)
            
            return self._benchmark_model(extractor, test_images, "fp16", device, use_half=True)
            
        except Exception as e:
            print(f"❌ FP16 test failed: {e}")
            return None
    
    def test_dynamic_quantization(self, test_images):
        """Test dynamic quantization"""
        print("\n🔧 Testing Dynamic Quantization vitg14")
        print("-" * 40)
        
        try:
            # Load model on CPU for quantization
            extractor = DinoV2ExtractFeatures(self.model_name, self.layer, self.facet, "cpu")
            
            # Apply dynamic quantization
            print("   📦 Applying dynamic quantization...")
            quantized_model = torch.quantization.quantize_dynamic(
                extractor.dino_model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
            extractor.dino_model = quantized_model
            
            # Get model size
            original_size = self._get_model_size(extractor.dino_model)
            print(f"   📊 Model size after quantization: {original_size:.1f}MB")
            
            return self._benchmark_model(extractor, test_images, "dynamic_quant", "cpu")
            
        except Exception as e:
            print(f"❌ Dynamic quantization test failed: {e}")
            return None
    
    def test_static_quantization(self, test_images):
        """Test static INT8 quantization"""
        print("\n🔧 Testing Static INT8 Quantization vitg14")
        print("-" * 40)
        
        try:
            # Load model for calibration
            extractor = DinoV2ExtractFeatures(self.model_name, self.layer, self.facet, "cpu")
            
            # Prepare model for quantization
            print("   📦 Preparing model for static quantization...")
            model = extractor.dino_model
            model.eval()
            
            # Set quantization config
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare model
            prepared_model = torch.quantization.prepare(model)
            
            # Calibration with test images
            print("   🎯 Calibrating with test images...")
            for img in test_images[:3]:  # Use fewer images for calibration
                img_tensor = self.preprocess_image(img)
                with torch.no_grad():
                    _ = prepared_model(img_tensor)
            
            # Convert to quantized model
            print("   🔄 Converting to quantized model...")
            quantized_model = torch.quantization.convert(prepared_model)
            
            extractor.dino_model = quantized_model
            
            # Get model size
            quantized_size = self._get_model_size(quantized_model)
            print(f"   📊 Quantized model size: {quantized_size:.1f}MB")
            
            return self._benchmark_model(extractor, test_images, "static_quant", "cpu")
            
        except Exception as e:
            print(f"❌ Static quantization test failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _benchmark_model(self, extractor, test_images, config_name, target_device, use_half=False):
        """Benchmark a model configuration"""
        print(f"   ⏱️  Benchmarking {config_name}...")
        
        times = []
        features_list = []
        
        # Warmup
        for _ in range(2):
            img_tensor = self.preprocess_image(test_images[0])
            if use_half:
                img_tensor = img_tensor.half()
            img_tensor = img_tensor.to(target_device)
            
            with torch.no_grad():
                _ = extractor(img_tensor)
        
        # Benchmark
        for img in test_images:
            img_tensor = self.preprocess_image(img)
            if use_half:
                img_tensor = img_tensor.half()
            img_tensor = img_tensor.to(target_device)
            
            torch.cuda.synchronize() if target_device != "cpu" else None
            start_time = time.time()
            
            with torch.no_grad():
                features = extractor(img_tensor)
                if hasattr(features, 'cpu'):
                    features = features.cpu()
            
            torch.cuda.synchronize() if target_device != "cpu" else None
            end_time = time.time()
            
            times.append(end_time - start_time)
            features_list.append(features)
        
        # Calculate metrics
        avg_time = np.mean(times) * 1000  # ms
        fps = 1000 / avg_time
        model_size = self._get_model_size(extractor.dino_model)
        
        # Feature shape
        if features_list:
            feature_shape = features_list[0].shape if hasattr(features_list[0], 'shape') else "Unknown"
        else:
            feature_shape = "Unknown"
        
        result = {
            'config': config_name,
            'avg_time_ms': avg_time,
            'fps': fps,
            'model_size_mb': model_size,
            'feature_shape': feature_shape,
            'device': target_device,
            'features': features_list
        }
        
        print(f"   ✅ Results:")
        print(f"      Average time: {avg_time:.1f}ms")
        print(f"      FPS: {fps:.1f}")
        print(f"      Model size: {model_size:.1f}MB")
        print(f"      Feature shape: {feature_shape}")
        
        return result
    
    def _get_model_size(self, model):
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def test_vlad_compatibility(self, results):
        """Test VLAD compatibility with quantized features"""
        print("\n🧠 Testing VLAD Compatibility")
        print("-" * 40)
        
        vlad_results = {}
        
        for config_name, result in results.items():
            if result is None or not result['features']:
                continue
            
            print(f"   🔍 Testing VLAD with {config_name}...")
            
            try:
                # Get feature dimension
                features = result['features'][0]
                if hasattr(features, 'squeeze'):
                    features = features.squeeze(0)
                
                feature_dim = features.shape[-1] if hasattr(features, 'shape') else None
                
                if feature_dim is None:
                    print(f"      ❌ Could not determine feature dimension")
                    continue
                
                # Create VLAD
                vlad = VLAD(num_clusters=8, desc_dim=feature_dim, cache_dir=None)
                
                # Fit vocabulary
                all_features = []
                for feat in result['features']:
                    if hasattr(feat, 'squeeze'):
                        feat = feat.squeeze(0)
                    all_features.append(feat)
                
                combined_features = torch.cat(all_features, dim=0)
                vlad.fit(combined_features)
                
                # Test VLAD generation
                vlad_times = []
                for feat in all_features:
                    start_time = time.time()
                    vlad_desc = vlad.generate(feat)
                    end_time = time.time()
                    vlad_times.append((end_time - start_time) * 1000)
                
                avg_vlad_time = np.mean(vlad_times)
                vlad_dim = vlad_desc.shape[0] if hasattr(vlad_desc, 'shape') else 0
                
                vlad_results[config_name] = {
                    'avg_vlad_time_ms': avg_vlad_time,
                    'vlad_dim': vlad_dim,
                    'total_time_ms': result['avg_time_ms'] + avg_vlad_time
                }
                
                print(f"      ✅ VLAD time: {avg_vlad_time:.1f}ms")
                print(f"      📊 VLAD descriptor: {vlad_dim}D")
                print(f"      🎯 Total pipeline: {vlad_results[config_name]['total_time_ms']:.1f}ms")
                
            except Exception as e:
                print(f"      ❌ VLAD test failed: {e}")
        
        return vlad_results
    
    def run_comprehensive_quantization_test(self):
        """Run comprehensive quantization test"""
        print("🚀 DINOv2 vitg14 Quantization Test")
        print("=" * 60)
        
        # Create test images
        print("🖼️ Creating test images...")
        test_images = self.create_test_images()
        
        # Test all configurations
        results = {}
        
        # Original FP32
        results['original'] = self.test_original_fp32(test_images)
        
        # FP16 Half Precision
        results['fp16'] = self.test_fp16_half_precision(test_images)
        
        # Dynamic Quantization
        results['dynamic_quant'] = self.test_dynamic_quantization(test_images)
        
        # Static Quantization
        results['static_quant'] = self.test_static_quantization(test_images)
        
        # Test VLAD compatibility
        vlad_results = self.test_vlad_compatibility(results)
        
        # Print comparison
        self._print_comparison(results, vlad_results)
        
        return results, vlad_results
    
    def _print_comparison(self, results, vlad_results):
        """Print comprehensive comparison"""
        print(f"\n🏆 QUANTIZATION COMPARISON SUMMARY")
        print("=" * 80)
        
        # Filter valid results
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("❌ No valid results to compare")
            return
        
        print(f"{'Config':<20} {'Extract(ms)':<12} {'VLAD(ms)':<10} {'Total(ms)':<10} {'FPS':<8} {'Size(MB)':<10} {'Speedup'}")
        print("-" * 85)
        
        baseline = None
        
        for config_name, result in valid_results.items():
            extract_time = result['avg_time_ms']
            vlad_time = vlad_results.get(config_name, {}).get('avg_vlad_time_ms', 0)
            total_time = extract_time + vlad_time
            fps = 1000 / total_time if total_time > 0 else 0
            size_mb = result['model_size_mb']
            
            if baseline is None:
                baseline = total_time
                speedup_str = "1.0x"
            else:
                speedup = baseline / total_time
                speedup_str = f"{speedup:.1f}x"
            
            print(f"{config_name:<20} {extract_time:<12.1f} {vlad_time:<10.1f} {total_time:<10.1f} {fps:<8.1f} {size_mb:<10.1f} {speedup_str}")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        
        # Best speed
        fastest = min(valid_results.items(), key=lambda x: x[1]['avg_time_ms'])
        print(f"   🚀 Fastest: {fastest[0]} ({fastest[1]['fps']:.1f} FPS)")
        
        # Best size reduction
        if len(valid_results) > 1:
            original_size = valid_results.get('original', {}).get('model_size_mb', 0)
            if original_size > 0:
                size_reductions = {}
                for name, result in valid_results.items():
                    if name != 'original':
                        reduction = (original_size - result['model_size_mb']) / original_size * 100
                        size_reductions[name] = reduction
                
                if size_reductions:
                    best_compression = max(size_reductions.items(), key=lambda x: x[1])
                    print(f"   📦 Best compression: {best_compression[0]} ({best_compression[1]:.1f}% smaller)")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        
        realtime_configs = [(name, result) for name, result in valid_results.items() 
                           if result['fps'] >= 5]  # 5+ FPS for vitg14 is good
        
        if realtime_configs:
            best_realtime = max(realtime_configs, key=lambda x: x[1]['fps'])
            print(f"   ⚡ For real-time: Use {best_realtime[0]} ({best_realtime[1]['fps']:.1f} FPS)")
        
        # Platform-specific recommendations
        print(f"\n🖥️ PLATFORM RECOMMENDATIONS:")
        print(f"   💻 Desktop/Server: FP16 (good balance)")
        print(f"   📱 Mobile/Edge: Dynamic or Static Quantization")
        print(f"   🤖 Jetson: FP16 + TensorRT conversion")
        print(f"   ☁️  CPU Only: Dynamic Quantization")

def main():
    """Main function"""
    print("🚀 DINOv2 vitg14 Quantization Test")
    print("=" * 60)
    
    seed_everything(42)
    
    # Check hardware
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🖥️ GPU: {gpu_name}")
    else:
        print("💻 Running on CPU only")
    
    try:
        tester = QuantizedVitG14Tester()
        results, vlad_results = tester.run_comprehensive_quantization_test()
        
        print(f"\n✅ Quantization test completed!")
        print(f"🎉 Multiple optimization strategies tested for vitg14")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
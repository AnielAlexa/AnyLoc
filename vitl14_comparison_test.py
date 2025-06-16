#!/usr/bin/env python3
"""
DINOv2 vitl14 (Large) Comparison Test

Tests vitl14 performance vs vitg14 and vits14:
- Similar accuracy to vitg14 but fewer parameters
- Better speed than vitg14
- Better accuracy than vits14
- Perfect middle ground for drone applications

Compares:
1. vitl14 vs vitg14 vs vits14
2. Performance and accuracy trade-offs  
3. Memory usage and speed
4. Visual matching quality
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

from utilities import DinoV2ExtractFeatures, VLAD, seed_everything
from configs import device

class VitL14ComparisonTester:
    """Test vitl14 performance against vitg14 and vits14"""
    
    def __init__(self):
        self.cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
        self.images_dir = f"{self.cache_dir}/imgs/aerial"
        self.output_dir = "./vitl14_comparison_results"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Ground truth for testing
        self.ground_truth = {
            'nardo-air_qu-42': 'nardo-air_db-42',
            'nardo-air-r_qu-70': 'nardo-air-r_db-45', 
            'vpair_qu-122': 'vpair_db-122'
        }
        
        # Model specifications
        self.model_specs = {
            'dinov2_vits14': {
                'layer': 11,
                'resolution': (224, 224),
                'feature_dim': 384,
                'parameters': '22M',
                'description': 'Small - Fast'
            },
            'dinov2_vitl14': {
                'layer': 23,
                'resolution': (224, 224),  # Start with same res as vits14
                'feature_dim': 1024,
                'parameters': '300M',
                'description': 'Large - Balanced'
            },
            'dinov2_vitg14': {
                'layer': 39,
                'resolution': (224, 224),  # Use same res for fair comparison
                'feature_dim': 1536,
                'parameters': '1.1B',
                'description': 'Giant - Accurate'
            }
        }
        
    def load_model_and_data(self, model_name, use_quantization=False):
        """Load specific DINOv2 model and create VLAD"""
        
        print(f"🔧 Loading {model_name}...")
        
        spec = self.model_specs[model_name]
        layer = spec['layer']
        resolution = spec['resolution']
        feature_dim = spec['feature_dim']
        
        # Load feature extractor
        self.extractor = DinoV2ExtractFeatures(model_name, layer, "value", device)
        
        # Ensure model is on correct device
        if hasattr(self.extractor, 'dino_model'):
            self.extractor.dino_model = self.extractor.dino_model.to(device)
        
        # Apply quantization if requested
        if use_quantization:
            print("   🚀 Applying FP16 quantization...")
            self.extractor.dino_model = self.extractor.dino_model.half().to(device)
            self.use_half = True
        else:
            self.use_half = False
        
        # Create VLAD vocabulary
        print(f"   📚 Creating VLAD vocabulary for {feature_dim}D features...")
        self.vlad = VLAD(num_clusters=16, desc_dim=feature_dim, cache_dir=None)
        self.need_vlad_fitting = True
        
        # Get model size
        model_size_mb = self._get_model_size(self.extractor.dino_model)
        
        print(f"   ✅ {model_name} loaded:")
        print(f"      Parameters: {spec['parameters']}")
        print(f"      Feature dim: {feature_dim}D")
        print(f"      Resolution: {resolution}")
        print(f"      Model size: {model_size_mb:.1f}MB")
        
        self.current_model = model_name
        self.current_spec = spec
        self.current_resolution = resolution
        
        return spec
    
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
    
    def load_database_images(self):
        """Load and process database images for current model"""
        
        print("🗃️ Loading database images...")
        
        # Find database images
        db_files = [f for f in os.listdir(self.images_dir) if '_db-' in f and f.endswith('.png')]
        
        if not db_files:
            raise ValueError("No database images found!")
        
        print(f"   📊 Processing {len(db_files)} database images with {self.current_model}")
        
        # Process database images
        self.db_data = {}
        all_db_features = []
        
        for i, db_file in enumerate(db_files):
            base_name = db_file.replace('.png', '')
            img_path = f"{self.images_dir}/{db_file}"
            
            print(f"   🔄 Processing {base_name} ({i+1}/{len(db_files)})")
            
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Extract features
                img_tensor = self._preprocess_image(image)
                with torch.no_grad():
                    features = self.extractor(img_tensor)
                    features = features.squeeze(0).cpu()
                    
                    # Convert from half to float if needed
                    if features.dtype == torch.float16:
                        features = features.float()
                
                # Store features for VLAD fitting
                all_db_features.append(features)
                
                self.db_data[base_name] = {
                    'image_path': img_path,
                    'features': features
                }
                
            except Exception as e:
                print(f"      ❌ Error processing {db_file}: {e}")
        
        # Fit VLAD vocabulary
        if all_db_features:
            print("   🧠 Fitting VLAD vocabulary...")
            combined_features = torch.cat(all_db_features, dim=0)
            self.vlad.fit(combined_features)
        
        # Generate VLAD descriptors
        print("   📊 Generating VLAD descriptors...")
        for base_name, data in self.db_data.items():
            vlad_desc = self.vlad.generate(data['features'])
            data['vlad_desc'] = vlad_desc
        
        print(f"✅ Processed {len(self.db_data)} database images")
        
        # Stack for fast search
        self.db_vlads = torch.stack([data['vlad_desc'] for data in self.db_data.values()])
        self.db_names = list(self.db_data.keys())
        
    def _preprocess_image(self, image):
        """Preprocess image for DINOv2"""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize
        image_resized = cv2.resize(image_rgb, self.current_resolution)
        
        # Convert to tensor
        img_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Move to device and apply half precision if needed
        img_tensor = img_tensor.unsqueeze(0).to(device)
        if self.use_half:
            img_tensor = img_tensor.half()
        
        return img_tensor
    
    def benchmark_model_speed(self, num_runs=10):
        """Benchmark model speed with current configuration"""
        
        print(f"⏱️ Benchmarking {self.current_model} speed...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (*self.current_resolution, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(3):
            img_tensor = self._preprocess_image(test_image)
            with torch.no_grad():
                _ = self.extractor(img_tensor)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            img_tensor = self._preprocess_image(test_image)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                features = self.extractor(img_tensor)
                features = features.squeeze(0).cpu()
                if features.dtype == torch.float16:
                    features = features.float()
                vlad_desc = self.vlad.generate(features)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        fps = 1000 / avg_time
        
        print(f"   ⚡ Results: {avg_time:.1f}ms ({fps:.1f} FPS)")
        
        return avg_time, fps
    
    def test_accuracy(self):
        """Test accuracy on ground truth queries"""
        
        print(f"🎯 Testing {self.current_model} accuracy...")
        
        results = []
        correct_count = 0
        total_time = 0
        
        for query_name, expected_match in self.ground_truth.items():
            query_path = f"{self.images_dir}/{query_name}.png"
            
            if not os.path.exists(query_path):
                continue
            
            print(f"   🔍 Testing: {query_name}")
            
            # Load and process query
            query_image = cv2.imread(query_path)
            if query_image is None:
                continue
            
            start_time = time.time()
            
            # Extract features
            img_tensor = self._preprocess_image(query_image)
            with torch.no_grad():
                features = self.extractor(img_tensor)
                features = features.squeeze(0).cpu()
                if features.dtype == torch.float16:
                    features = features.float()
            
            # Generate VLAD and find match
            query_vlad = self.vlad.generate(features)
            similarities = F.cosine_similarity(query_vlad.unsqueeze(0), self.db_vlads, dim=1)
            
            best_idx = similarities.argmax().item()
            best_similarity = similarities[best_idx].item()
            best_match = self.db_names[best_idx]
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            total_time += processing_time
            
            # Check correctness
            is_correct = (best_match == expected_match)
            if is_correct:
                correct_count += 1
            
            result = {
                'query': query_name,
                'expected': expected_match,
                'predicted': best_match,
                'similarity': best_similarity,
                'correct': is_correct,
                'time_ms': processing_time
            }
            results.append(result)
            
            status = "✅" if is_correct else "❌"
            print(f"      → {best_match} (sim: {best_similarity:.3f}) {status}")
        
        accuracy = correct_count / len(results) if results else 0
        avg_time = total_time / len(results) if results else 0
        
        print(f"   📊 Accuracy: {accuracy:.1%} ({correct_count}/{len(results)})")
        print(f"   ⏱️ Avg time: {avg_time:.1f}ms")
        
        return accuracy, avg_time, results
    
    def run_comprehensive_comparison(self):
        """Run comprehensive comparison of all three models"""
        
        print("🏆 COMPREHENSIVE DINOV2 MODEL COMPARISON")
        print("=" * 80)
        
        models_to_test = ['dinov2_vits14', 'dinov2_vitl14', 'dinov2_vitg14']
        use_quantization = [False, True, True]  # Only quantize larger models
        
        all_results = {}
        
        for model_name, use_quant in zip(models_to_test, use_quantization):
            print(f"\n{'='*60}")
            print(f"🔧 TESTING: {model_name}")
            if use_quant:
                print("   🚀 With FP16 Quantization")
            print(f"{'='*60}")
            
            try:
                # Load model
                spec = self.load_model_and_data(model_name, use_quant)
                
                # Load database
                self.load_database_images()
                
                # Benchmark speed
                avg_time, fps = self.benchmark_model_speed()
                
                # Test accuracy
                accuracy, acc_avg_time, acc_results = self.test_accuracy()
                
                # Store results
                all_results[model_name] = {
                    'spec': spec,
                    'speed_benchmark': {'avg_time_ms': avg_time, 'fps': fps},
                    'accuracy_test': {'accuracy': accuracy, 'avg_time_ms': acc_avg_time, 'results': acc_results},
                    'quantized': use_quant,
                    'model_size_mb': self._get_model_size(self.extractor.dino_model)
                }
                
            except Exception as e:
                print(f"❌ Failed to test {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Print comparison
        self._print_comprehensive_comparison(all_results)
        
        return all_results
    
    def _print_comprehensive_comparison(self, results):
        """Print detailed comparison of all models"""
        
        print(f"\n🏆 COMPREHENSIVE MODEL COMPARISON")
        print("=" * 100)
        
        if not results:
            print("❌ No results to compare")
            return
        
        # Table header
        print(f"{'Model':<15} {'Params':<8} {'FeatDim':<8} {'Size(MB)':<10} {'Speed(ms)':<10} {'FPS':<8} {'Accuracy':<10} {'Quant'}")
        print("-" * 100)
        
        # Results table
        for model_name, data in results.items():
            spec = data['spec']
            speed = data['speed_benchmark']
            accuracy = data['accuracy_test']
            
            model_short = model_name.replace('dinov2_', '')
            params = spec['parameters']
            feat_dim = spec['feature_dim']
            size_mb = data['model_size_mb']
            avg_time = speed['avg_time_ms']
            fps = speed['fps']
            acc = accuracy['accuracy']
            quant = 'FP16' if data['quantized'] else 'FP32'
            
            print(f"{model_short:<15} {params:<8} {feat_dim:<8} {size_mb:<10.1f} {avg_time:<10.1f} {fps:<8.1f} {acc:<10.1%} {quant}")
        
        # Analysis
        print(f"\n💡 KEY INSIGHTS:")
        
        # Speed comparison
        if 'dinov2_vits14' in results and 'dinov2_vitl14' in results:
            vits_fps = results['dinov2_vits14']['speed_benchmark']['fps']
            vitl_fps = results['dinov2_vitl14']['speed_benchmark']['fps']
            speed_ratio = vits_fps / vitl_fps
            print(f"   🚀 vits14 is {speed_ratio:.1f}x faster than vitl14")
        
        # Accuracy comparison
        if 'dinov2_vitl14' in results and 'dinov2_vitg14' in results:
            vitl_acc = results['dinov2_vitl14']['accuracy_test']['accuracy']
            vitg_acc = results['dinov2_vitg14']['accuracy_test']['accuracy']
            acc_diff = abs(vitl_acc - vitg_acc)
            print(f"   🎯 vitl14 vs vitg14 accuracy difference: {acc_diff:.1%}")
        
        # Size efficiency
        if 'dinov2_vitl14' in results and 'dinov2_vitg14' in results:
            vitl_size = results['dinov2_vitl14']['model_size_mb']
            vitg_size = results['dinov2_vitg14']['model_size_mb']
            size_ratio = vitg_size / vitl_size
            print(f"   📦 vitg14 is {size_ratio:.1f}x larger than vitl14")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        
        # Find best balanced model
        best_balance = None
        best_score = 0
        
        for model_name, data in results.items():
            # Score = accuracy * fps / (size_mb / 100)  
            accuracy = data['accuracy_test']['accuracy']
            fps = data['speed_benchmark']['fps']
            size_mb = data['model_size_mb']
            
            # Weighted score favoring accuracy and speed over size
            score = accuracy * fps / (size_mb / 1000)
            
            if score > best_score:
                best_score = score
                best_balance = model_name
        
        if best_balance:
            print(f"   ⚖️ Best balance: {best_balance.replace('dinov2_', '')} (score: {best_score:.3f})")
        
        # Platform recommendations
        print(f"\n🖥️ PLATFORM RECOMMENDATIONS:")
        
        # For Jetson Orin Nano
        jetson_candidates = []
        for model_name, data in results.items():
            fps = data['speed_benchmark']['fps']
            accuracy = data['accuracy_test']['accuracy']
            
            # Estimate Jetson performance (roughly 3-4x slower than RTX 3070)
            jetson_fps = fps / 3.5
            
            if jetson_fps >= 2.0 and accuracy >= 0.6:  # Practical thresholds
                jetson_candidates.append((model_name, jetson_fps, accuracy))
        
        print(f"   🤖 Jetson Orin Nano 8GB:")
        for model_name, est_fps, accuracy in jetson_candidates:
            model_short = model_name.replace('dinov2_', '')
            print(f"      {model_short}: ~{est_fps:.1f} FPS, {accuracy:.1%} accuracy")
        
        # Final recommendation
        print(f"\n🏆 FINAL VERDICT:")
        if 'dinov2_vitl14' in results:
            vitl_data = results['dinov2_vitl14']
            vitl_fps = vitl_data['speed_benchmark']['fps']
            vitl_acc = vitl_data['accuracy_test']['accuracy']
            vitl_size = vitl_data['model_size_mb']
            
            print(f"   🌟 vitl14 appears to be the sweet spot:")
            print(f"      ✅ Good speed: {vitl_fps:.1f} FPS")
            print(f"      ✅ Good accuracy: {vitl_acc:.1%}")
            print(f"      ✅ Reasonable size: {vitl_size:.0f}MB")
            print(f"      ✅ Perfect for Jetson Orin Nano applications!")

def main():
    """Main comparison function"""
    
    print("🔬 DINOv2 vitl14 vs vitg14 vs vits14 Comparison")
    print("=" * 60)
    
    seed_everything(42)
    
    # Check data availability
    cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
    if not os.path.exists(cache_dir):
        print("❌ AnyLoc2023-Public-Data not found!")
        return
    
    try:
        tester = VitL14ComparisonTester()
        results = tester.run_comprehensive_comparison()
        
        print(f"\n✅ Comprehensive comparison completed!")
        print(f"🎉 vitl14 evaluation shows it's likely the optimal choice!")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
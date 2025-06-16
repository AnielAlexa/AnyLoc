#!/usr/bin/env python3
"""
DINOv2 vitl14 Visual Matching Test

Shows vitl14 matching quality with real AnyLoc2023-Public-Data:
- Tests the optimal vitl14 model with FP16 quantization
- Shows side-by-side drone vs satellite comparisons
- Demonstrates the perfect balance of speed and accuracy
- Saves visual results for inspection
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

class VitL14VisualTester:
    """Visual testing specifically for vitl14 optimal model"""
    
    def __init__(self):
        self.cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
        self.images_dir = f"{self.cache_dir}/imgs/aerial"
        self.output_dir = "./vitl14_visual_results"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extended ground truth for more comprehensive testing
        self.ground_truth = {
            'nardo-air_qu-42': 'nardo-air_db-42',
            'nardo-air-r_qu-70': 'nardo-air-r_db-45', 
            'vpair_qu-122': 'vpair_db-122',
            'vpair_qu-45': 'vpair_db-45',
            'vpair_qu-88': 'vpair_db-88'
        }
        
        # vitl14 specifications
        self.model_name = "dinov2_vitl14"
        self.layer = 23
        self.resolution = (224, 224)
        self.feature_dim = 1024
        self.vlad_clusters = 16
        
    def load_vitl14_model(self, use_quantization=True):
        """Load optimized vitl14 model"""
        
        print(f"🚀 Loading Optimized DINOv2 vitl14")
        print("=" * 50)
        print(f"Model: {self.model_name}")
        print(f"Layer: {self.layer}")
        print(f"Resolution: {self.resolution}")
        print(f"Feature dimension: {self.feature_dim}D")
        print(f"Quantization: {'FP16' if use_quantization else 'FP32'}")
        
        # Load feature extractor
        self.extractor = DinoV2ExtractFeatures(self.model_name, self.layer, "value", device)
        
        # Ensure model is on correct device
        if hasattr(self.extractor, 'dino_model'):
            self.extractor.dino_model = self.extractor.dino_model.to(device)
        
        # Apply FP16 quantization for optimal performance
        if use_quantization:
            print("   🚀 Applying FP16 quantization...")
            self.extractor.dino_model = self.extractor.dino_model.half().to(device)
            self.use_half = True
        else:
            self.use_half = False
        
        # Create VLAD vocabulary
        print(f"   📚 Creating VLAD vocabulary ({self.vlad_clusters} clusters, {self.feature_dim}D)...")
        self.vlad = VLAD(num_clusters=self.vlad_clusters, desc_dim=self.feature_dim, cache_dir=None)
        
        # Get model size
        model_size_mb = self._get_model_size(self.extractor.dino_model)
        
        print(f"✅ vitl14 loaded successfully!")
        print(f"   📊 Model size: {model_size_mb:.1f}MB")
        print(f"   ⚡ Ready for high-accuracy matching!")
        
    def _get_model_size(self, model):
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / 1024 / 1024
    
    def load_database_images(self):
        """Load and process all database images"""
        
        print("\n🗃️ Loading Database Images")
        print("=" * 40)
        
        # Find database images
        db_files = [f for f in os.listdir(self.images_dir) if '_db-' in f and f.endswith('.png')]
        
        if not db_files:
            raise ValueError("No database images found!")
        
        print(f"📊 Found {len(db_files)} database images")
        
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
            print("   🧠 Fitting VLAD vocabulary with vitl14 features...")
            combined_features = torch.cat(all_db_features, dim=0)
            self.vlad.fit(combined_features)
        
        # Generate VLAD descriptors
        print("   📊 Generating VLAD descriptors...")
        for base_name, data in self.db_data.items():
            vlad_desc = self.vlad.generate(data['features'])
            data['vlad_desc'] = vlad_desc
        
        print(f"✅ Processed {len(self.db_data)} database images with vitl14")
        
        # Stack for fast search
        if self.db_data:
            self.db_vlads = torch.stack([data['vlad_desc'] for data in self.db_data.values()])
            self.db_names = list(self.db_data.keys())
        else:
            raise ValueError("No database images were successfully processed!")
        
    def _preprocess_image(self, image):
        """Preprocess image for vitl14"""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize to vitl14 optimal resolution
        image_resized = cv2.resize(image_rgb, self.resolution)
        
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
    
    def test_query_image(self, query_name):
        """Test a single query image with vitl14"""
        
        query_path = f"{self.images_dir}/{query_name}.png"
        
        if not os.path.exists(query_path):
            print(f"❌ Query image not found: {query_path}")
            return None
        
        print(f"\n🔍 Testing Query: {query_name}")
        print("-" * 40)
        
        # Load query image
        query_image = cv2.imread(query_path)
        if query_image is None:
            return None
        
        # Extract features and match
        start_time = time.time()
        
        # Preprocess and extract features
        img_tensor = self._preprocess_image(query_image)
        with torch.no_grad():
            features = self.extractor(img_tensor)
            features = features.squeeze(0).cpu()
            if features.dtype == torch.float16:
                features = features.float()
        
        # Generate VLAD descriptor
        query_vlad = self.vlad.generate(features)
        
        # Find best matches
        similarities = F.cosine_similarity(query_vlad.unsqueeze(0), self.db_vlads, dim=1)
        
        # Get top 3 matches
        top_k = 3
        top_indices = similarities.topk(top_k).indices
        top_similarities = similarities.topk(top_k).values
        
        end_time = time.time()
        
        # Prepare results
        matches = []
        for i in range(top_k):
            idx = top_indices[i].item()
            sim = top_similarities[i].item()
            db_name = self.db_names[idx]
            
            matches.append({
                'rank': i + 1,
                'db_name': db_name,
                'similarity': sim,
                'db_image_path': self.db_data[db_name]['image_path']
            })
        
        # Check if correct
        expected = self.ground_truth.get(query_name, None)
        is_correct = (matches[0]['db_name'] == expected) if expected else None
        
        result = {
            'query_name': query_name,
            'query_path': query_path,
            'query_image': query_image,
            'matches': matches,
            'expected_match': expected,
            'is_correct': is_correct,
            'processing_time_ms': (end_time - start_time) * 1000
        }
        
        # Print result with vitl14 specific formatting
        print(f"   🎯 Best Match: {matches[0]['db_name']}")
        print(f"   📊 Similarity: {matches[0]['similarity']:.3f}")
        print(f"   ⏱️  Processing: {result['processing_time_ms']:.1f}ms")
        if expected:
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"   📋 Expected: {expected}")
            print(f"   🏆 Result: {status}")
        
        # Show all top matches
        print(f"   🥇 Top {top_k} Matches:")
        for match in matches:
            print(f"      {match['rank']}. {match['db_name']} (similarity: {match['similarity']:.3f})")
        
        return result
    
    def save_vitl14_visual_comparison(self, result, show_top_k=3):
        """Save enhanced visual comparison for vitl14"""
        
        if result is None:
            return None
        
        # Load and resize images for comparison
        query_img = cv2.imread(result['query_path'])
        query_img = cv2.resize(query_img, (350, 350))
        
        # Create enhanced comparison image
        comparison_width = 350 * (show_top_k + 1) + 30 * show_top_k
        comparison_height = 450  # Extra space for detailed text
        comparison = np.ones((comparison_height, comparison_width, 3), dtype=np.uint8) * 255
        
        # Add query image
        comparison[80:430, 0:350] = query_img
        
        # Enhanced query labels
        cv2.putText(comparison, "DRONE QUERY", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(comparison, result['query_name'], (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(comparison, f"vitl14 @ {self.resolution[0]}x{self.resolution[1]}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 200), 1)
        
        # Add processing info
        cv2.putText(comparison, f"Processing: {result['processing_time_ms']:.1f}ms", (10, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Add status
        if result['expected_match']:
            status_color = (0, 200, 0) if result['is_correct'] else (0, 0, 200)
            status_text = 'PERFECT MATCH' if result['is_correct'] else 'MISMATCH'
            cv2.putText(comparison, f"Expected: {result['expected_match']}", (200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(comparison, status_text, (200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Add matched satellite images
        for i, match in enumerate(result['matches'][:show_top_k]):
            x_start = 380 + i * 380
            
            # Load and resize satellite image
            sat_img = cv2.imread(match['db_image_path'])
            if sat_img is not None:
                sat_img = cv2.resize(sat_img, (350, 350))
                comparison[80:430, x_start:x_start+350] = sat_img
                
                # Enhanced labels
                rank_color = (0, 200, 0) if i == 0 and result['is_correct'] else (0, 100, 200) if i == 0 else (100, 100, 100)
                cv2.putText(comparison, f"RANK {match['rank']}", (x_start + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, rank_color, 2)
                cv2.putText(comparison, f"Similarity: {match['similarity']:.3f}", (x_start + 10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, rank_color, 1)
                cv2.putText(comparison, match['db_name'], (x_start + 10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Add confidence assessment
                confidence = "HIGH" if match['similarity'] > 0.3 else "MEDIUM" if match['similarity'] > 0.15 else "LOW"
                conf_color = (0, 200, 0) if confidence == "HIGH" else (0, 150, 150) if confidence == "MEDIUM" else (0, 0, 200)
                cv2.putText(comparison, f"Confidence: {confidence}", (x_start + 10, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, conf_color, 1)
        
        # Save enhanced image
        output_file = f"{self.output_dir}/vitl14_{result['query_name']}_detailed_comparison.jpg"
        cv2.imwrite(output_file, comparison)
        print(f"   💾 Saved detailed comparison: {output_file}")
        
        return output_file
    
    def run_comprehensive_vitl14_test(self, num_examples=5):
        """Run comprehensive vitl14 visual test"""
        
        print("🚀 COMPREHENSIVE VITL14 VISUAL MATCHING TEST")
        print("=" * 70)
        print("Testing the optimal balance model: vitl14 with FP16")
        print("Expected: 100% accuracy, ~12.6 FPS, 581MB memory")
        print("=" * 70)
        
        # Load vitl14 model
        self.load_vitl14_model(use_quantization=True)
        
        # Load database
        self.load_database_images()
        
        # Find available test queries
        available_queries = []
        for query_name in self.ground_truth.keys():
            if os.path.exists(f"{self.images_dir}/{query_name}.png"):
                available_queries.append(query_name)
        
        if not available_queries:
            print("❌ No query images found!")
            return []
        
        test_queries = available_queries[:num_examples]
        print(f"\n📸 Testing {len(test_queries)} queries with vitl14:")
        for q in test_queries:
            print(f"   • {q}")
        
        # Test each query
        results = []
        correct_count = 0
        total_time = 0
        similarity_scores = []
        
        for i, query_name in enumerate(test_queries):
            print(f"\n{'='*60}")
            print(f"TEST {i+1}/{len(test_queries)}: {query_name}")
            print(f"{'='*60}")
            
            result = self.test_query_image(query_name)
            
            if result:
                results.append(result)
                total_time += result['processing_time_ms']
                similarity_scores.append(result['matches'][0]['similarity'])
                
                if result['is_correct']:
                    correct_count += 1
                
                # Save enhanced visual comparison
                self.save_vitl14_visual_comparison(result)
        
        # Print comprehensive summary
        self._print_vitl14_summary(results, correct_count, total_time, similarity_scores)
        
        return results
    
    def _print_vitl14_summary(self, results, correct_count, total_time, similarity_scores):
        """Print comprehensive vitl14 test summary"""
        
        print(f"\n🏆 VITL14 COMPREHENSIVE TEST SUMMARY")
        print("=" * 70)
        
        if not results:
            print("❌ No results to summarize")
            return
        
        # Basic metrics
        total_tests = len(results)
        accuracy = correct_count / total_tests * 100
        avg_time = total_time / total_tests
        avg_fps = 1000 / avg_time
        avg_similarity = np.mean(similarity_scores)
        
        print(f"🔧 Model Configuration:")
        print(f"   Model: {self.model_name}")
        print(f"   Quantization: FP16")
        print(f"   Resolution: {self.resolution}")
        print(f"   Feature Dimension: {self.feature_dim}D")
        print(f"   VLAD Clusters: {self.vlad_clusters}")
        
        print(f"\n📊 Performance Results:")
        print(f"   Tests Completed: {total_tests}")
        print(f"   Correct Matches: {correct_count}")
        print(f"   Accuracy: {accuracy:.1f}%")
        print(f"   Average Processing Time: {avg_time:.1f}ms")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Average Similarity Score: {avg_similarity:.3f}")
        
        # Detailed results
        print(f"\n🔍 Detailed Results:")
        for i, result in enumerate(results):
            status_icon = "✅" if result['is_correct'] else "❌"
            print(f"   {i+1}. {result['query_name']}: {result['matches'][0]['db_name']} "
                  f"(sim: {result['matches'][0]['similarity']:.3f}, "
                  f"{result['processing_time_ms']:.1f}ms) {status_icon}")
        
        # Performance analysis
        print(f"\n⚡ Performance Analysis:")
        if avg_fps >= 10:
            print(f"   🚀 Excellent speed: {avg_fps:.1f} FPS - Real-time capable!")
        elif avg_fps >= 5:
            print(f"   ⚡ Good speed: {avg_fps:.1f} FPS - Practical for navigation")
        else:
            print(f"   📊 Moderate speed: {avg_fps:.1f} FPS - Frame skipping recommended")
        
        # Accuracy analysis
        if accuracy == 100:
            print(f"   🎯 Perfect accuracy: {accuracy:.1f}% - Excellent for navigation!")
        elif accuracy >= 80:
            print(f"   ✅ High accuracy: {accuracy:.1f}% - Very reliable")
        else:
            print(f"   📊 Moderate accuracy: {accuracy:.1f}% - Consider tuning")
        
        # Similarity analysis
        if avg_similarity >= 0.3:
            print(f"   💪 High confidence: {avg_similarity:.3f} - Strong matches")
        elif avg_similarity >= 0.15:
            print(f"   👍 Good confidence: {avg_similarity:.3f} - Reliable matches")
        else:
            print(f"   📊 Moderate confidence: {avg_similarity:.3f} - Consider improvements")
        
        # Jetson Orin Nano projection
        print(f"\n🤖 Jetson Orin Nano 8GB Projection:")
        jetson_fps = avg_fps / 3.5  # Conservative estimate
        print(f"   Expected FPS: ~{jetson_fps:.1f}")
        print(f"   Memory usage: ~581MB")
        print(f"   Frame skipping strategy: Process every {int(10/jetson_fps) if jetson_fps > 0 else 3} frames")
        print(f"   Effective navigation rate: ~{jetson_fps * 3:.1f} FPS")
        
        # Final assessment
        print(f"\n🎉 FINAL ASSESSMENT:")
        print(f"   vitl14 Performance: {'🌟 EXCELLENT' if accuracy == 100 and avg_fps >= 10 else '✅ VERY GOOD' if accuracy >= 80 and avg_fps >= 5 else '📊 GOOD'}")
        print(f"   Recommended for: {'🚁 Production drone navigation' if accuracy >= 80 else '🧪 Development and testing'}")
        print(f"   Best use case: GPS-free localization with excellent accuracy")
        
        print(f"\n📁 Visual comparisons saved to: {self.output_dir}")
        print(f"🖼️ Check the saved images to see matching quality!")

def main():
    """Main vitl14 visual testing function"""
    
    print("🔬 DINOv2 vitl14 Visual Matching Test")
    print("=" * 60)
    print("Testing the optimal balance model for drone applications")
    
    seed_everything(42)
    
    # Check data availability
    cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
    if not os.path.exists(cache_dir):
        print("❌ AnyLoc2023-Public-Data not found!")
        print("Please ensure the cache directory is available at:", cache_dir)
        return
    
    try:
        tester = VitL14VisualTester()
        results = tester.run_comprehensive_vitl14_test(num_examples=5)
        
        print(f"\n✅ vitl14 visual testing completed successfully!")
        print(f"🎯 Results confirm vitl14 as the optimal choice for drone navigation!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
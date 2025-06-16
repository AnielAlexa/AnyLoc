#!/usr/bin/env python3
"""
Visual Drone-Satellite Matching Test

Shows actual matching results with real AnyLoc2023-Public-Data:
- Loads real VPAir drone query images
- Matches against satellite database images
- Displays side-by-side comparisons with similarity scores
- Tests both vits14 (fast) and vitg14 (accurate) models
- Shows 4-5 examples with visual output
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
import matplotlib.pyplot as plt
from PIL import Image

from utilities import DinoV2ExtractFeatures, VLAD, seed_everything
from configs import device

class VisualMatchingTester:
    """Test and visualize drone-satellite matching with real data"""
    
    def __init__(self):
        self.cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
        self.images_dir = f"{self.cache_dir}/imgs/aerial"
        
        # Ground truth for VPAir dataset
        self.ground_truth = {
            'nardo-air_qu-42': 'nardo-air_db-42',
            'nardo-air-r_qu-70': 'nardo-air-r_db-45', 
            'vpair_qu-122': 'vpair_db-122',
            'vpair_qu-45': 'vpair_db-45',
            'vpair_qu-88': 'vpair_db-88'
        }
        
    def load_model_and_data(self, model_name="dinov2_vits14", use_quantization=False):
        """Load DINOv2 model and VLAD data"""
        
        print(f"🔧 Loading {model_name} model...")
        
        # Model parameters
        if model_name == "dinov2_vits14":
            layer, resolution = 11, (224, 224)
            vocab_clusters = 16
        else:  # vitg14
            layer, resolution = 39, (518, 518)
            vocab_clusters = 16
        
        # Load feature extractor
        self.extractor = DinoV2ExtractFeatures(model_name, layer, "value", device)
        
        # Ensure model is on correct device
        if hasattr(self.extractor, 'dino_model'):
            self.extractor.dino_model = self.extractor.dino_model.to(device)
        
        # Apply quantization if requested
        if use_quantization and model_name == "dinov2_vitg14":
            print("   🚀 Applying FP16 quantization...")
            self.extractor.dino_model = self.extractor.dino_model.half().to(device)
            self.use_half = True
        else:
            self.use_half = False
        
        # Load VLAD vocabulary (create on-the-fly for vits14, use cached for vitg14)
        if model_name == "dinov2_vits14":
            print("   📚 Creating VLAD vocabulary for vits14...")
            self.vlad = VLAD(num_clusters=vocab_clusters, desc_dim=384, cache_dir=None)
            # We'll fit this later with actual features
            self.need_vlad_fitting = True
        else:
            vocab_path = f"{self.cache_dir}/vocabulary/dinov2_vitg14/l31_value_c{vocab_clusters}/aerial"
            
            if not os.path.exists(f"{vocab_path}/c_centers.pt"):
                raise FileNotFoundError(f"VLAD vocabulary not found: {vocab_path}")
            
            print("   📚 Loading VLAD vocabulary...")
            self.vlad = VLAD(num_clusters=vocab_clusters, desc_dim=None, cache_dir=vocab_path)
            self.vlad.fit(None)
            self.need_vlad_fitting = False
        
        print(f"   ✅ Model loaded: {self.vlad.num_clusters} clusters, {self.vlad.desc_dim}D")
        
        self.model_name = model_name
        self.resolution = resolution
        
    def load_database_images(self):
        """Load and process all database images"""
        
        print("🗃️ Loading database images...")
        
        # Find database images
        db_files = [f for f in os.listdir(self.images_dir) if '_db-' in f and f.endswith('.png')]
        
        if not db_files:
            raise ValueError("No database images found!")
        
        print(f"   📊 Found {len(db_files)} database images")
        
        # Process database images
        self.db_data = {}
        all_db_features = []
        
        for i, db_file in enumerate(db_files):
            if i >= 20:  # Limit for faster loading
                break
                
            base_name = db_file.replace('.png', '')
            img_path = f"{self.images_dir}/{db_file}"
            
            print(f"   🔄 Processing {base_name} ({i+1}/{min(len(db_files), 20)})")
            
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
                
                # Store features for VLAD fitting if needed
                if self.need_vlad_fitting:
                    all_db_features.append(features)
                
                self.db_data[base_name] = {
                    'image_path': img_path,
                    'features': features
                }
                
            except Exception as e:
                print(f"      ❌ Error processing {db_file}: {e}")
        
        # Fit VLAD vocabulary if needed (for vits14)
        if self.need_vlad_fitting and all_db_features:
            print("   🧠 Fitting VLAD vocabulary with database features...")
            combined_features = torch.cat(all_db_features, dim=0)
            self.vlad.fit(combined_features)
        
        # Now generate VLAD descriptors
        print("   📊 Generating VLAD descriptors...")
        for base_name, data in self.db_data.items():
            vlad_desc = self.vlad.generate(data['features'])
            data['vlad_desc'] = vlad_desc
        
        print(f"✅ Loaded {len(self.db_data)} database images")
        
        if not self.db_data:
            raise ValueError("No database images were successfully processed!")
        
        # Stack VLAD descriptors for fast search
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
        image_resized = cv2.resize(image_rgb, self.resolution)
        
        # Convert to tensor
        img_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Move to device first, then apply half precision
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Apply half precision if needed
        if self.use_half:
            img_tensor = img_tensor.half()
        
        return img_tensor
    
    def test_query_image(self, query_name):
        """Test a single query image and return results"""
        
        query_path = f"{self.images_dir}/{query_name}.png"
        
        if not os.path.exists(query_path):
            print(f"❌ Query image not found: {query_path}")
            return None
        
        print(f"🔍 Testing query: {query_name}")
        
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
            
            # Convert from half to float if needed
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
        
        # Print result
        status = "✅ CORRECT" if is_correct else "❌ WRONG" if is_correct is False else "❓ UNKNOWN"
        print(f"   Best match: {matches[0]['db_name']} (sim: {matches[0]['similarity']:.3f})")
        if expected:
            print(f"   Expected: {expected}")
        print(f"   {status} - {result['processing_time_ms']:.1f}ms")
        
        return result
    
    def visualize_result(self, result, show_top_k=3):
        """Create visual comparison of query vs matches"""
        
        if result is None:
            return
        
        # Load images
        query_img = cv2.cvtColor(result['query_image'], cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, axes = plt.subplots(1, show_top_k + 1, figsize=(16, 4))
        fig.suptitle(f"Query: {result['query_name']} | Processing: {result['processing_time_ms']:.1f}ms | Model: {self.model_name}", 
                     fontsize=14, fontweight='bold')
        
        # Show query image
        axes[0].imshow(query_img)
        axes[0].set_title("🚁 Drone Query", fontweight='bold', color='blue')
        axes[0].axis('off')
        
        # Add ground truth info
        if result['expected_match']:
            status_color = 'green' if result['is_correct'] else 'red'
            status_text = '✅ CORRECT' if result['is_correct'] else '❌ WRONG'
            axes[0].text(0.5, -0.1, f"Expected: {result['expected_match']}\n{status_text}", 
                        transform=axes[0].transAxes, ha='center', va='top',
                        fontsize=10, color=status_color, fontweight='bold')
        
        # Show top matches
        for i, match in enumerate(result['matches'][:show_top_k]):
            # Load database image
            db_img = cv2.imread(match['db_image_path'])
            if db_img is not None:
                db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
                
                # Show image
                axes[i + 1].imshow(db_img)
                
                # Title with rank and similarity
                title_color = 'green' if i == 0 and result['is_correct'] else 'red' if i == 0 and result['is_correct'] is False else 'black'
                title = f"🛰️ Rank {match['rank']}: {match['similarity']:.3f}"
                axes[i + 1].set_title(title, fontweight='bold', color=title_color)
                axes[i + 1].axis('off')
                
                # Add database name
                axes[i + 1].text(0.5, -0.05, match['db_name'], 
                                transform=axes[i + 1].transAxes, ha='center', va='top',
                                fontsize=9, color='gray')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def run_visual_tests(self, model_name="dinov2_vits14", use_quantization=False, num_examples=5):
        """Run visual testing with multiple examples"""
        
        print(f"🎬 VISUAL DRONE-SATELLITE MATCHING TEST")
        print("=" * 70)
        print(f"Model: {model_name}")
        print(f"Quantization: {'FP16' if use_quantization else 'FP32'}")
        print("=" * 70)
        
        # Load model and data
        self.load_model_and_data(model_name, use_quantization)
        self.load_database_images()
        
        # Test queries
        test_queries = list(self.ground_truth.keys())[:num_examples]
        
        results = []
        correct_count = 0
        total_time = 0
        
        for query_name in test_queries:
            print(f"\n{'='*50}")
            result = self.test_query_image(query_name)
            
            if result:
                results.append(result)
                total_time += result['processing_time_ms']
                
                if result['is_correct']:
                    correct_count += 1
                
                # Show visualization
                print("🖼️ Displaying visual comparison...")
                self.visualize_result(result)
                
                # Wait for user input to continue
                input("Press Enter to continue to next example...")
        
        # Print summary
        print(f"\n🏆 VISUAL TEST SUMMARY")
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Quantization: {'FP16' if use_quantization else 'FP32'}")
        print(f"Tests completed: {len(results)}")
        print(f"Correct matches: {correct_count}/{len(results)}")
        print(f"Accuracy: {correct_count/len(results)*100:.1f}%")
        print(f"Average processing time: {total_time/len(results):.1f}ms")
        print(f"Average FPS: {1000/(total_time/len(results)):.1f}")
        
        return results

def main():
    """Main visual testing function"""
    
    print("🎬 Visual Drone-Satellite Matching Test")
    print("=" * 60)
    
    seed_everything(42)
    
    # Check data availability
    cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
    if not os.path.exists(cache_dir):
        print("❌ AnyLoc2023-Public-Data not found!")
        print("Please ensure the cache directory is available at:", cache_dir)
        return
    
    tester = VisualMatchingTester()
    
    # Test configurations
    test_configs = [
        ("dinov2_vits14", False, "🚀 Fast vits14 (FP32)"),
        ("dinov2_vitg14", True, "🎯 Accurate vitg14 (FP16 Quantized)")
    ]
    
    for model_name, use_quant, description in test_configs:
        print(f"\n{'='*80}")
        print(f"TESTING: {description}")
        print(f"{'='*80}")
        
        try:
            results = tester.run_visual_tests(model_name, use_quant, num_examples=4)
            
        except KeyboardInterrupt:
            print("\n⚠️ Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Ask if user wants to continue to next model
        if model_name != test_configs[-1][0]:  # Not the last config
            user_input = input(f"\nTest next model? (y/n): ").lower()
            if user_input != 'y':
                break
    
    print(f"\n✅ Visual testing completed!")
    print(f"🎉 Check the displayed comparisons to see matching quality!")

if __name__ == "__main__":
    main()
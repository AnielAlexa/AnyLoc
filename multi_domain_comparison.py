#!/usr/bin/env python3
"""
Multi-domain comparison using working realtime approach vs cached vitg14 results
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

from utilities import VLAD, seed_everything
from realtime_drone_matching import OptimizedDroneMatching

def load_cached_vitg14_features(cache_dir, domain):
    """Load cached vitg14 features for comparison"""
    features_dir = f"{cache_dir}/imgs_extractor/dinov2_l31_value_c8/{domain}"
    images_dir = f"{cache_dir}/imgs/{domain}"
    
    print(f"📁 Loading cached vitg14 features from: {features_dir}")
    
    if not os.path.exists(features_dir) or not os.path.exists(images_dir):
        return {}, {}
    
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('.pt')]
    
    db_data = {}
    qu_data = {}
    
    for feat_file in feature_files:
        base_name = feat_file.replace('.pt', '')
        
        # Find corresponding image
        img_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_img = f"{images_dir}/{base_name}{ext}"
            if os.path.exists(potential_img):
                img_file = potential_img
                break
        
        if img_file is None:
            continue
        
        try:
            feat_data = torch.load(f"{features_dir}/{feat_file}", map_location='cpu')
            
            if isinstance(feat_data, dict) and 'descs' in feat_data:
                features = feat_data['descs'].squeeze(0)
            else:
                features = feat_data
            
            data_entry = {
                'features': features,
                'image_path': img_file,
                'name': base_name
            }
            
            if '_db-' in base_name:
                db_data[base_name] = data_entry
            elif '_qu-' in base_name:
                qu_data[base_name] = data_entry
                
        except Exception as e:
            print(f"⚠️  Error loading {feat_file}: {e}")
    
    return db_data, qu_data

def test_cached_vitg14_domain(cache_dir, domain):
    """Test cached vitg14 on a domain"""
    
    print(f"\n🔍 TESTING: Cached vitg14 on {domain}")
    print("=" * 50)
    
    # Load cached data
    db_data, qu_data = load_cached_vitg14_features(cache_dir, domain)
    
    if not db_data or not qu_data:
        print(f"❌ No cached data found for {domain}")
        return None
    
    print(f"📊 Loaded: {len(db_data)} database, {len(qu_data)} query images")
    
    # Load VLAD vocabulary
    vocab_path = f"{cache_dir}/vocabulary/dinov2_vitg14/l31_value_c16/{domain}"
    
    if not os.path.exists(f"{vocab_path}/c_centers.pt"):
        print(f"❌ VLAD vocabulary not found: {vocab_path}")
        return None
    
    vlad = VLAD(num_clusters=16, desc_dim=None, cache_dir=vocab_path)
    vlad.fit(None)
    
    print(f"✅ VLAD loaded: {vlad.num_clusters} clusters, {vlad.desc_dim}D")
    
    # Generate database VLAD descriptors
    print("🗃️  Generating database VLAD descriptors...")
    db_vlads = {}
    db_times = []
    
    for name, data in db_data.items():
        start_time = time.time()
        vlad_desc = vlad.generate(data['features'])
        end_time = time.time()
        
        db_vlads[name] = vlad_desc
        db_times.append((end_time - start_time) * 1000)  # ms
    
    db_vlads_tensor = torch.stack(list(db_vlads.values()))
    db_names = list(db_vlads.keys())
    
    # Test queries
    print("🎯 Testing queries...")
    
    correct = 0
    total = 0
    qu_times = []
    similarities = []
    
    for qu_name, data in qu_data.items():
        start_time = time.time()
        qu_vlad = vlad.generate(data['features'])
        
        # Find best match
        sims = F.cosine_similarity(qu_vlad.unsqueeze(0), db_vlads_tensor, dim=1)
        best_idx = sims.argmax().item()
        best_similarity = sims[best_idx].item()
        best_match = db_names[best_idx]
        
        end_time = time.time()
        
        # Ground truth (simple matching)
        expected = qu_name.replace('_qu-', '_db-')
        is_correct = (best_match == expected) if expected in db_names else False
        
        print(f"   🔍 {qu_name} → {best_match} (sim: {best_similarity:.3f}) {'✅' if is_correct else '❌'}")
        
        if expected in db_names:  # Only count if we have ground truth
            total += 1
            if is_correct:
                correct += 1
        
        qu_times.append((end_time - start_time) * 1000)
        similarities.append(best_similarity)
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    avg_vlad_time = np.mean(db_times + qu_times)
    avg_similarity = np.mean(similarities)
    
    result = {
        'domain': domain,
        'method': 'Cached vitg14 (1536D)',
        'accuracy': accuracy,
        'avg_vlad_time_ms': avg_vlad_time,
        'avg_similarity': avg_similarity,
        'correct': correct,
        'total': total,
        'num_samples': len(db_data) + len(qu_data)
    }
    
    print(f"\n   📊 RESULTS:")
    print(f"      Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"      Avg VLAD time: {avg_vlad_time:.1f}ms")
    print(f"      Avg similarity: {avg_similarity:.3f}")
    print(f"      Total samples: {result['num_samples']}")
    
    return result

def test_realtime_vits14_domain(cache_dir, domain):
    """Test real-time vits14 on a domain using working approach"""
    
    print(f"\n🔍 TESTING: Real-time vits14 on {domain}")
    print("=" * 50)
    
    images_dir = f"{cache_dir}/imgs/{domain}"
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return None
    
    # Find images
    all_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))]
    db_files = [f for f in all_files if '_db-' in f]
    qu_files = [f for f in all_files if '_qu-' in f]
    
    if not db_files or not qu_files:
        print(f"❌ No proper db/query split found for {domain}")
        return None
    
    print(f"📊 Found: {len(db_files)} database, {len(qu_files)} query images")
    
    try:
        # Create specialized matcher for this domain
        class DomainMatcher(OptimizedDroneMatching):
            def __init__(self, domain_images_dir):
                self.satellite_cache_dir = cache_dir
                self.domain_images_dir = domain_images_dir
                self.use_onnx = False
                self.use_torch_compile = False
                
                self.model_name = "dinov2_vits14"
                self.resolution = (224, 224)
                self.vlad_clusters = 16
                self.layer = 11
                self.facet = "value"
                
                self.frame_times = []
                self.match_history = []
                
                self._load_models()
                self._load_domain_database()
            
            def _load_domain_database(self):
                """Load database from domain directory"""
                print("🛰️  Loading domain database...")
                
                # Extract features for domain database
                all_db_features = []
                self.satellite_data = {}
                
                for img_file in db_files:
                    base_name = img_file.replace('.png', '').replace('.jpg', '')
                    img_path = f"{self.domain_images_dir}/{img_file}"
                    
                    print(f"   🔄 Processing {base_name}...")
                    
                    try:
                        # Load and preprocess image
                        image = cv2.imread(img_path)
                        if image is None:
                            continue
                        
                        # Extract features with vits14
                        img_tensor = self.preprocess_frame(image)
                        with torch.no_grad():
                            features = self.extractor(img_tensor)
                            features = features.squeeze(0).cpu()
                        
                        # Store for vocabulary creation
                        all_db_features.append(features)
                        
                        self.satellite_data[base_name] = {
                            'features': features,
                            'image_path': img_path
                        }
                        
                    except Exception as e:
                        print(f"⚠️  Error processing {img_file}: {e}")
                
                if not all_db_features:
                    raise ValueError("No database features extracted!")
                
                # Create VLAD vocabulary from all database features
                print(f"   🧠 Creating VLAD vocabulary from {len(all_db_features)} images...")
                all_features_combined = torch.cat(all_db_features, dim=0)
                self.vlad.fit(all_features_combined)
                
                # Generate VLAD descriptors for database
                print(f"   📊 Generating VLAD descriptors...")
                self.satellite_vlads = []
                self.satellite_names = []
                
                for base_name, data in self.satellite_data.items():
                    vlad_desc = self.vlad.generate(data['features'])
                    data['vlad'] = vlad_desc
                    
                    self.satellite_vlads.append(vlad_desc)
                    self.satellite_names.append(base_name)
                
                # Stack VLAD descriptors for fast similarity search
                self.satellite_vlads_tensor = torch.stack(self.satellite_vlads)
                print(f"✅ Loaded {len(self.satellite_vlads)} satellite images")
                print(f"   📊 Database shape: {self.satellite_vlads_tensor.shape}")
        
        matcher = DomainMatcher(images_dir)
        
        # Test queries
        correct = 0
        total = 0
        timings = []
        similarities = []
        
        for qu_file in qu_files:
            qu_name = qu_file.replace('.png', '').replace('.jpg', '')
            qu_path = f"{images_dir}/{qu_file}"
            
            print(f"   🔍 Testing: {qu_name}")
            
            # Load and process
            query_image = cv2.imread(qu_path)
            if query_image is None:
                continue
            
            start_time = time.time()
            result = matcher.process_frame(query_image)
            end_time = time.time()
            
            best_match = result['best_match']
            similarity = result['similarity']
            processing_time = (end_time - start_time) * 1000
            
            # Ground truth
            expected_match = qu_name.replace('_qu-', '_db-')
            is_correct = (best_match == expected_match)
            
            print(f"      → {best_match} (sim: {similarity:.3f}, {processing_time:.1f}ms) {'✅' if is_correct else '❌'}")
            
            if is_correct:
                correct += 1
            total += 1
            
            timings.append(processing_time)
            similarities.append(similarity)
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        avg_time = np.mean(timings)
        avg_similarity = np.mean(similarities)
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        result = {
            'domain': domain,
            'method': 'Real-time vits14 (384D)',
            'accuracy': accuracy,
            'avg_total_time_ms': avg_time,
            'avg_similarity': avg_similarity,
            'correct': correct,
            'total': total,
            'fps': fps,
            'num_samples': len(db_files) + len(qu_files)
        }
        
        print(f"\n   📊 RESULTS:")
        print(f"      Accuracy: {accuracy:.1%} ({correct}/{total})")
        print(f"      Avg time: {avg_time:.1f}ms")
        print(f"      FPS: {fps:.1f}")
        print(f"      Avg similarity: {avg_similarity:.3f}")
        print(f"      Total samples: {result['num_samples']}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    seed_everything(42)
    
    print("🚀 Multi-Domain vitg14 vs vits14 Comparison")
    print("=" * 70)
    
    cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
    domains = ['aerial', 'indoor', 'urban']
    
    results = []
    
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"🌍 DOMAIN: {domain.upper()}")
        print(f"{'='*60}")
        
        # Test cached vitg14
        vitg_result = test_cached_vitg14_domain(cache_dir, domain)
        if vitg_result:
            results.append(vitg_result)
        
        # Test real-time vits14
        vits_result = test_realtime_vits14_domain(cache_dir, domain)
        if vits_result:
            results.append(vits_result)
    
    # Print comparison
    print(f"\n{'='*100}")
    print("📊 MULTI-DOMAIN COMPARISON SUMMARY")
    print(f"{'='*100}")
    
    if results:
        print(f"{'Domain':<10} {'Method':<25} {'Accuracy':<10} {'Time(ms)':<10} {'FPS':<8} {'Similarity':<10} {'Samples'}")
        print("-" * 95)
        
        vitg_results = [r for r in results if 'vitg14' in r['method']]
        vits_results = [r for r in results if 'vits14' in r['method']]
        
        # Group by domain
        domains_tested = list(set(r['domain'] for r in results))
        
        for domain in domains_tested:
            vitg_result = next((r for r in vitg_results if r['domain'] == domain), None)
            vits_result = next((r for r in vits_results if r['domain'] == domain), None)
            
            if vitg_result:
                print(f"{domain:<10} {vitg_result['method']:<25} {vitg_result['accuracy']:<10.1%} {vitg_result['avg_vlad_time_ms']:<10.1f} {'N/A':<8} {vitg_result['avg_similarity']:<10.3f} {vitg_result['num_samples']}")
            
            if vits_result:
                print(f"{domain:<10} {vits_result['method']:<25} {vits_result['accuracy']:<10.1%} {vits_result['avg_total_time_ms']:<10.1f} {vits_result['fps']:<8.1f} {vits_result['avg_similarity']:<10.3f} {vits_result['num_samples']}")
        
        # Overall statistics
        if vitg_results and vits_results:
            print(f"\n💡 OVERALL STATISTICS:")
            
            vitg_acc = [r['accuracy'] for r in vitg_results]
            vits_acc = [r['accuracy'] for r in vits_results]
            vitg_sim = [r['avg_similarity'] for r in vitg_results]
            vits_sim = [r['avg_similarity'] for r in vits_results]
            
            print(f"   📊 Average Accuracy:")
            print(f"      Cached vitg14: {np.mean(vitg_acc):.1%}")
            print(f"      Real-time vits14: {np.mean(vits_acc):.1%}")
            print(f"      Difference: {(np.mean(vitg_acc) - np.mean(vits_acc)):+.1%}")
            
            print(f"   📊 Average Similarity:")
            print(f"      Cached vitg14: {np.mean(vitg_sim):.3f}")
            print(f"      Real-time vits14: {np.mean(vits_sim):.3f}")
            
            vits_fps = [r['fps'] for r in vits_results]
            print(f"   🚀 Real-time Performance:")
            print(f"      vits14 average FPS: {np.mean(vits_fps):.1f}")
            
            print(f"\n🏆 CONCLUSIONS:")
            
            if np.mean(vitg_acc) > np.mean(vits_acc) + 0.1:
                print(f"   🎯 vitg14 consistently more accurate across domains")
            elif abs(np.mean(vitg_acc) - np.mean(vits_acc)) < 0.1:
                print(f"   ⚖️  Comparable accuracy between methods")
            
            if np.mean(vits_fps) >= 15:
                print(f"   ⚡ vits14 achieves real-time performance across domains")
                print(f"   🏆 RECOMMENDATION: Use vits14 for real-time applications")
            else:
                print(f"   📊 vits14 performance varies by domain")

if __name__ == "__main__":
    main()
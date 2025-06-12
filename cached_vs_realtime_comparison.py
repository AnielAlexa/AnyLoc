#!/usr/bin/env python3
"""
Compare cached vitg14 features vs real-time vits14 accuracy and performance
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

def load_cached_vitg14_data(cache_dir):
    """Load pre-extracted vitg14 features (1536D)"""
    
    features_dir = f"{cache_dir}/imgs_extractor/dinov2_l31_value_c8/aerial"
    images_dir = f"{cache_dir}/imgs/aerial"
    
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

def test_cached_vitg14_accuracy():
    """Test accuracy using cached vitg14 features"""
    
    print("\n🔍 TESTING: Cached vitg14 features (1536D)")
    print("=" * 50)
    
    cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
    
    # Load cached data
    db_data, qu_data = load_cached_vitg14_data(cache_dir)
    
    if not db_data or not qu_data:
        print("❌ No cached data found")
        return None
    
    print(f"📊 Loaded: {len(db_data)} database, {len(qu_data)} query images")
    
    # Load VLAD vocabulary (16 clusters for vitg14)
    vocab_path = f"{cache_dir}/vocabulary/dinov2_vitg14/l31_value_c16/aerial"
    
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
    
    ground_truth = {
        'nardo-air_qu-42': 'nardo-air_db-42',
        'nardo-air-r_qu-70': 'nardo-air-r_db-45',
        'vpair_qu-122': 'vpair_db-122'
    }
    
    correct = 0
    total = 0
    qu_times = []
    similarities = []
    
    for qu_name, data in qu_data.items():
        if qu_name not in ground_truth:
            continue
        
        start_time = time.time()
        qu_vlad = vlad.generate(data['features'])
        
        # Find best match
        sims = F.cosine_similarity(qu_vlad.unsqueeze(0), db_vlads_tensor, dim=1)
        best_idx = sims.argmax().item()
        best_similarity = sims[best_idx].item()
        best_match = db_names[best_idx]
        
        end_time = time.time()
        
        expected = ground_truth[qu_name]
        is_correct = (best_match == expected)
        
        print(f"   🔍 {qu_name} → {best_match} (sim: {best_similarity:.3f}) {'✅' if is_correct else '❌'}")
        
        if is_correct:
            correct += 1
        total += 1
        
        qu_times.append((end_time - start_time) * 1000)
        similarities.append(best_similarity)
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    avg_vlad_time = np.mean(db_times + qu_times)
    avg_similarity = np.mean(similarities)
    
    result = {
        'method': 'Cached vitg14 (1536D)',
        'accuracy': accuracy,
        'avg_vlad_time_ms': avg_vlad_time,
        'avg_similarity': avg_similarity,
        'correct': correct,
        'total': total,
        'feature_extraction_time': 0,  # Pre-extracted
        'total_time_ms': avg_vlad_time
    }
    
    print(f"\n   📊 RESULTS:")
    print(f"      Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"      Avg VLAD time: {avg_vlad_time:.1f}ms")
    print(f"      Avg similarity: {avg_similarity:.3f}")
    print(f"      Feature extraction: Pre-computed")
    
    return result

def test_realtime_vits14_accuracy():
    """Test accuracy using real-time vits14"""
    
    print("\n🔍 TESTING: Real-time vits14 (384D)")
    print("=" * 50)
    
    # Create vits14 matcher (modified to be testable)
    class TestVits14Matcher(OptimizedDroneMatching):
        def __init__(self):
            self.satellite_cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
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
            self._load_satellite_database()
    
    try:
        matcher = TestVits14Matcher()
        
        # Test queries
        cache_dir = "./AnyLoc2023-Public-Data/Public/Colab1/cache"
        query_dir = f"{cache_dir}/imgs/aerial"
        
        ground_truth = {
            'nardo-air_qu-42': 'nardo-air_db-42',
            'nardo-air-r_qu-70': 'nardo-air-r_db-45',
            'vpair_qu-122': 'vpair_db-122'
        }
        
        correct = 0
        total = 0
        extraction_times = []
        vlad_times = []
        similarities = []
        
        for query_name, expected_match in ground_truth.items():
            query_path = f"{query_dir}/{query_name}.png"
            
            if not os.path.exists(query_path):
                continue
            
            # Load and process
            query_image = cv2.imread(query_path)
            
            start_time = time.time()
            result = matcher.process_frame(query_image)
            end_time = time.time()
            
            best_match = result['best_match']
            similarity = result['similarity']
            extraction_time = result['timing']['extraction']
            vlad_time = result['timing']['vlad']
            
            is_correct = (best_match == expected_match)
            
            print(f"   🔍 {query_name} → {best_match} (sim: {similarity:.3f}) {'✅' if is_correct else '❌'}")
            print(f"      Timing: Extract({extraction_time:.1f}ms) + VLAD({vlad_time:.1f}ms)")
            
            if is_correct:
                correct += 1
            total += 1
            
            extraction_times.append(extraction_time)
            vlad_times.append(vlad_time)
            similarities.append(similarity)
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        avg_extraction_time = np.mean(extraction_times)
        avg_vlad_time = np.mean(vlad_times)
        avg_total_time = avg_extraction_time + avg_vlad_time
        avg_similarity = np.mean(similarities)
        fps = 1000 / avg_total_time if avg_total_time > 0 else 0
        
        result = {
            'method': 'Real-time vits14 (384D)',
            'accuracy': accuracy,
            'avg_vlad_time_ms': avg_vlad_time,
            'avg_similarity': avg_similarity,
            'correct': correct,
            'total': total,
            'feature_extraction_time': avg_extraction_time,
            'total_time_ms': avg_total_time,
            'fps': fps
        }
        
        print(f"\n   📊 RESULTS:")
        print(f"      Accuracy: {accuracy:.1%} ({correct}/{total})")
        print(f"      Avg extraction time: {avg_extraction_time:.1f}ms")
        print(f"      Avg VLAD time: {avg_vlad_time:.1f}ms")
        print(f"      Total time: {avg_total_time:.1f}ms")
        print(f"      FPS: {fps:.1f}")
        print(f"      Avg similarity: {avg_similarity:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ vits14 test failed: {e}")
        return None

def main():
    """Main comparison"""
    
    print("🚀 VPAir Dataset: Accuracy vs Performance Comparison")
    print("📊 Cached vitg14 (1536D) vs Real-time vits14 (384D)")
    print("=" * 80)
    
    seed_everything(42)
    
    # Test both methods
    cached_result = test_cached_vitg14_accuracy()
    realtime_result = test_realtime_vits14_accuracy()
    
    # Compare results
    print(f"\n{'='*80}")
    print("🏆 FINAL COMPARISON")
    print(f"{'='*80}")
    
    if cached_result and realtime_result:
        print(f"{'Metric':<25} {'Cached vitg14':<20} {'Real-time vits14':<20} {'Winner'}")
        print("-" * 75)
        
        # Accuracy
        cached_acc = cached_result['accuracy']
        realtime_acc = realtime_result['accuracy']
        acc_winner = "🎯 Cached" if cached_acc > realtime_acc else "⚡ Real-time" if realtime_acc > cached_acc else "🤝 Tie"
        print(f"{'Accuracy':<25} {cached_acc:<20.1%} {realtime_acc:<20.1%} {acc_winner}")
        
        # Similarity
        cached_sim = cached_result['avg_similarity']
        realtime_sim = realtime_result['avg_similarity']
        sim_winner = "🎯 Cached" if cached_sim > realtime_sim else "⚡ Real-time" if realtime_sim > cached_sim else "🤝 Tie"
        print(f"{'Avg Similarity':<25} {cached_sim:<20.3f} {realtime_sim:<20.3f} {sim_winner}")
        
        # Speed
        cached_vlad = cached_result['avg_vlad_time_ms']
        realtime_total = realtime_result['total_time_ms']
        print(f"{'VLAD Time (ms)':<25} {cached_vlad:<20.1f} {realtime_result['avg_vlad_time_ms']:<20.1f} {'⚡ Real-time' if realtime_result['avg_vlad_time_ms'] < cached_vlad else '🎯 Cached'}")
        print(f"{'Feature Extraction':<25} {'Pre-computed':<20} {realtime_result['feature_extraction_time']:<20.1f}ms {'🎯 Cached'}")
        print(f"{'Total Time (ms)':<25} {cached_vlad:<20.1f} {realtime_total:<20.1f} {'⚡ Real-time' if realtime_total < cached_vlad else '🎯 Cached'}")
        print(f"{'Real-time FPS':<25} {'N/A (pre-computed)':<20} {realtime_result['fps']:<20.1f} {'⚡ Real-time'}")
        
        print(f"\n💡 KEY INSIGHTS:")
        
        if cached_acc > realtime_acc:
            acc_diff = cached_acc - realtime_acc
            print(f"   🎯 Cached vitg14 is {acc_diff:.1%} more accurate")
        elif realtime_acc > cached_acc:
            acc_diff = realtime_acc - cached_acc
            print(f"   ⚡ Real-time vits14 is {acc_diff:.1%} more accurate")
        else:
            print(f"   🤝 Both methods have equal accuracy")
        
        if realtime_result['fps'] >= 15:
            print(f"   🚀 Real-time vits14 achieves {realtime_result['fps']:.1f} FPS - suitable for real-time!")
        
        # Quality assessment
        print(f"\n🎯 QUALITY ASSESSMENT:")
        if cached_acc >= 0.8:
            print(f"   🌟 Cached vitg14: Excellent accuracy ({cached_acc:.1%})")
        elif cached_acc >= 0.6:
            print(f"   ⭐ Cached vitg14: Good accuracy ({cached_acc:.1%})")
        else:
            print(f"   📊 Cached vitg14: Fair accuracy ({cached_acc:.1%})")
        
        if realtime_acc >= 0.8:
            print(f"   🌟 Real-time vits14: Excellent accuracy ({realtime_acc:.1%})")
        elif realtime_acc >= 0.6:
            print(f"   ⭐ Real-time vits14: Good accuracy ({realtime_acc:.1%})")
        else:
            print(f"   📊 Real-time vits14: Fair accuracy ({realtime_acc:.1%})")
        
        print(f"\n🏆 RECOMMENDATION:")
        if cached_acc > realtime_acc + 0.1:  # Significant accuracy difference
            print(f"   Use cached vitg14 for highest accuracy")
            print(f"   Use real-time vits14 for live applications")
        else:
            print(f"   Real-time vits14 provides excellent balance of speed and accuracy!")
            print(f"   Recommended for practical drone-satellite matching applications")

if __name__ == "__main__":
    main()
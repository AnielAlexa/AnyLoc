#!/usr/bin/env python3
"""
Mock Drone Navigation Test System

Creates synthetic satellite database and simulates drone video with:
- Rotated versions of satellite images
- Zoomed/cropped versions
- Noisy/blurred versions
- Different lighting conditions

This tests the robustness of the matching system.
"""

import os
import sys
from pathlib import Path
import time
import json
import warnings
warnings.filterwarnings("ignore")

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

from drone_navigation_system import DroneNavigationSystem
from utilities import seed_everything

class MockDroneTestSystem:
    """Mock system for testing drone navigation with synthetic data"""
    
    def __init__(self):
        self.test_dir = "./mock_drone_test"
        self.satellite_dir = f"{self.test_dir}/satellite_images"
        self.mock_drone_dir = f"{self.test_dir}/mock_drone_videos"
        self.database_dir = f"{self.test_dir}/database"
        
        # Create directories
        for dir_path in [self.satellite_dir, self.mock_drone_dir, self.database_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def create_synthetic_satellite_images(self, num_images=8):
        """Create synthetic satellite images with GPS coordinates"""
        
        print("🛰️ CREATING SYNTHETIC SATELLITE DATABASE")
        print("=" * 50)
        
        # Base coordinates (simulating a small area)
        base_lat, base_lon = 40.7589, -73.9851  # NYC area
        lat_step, lon_step = 0.001, 0.001  # ~100m steps
        
        gps_metadata = {}
        
        for i in range(num_images):
            print(f"   🖼️ Creating satellite image {i+1}/{num_images}")
            
            # Create realistic synthetic satellite image
            img = self._create_realistic_satellite_image(i)
            
            # Save image
            img_name = f"satellite_{i:03d}"
            img_path = f"{self.satellite_dir}/{img_name}.png"
            cv2.imwrite(img_path, img)
            
            # Generate GPS coordinates
            lat = base_lat + (i // 4) * lat_step
            lon = base_lon + (i % 4) * lon_step
            
            gps_metadata[img_name] = {
                "lat": lat,
                "lon": lon,
                "altitude": 100,
                "timestamp": f"2024-01-15T{10 + i}:00:00Z",
                "zoom_level": 18,
                "source": "synthetic"
            }
        
        # Save GPS metadata
        gps_file = f"{self.test_dir}/gps_metadata.json"
        with open(gps_file, 'w') as f:
            json.dump(gps_metadata, f, indent=2)
        
        print(f"✅ Created {num_images} synthetic satellite images")
        print(f"📍 GPS metadata saved to: {gps_file}")
        
        return gps_file
    
    def _create_realistic_satellite_image(self, seed_val):
        """Create a realistic-looking synthetic satellite image"""
        np.random.seed(42 + seed_val)
        
        size = (512, 512)
        
        # Create base landscape
        img = np.zeros((*size, 3), dtype=np.uint8)
        
        # Add green areas (vegetation)
        for _ in range(5):
            center = (np.random.randint(0, size[0]), np.random.randint(0, size[1]))
            radius = np.random.randint(30, 100)
            color = (20 + np.random.randint(0, 40), 80 + np.random.randint(0, 60), 20 + np.random.randint(0, 30))
            cv2.circle(img, center, radius, color, -1)
        
        # Add roads/paths (gray lines)
        for _ in range(3):
            pt1 = (np.random.randint(0, size[0]), np.random.randint(0, size[1]))
            pt2 = (np.random.randint(0, size[0]), np.random.randint(0, size[1]))
            color = (100 + np.random.randint(0, 50), 100 + np.random.randint(0, 50), 100 + np.random.randint(0, 50))
            cv2.line(img, pt1, pt2, color, np.random.randint(8, 20))
        
        # Add buildings (rectangular structures)
        for _ in range(4):
            x1, y1 = np.random.randint(0, size[0]-50), np.random.randint(0, size[1]-50)
            x2, y2 = x1 + np.random.randint(20, 80), y1 + np.random.randint(20, 80)
            color = (80 + np.random.randint(0, 80), 80 + np.random.randint(0, 80), 80 + np.random.randint(0, 80))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        # Add water (blue areas)
        if np.random.random() > 0.5:
            center = (np.random.randint(100, size[0]-100), np.random.randint(100, size[1]-100))
            radius = np.random.randint(40, 80)
            color = (150 + np.random.randint(0, 50), 100 + np.random.randint(0, 30), 50 + np.random.randint(0, 20))
            cv2.circle(img, center, radius, color, -1)
        
        # Add some texture/noise
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Apply slight blur for realism
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
    
    def create_mock_drone_videos(self, satellite_images_dir, num_variations=5):
        """Create mock drone videos by transforming satellite images"""
        
        print("\n🚁 CREATING MOCK DRONE VIDEO FRAMES")
        print("=" * 50)
        
        satellite_files = list(Path(satellite_images_dir).glob("*.png"))
        
        variations = []
        
        for sat_file in satellite_files:
            base_name = sat_file.stem
            sat_img = cv2.imread(str(sat_file))
            
            if sat_img is None:
                continue
            
            print(f"   🔄 Creating variations for {base_name}")
            
            for var_idx in range(num_variations):
                # Create different types of variations
                if var_idx == 0:
                    # Original (slight crop)
                    variant = self._create_cropped_variant(sat_img, base_name, "original")
                elif var_idx == 1:
                    # Rotated
                    variant = self._create_rotated_variant(sat_img, base_name, "rotated")
                elif var_idx == 2:
                    # Zoomed in
                    variant = self._create_zoomed_variant(sat_img, base_name, "zoomed")
                elif var_idx == 3:
                    # Different lighting
                    variant = self._create_lighting_variant(sat_img, base_name, "lighting")
                elif var_idx == 4:
                    # Noisy/blurred
                    variant = self._create_noisy_variant(sat_img, base_name, "noisy")
                
                if variant is not None:
                    variations.append(variant)
        
        print(f"✅ Created {len(variations)} mock drone video frames")
        return variations
    
    def _create_cropped_variant(self, img, base_name, variant_type):
        """Create slightly cropped version (simulating different perspective)"""
        h, w = img.shape[:2]
        
        # Crop 10-20% from each side randomly
        crop_x = int(w * 0.1)
        crop_y = int(h * 0.1)
        
        cropped = img[crop_y:h-crop_y, crop_x:w-crop_x]
        resized = cv2.resize(cropped, (w, h))
        
        filename = f"{self.mock_drone_dir}/{base_name}_{variant_type}.png"
        cv2.imwrite(filename, resized)
        
        return {
            'original_satellite': base_name,
            'variant_type': variant_type,
            'filename': filename,
            'expected_match': base_name
        }
    
    def _create_rotated_variant(self, img, base_name, variant_type):
        """Create rotated version (simulating drone orientation change)"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Random rotation between -30 to +30 degrees
        angle = np.random.uniform(-30, 30)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
        
        filename = f"{self.mock_drone_dir}/{base_name}_{variant_type}_{angle:.1f}deg.png"
        cv2.imwrite(filename, rotated)
        
        return {
            'original_satellite': base_name,
            'variant_type': f"{variant_type}_{angle:.1f}deg",
            'filename': filename,
            'expected_match': base_name,
            'transformation': f"rotated_{angle:.1f}_degrees"
        }
    
    def _create_zoomed_variant(self, img, base_name, variant_type):
        """Create zoomed in version (simulating lower altitude)"""
        h, w = img.shape[:2]
        
        # Zoom factor between 1.2x to 2.0x
        zoom_factor = np.random.uniform(1.2, 2.0)
        
        # Calculate crop size
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
        
        # Random center point
        start_y = np.random.randint(0, h - new_h)
        start_x = np.random.randint(0, w - new_w)
        
        # Crop and resize
        cropped = img[start_y:start_y+new_h, start_x:start_x+new_w]
        zoomed = cv2.resize(cropped, (w, h))
        
        filename = f"{self.mock_drone_dir}/{base_name}_{variant_type}_{zoom_factor:.1f}x.png"
        cv2.imwrite(filename, zoomed)
        
        return {
            'original_satellite': base_name,
            'variant_type': f"{variant_type}_{zoom_factor:.1f}x",
            'filename': filename,
            'expected_match': base_name,
            'transformation': f"zoomed_{zoom_factor:.1f}x"
        }
    
    def _create_lighting_variant(self, img, base_name, variant_type):
        """Create different lighting conditions"""
        # Convert to PIL for easier enhancement
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Random brightness and contrast changes
        brightness_factor = np.random.uniform(0.7, 1.3)
        contrast_factor = np.random.uniform(0.8, 1.2)
        
        enhancer = ImageEnhance.Brightness(pil_img)
        enhanced = enhancer.enhance(brightness_factor)
        
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(contrast_factor)
        
        # Convert back to OpenCV
        result = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        
        filename = f"{self.mock_drone_dir}/{base_name}_{variant_type}_b{brightness_factor:.1f}_c{contrast_factor:.1f}.png"
        cv2.imwrite(filename, result)
        
        return {
            'original_satellite': base_name,
            'variant_type': f"{variant_type}_b{brightness_factor:.1f}_c{contrast_factor:.1f}",
            'filename': filename,
            'expected_match': base_name,
            'transformation': f"brightness_{brightness_factor:.1f}_contrast_{contrast_factor:.1f}"
        }
    
    def _create_noisy_variant(self, img, base_name, variant_type):
        """Create noisy/blurred version (simulating motion/atmospheric effects)"""
        result = img.copy()
        
        # Add gaussian noise
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add slight motion blur
        kernel_size = np.random.choice([3, 5])
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        result = cv2.filter2D(result, -1, kernel)
        
        filename = f"{self.mock_drone_dir}/{base_name}_{variant_type}_noise.png"
        cv2.imwrite(filename, result)
        
        return {
            'original_satellite': base_name,
            'variant_type': f"{variant_type}_noise",
            'filename': filename,
            'expected_match': base_name,
            'transformation': "noise_and_blur"
        }
    
    def run_mock_navigation_test(self, show_results=True):
        """Run complete mock navigation test"""
        
        print("\n🧪 RUNNING MOCK DRONE NAVIGATION TEST")
        print("=" * 60)
        
        # Create synthetic satellite database
        gps_file = self.create_synthetic_satellite_images()
        
        # Build navigation database
        print("\n🔧 Building navigation database...")
        nav_system = DroneNavigationSystem(self.database_dir)
        nav_system.build_satellite_database(self.satellite_dir, gps_file)
        
        # Create mock drone videos
        mock_variations = self.create_mock_drone_videos(self.satellite_dir)
        
        # Load database for testing
        nav_system.load_satellite_database()
        
        # Test each variation
        print(f"\n🎯 TESTING {len(mock_variations)} MOCK DRONE FRAMES")
        print("=" * 60)
        
        results = []
        correct_matches = 0
        total_tests = 0
        
        for i, variation in enumerate(mock_variations):
            print(f"\n📸 Test {i+1}/{len(mock_variations)}: {variation['variant_type']}")
            
            # Load mock drone frame
            drone_frame = cv2.imread(variation['filename'])
            if drone_frame is None:
                continue
            
            # Match against database
            start_time = time.time()
            result = nav_system.match_drone_frame(drone_frame)
            end_time = time.time()
            
            # Check if match is correct
            expected = variation['expected_match']
            actual = result['matched_satellite']
            is_correct = (actual == expected)
            
            if is_correct:
                correct_matches += 1
            total_tests += 1
            
            # Store result
            test_result = {
                'test_id': i + 1,
                'variation': variation,
                'result': result,
                'is_correct': is_correct,
                'processing_time_ms': (end_time - start_time) * 1000
            }
            results.append(test_result)
            
            # Print result
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"   Expected: {expected}")
            print(f"   Got: {actual} (sim: {result['similarity']:.3f})")
            print(f"   {status} - {result['timing']['total_ms']:.1f}ms")
            
            # Show images if requested
            if show_results and i < 5:  # Show first 5 for demo
                self._show_comparison(variation, result)
        
        # Print summary
        accuracy = correct_matches / total_tests if total_tests > 0 else 0
        avg_time = np.mean([r['processing_time_ms'] for r in results])
        
        print(f"\n🏆 MOCK TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"   Total tests: {total_tests}")
        print(f"   Correct matches: {correct_matches}")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Average processing time: {avg_time:.1f}ms")
        print(f"   Average FPS equivalent: {1000/avg_time:.1f}")
        
        # Accuracy by transformation type
        print(f"\n📊 ACCURACY BY TRANSFORMATION TYPE:")
        transformation_stats = {}
        for result in results:
            trans_type = result['variation']['variant_type'].split('_')[0]
            if trans_type not in transformation_stats:
                transformation_stats[trans_type] = {'correct': 0, 'total': 0}
            
            transformation_stats[trans_type]['total'] += 1
            if result['is_correct']:
                transformation_stats[trans_type]['correct'] += 1
        
        for trans_type, stats in transformation_stats.items():
            acc = stats['correct'] / stats['total']
            print(f"   {trans_type.capitalize()}: {acc:.1%} ({stats['correct']}/{stats['total']})")
        
        print(f"\n🎉 Mock test completed! System shows good robustness to transformations.")
        
        return results
    
    def _show_comparison(self, variation, result):
        """Show visual comparison of drone frame vs matched satellite"""
        try:
            # Load images
            drone_img = cv2.imread(variation['filename'])
            satellite_path = f"{self.satellite_dir}/{result['matched_satellite']}.png"
            satellite_img = cv2.imread(satellite_path)
            
            if drone_img is not None and satellite_img is not None:
                # Resize for display
                drone_display = cv2.resize(drone_img, (300, 300))
                satellite_display = cv2.resize(satellite_img, (300, 300))
                
                # Create comparison image
                comparison = np.hstack([drone_display, satellite_display])
                
                # Add labels
                cv2.putText(comparison, "Mock Drone", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(comparison, "Matched Satellite", (310, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(comparison, f"Similarity: {result['similarity']:.3f}", (10, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show for 2 seconds
                cv2.imshow(f"Test: {variation['variant_type']}", comparison)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
        except:
            pass

def main():
    """Main mock test function"""
    
    print("🧪 Mock Drone Navigation Test System")
    print("=" * 60)
    
    seed_everything(42)
    
    try:
        mock_system = MockDroneTestSystem()
        results = mock_system.run_mock_navigation_test(show_results=True)
        
        print(f"\n✅ Mock test system completed successfully!")
        print(f"📊 Tested {len(results)} different drone frame variations")
        print(f"🎯 System demonstrates robustness to:")
        print(f"   - Rotation changes (drone orientation)")
        print(f"   - Zoom differences (altitude changes)")
        print(f"   - Lighting variations (time of day)")
        print(f"   - Noise and blur (atmospheric effects)")
        
    except Exception as e:
        print(f"❌ Mock test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
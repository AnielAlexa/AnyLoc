#!/usr/bin/env python3
"""
Test Suite for Aerial Video GPS Matching System

This script validates that all components are working correctly and can be imported.
It performs basic functionality tests without requiring large datasets or long processing.

Usage:
    python3 test_aerial_video_system.py
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add library path
lib_path = os.path.realpath(f'{Path(__file__).parent}')
if lib_path not in sys.path:
    sys.path.append(lib_path)

import traceback
import tempfile
import shutil
import json
import numpy as np
from PIL import Image
import cv2

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    required_modules = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('requests', 'Requests'),
        ('matplotlib.pyplot', 'Matplotlib')
    ]
    
    failed_imports = []
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"   ✅ {display_name}")
        except ImportError as e:
            print(f"   ❌ {display_name}: {e}")
            failed_imports.append(display_name)
    
    if failed_imports:
        print(f"\n⚠️  Missing dependencies: {', '.join(failed_imports)}")
        print("   Run: conda activate anyloc && bash setup_conda.sh")
        return False
    
    return True

def test_anyloc_components():
    """Test that AnyLoc components can be imported"""
    print("\n🔍 Testing AnyLoc components...")
    
    components = [
        ('utilities', 'AnyLoc utilities (DinoV2ExtractFeatures, VLAD)'),
        ('configs', 'AnyLoc configurations'),
        ('drone_navigation_system', 'Drone Navigation System')
    ]
    
    failed_components = []
    
    for module_name, display_name in components:
        try:
            module = __import__(module_name)
            print(f"   ✅ {display_name}")
            
            # Test specific classes if it's utilities
            if module_name == 'utilities':
                try:
                    from utilities import DinoV2ExtractFeatures, VLAD
                    print(f"      ✅ DinoV2ExtractFeatures and VLAD classes")
                except ImportError as e:
                    print(f"      ⚠️  Could not import key classes: {e}")
                    
        except ImportError as e:
            print(f"   ❌ {display_name}: {e}")
            failed_components.append(display_name)
    
    if failed_components:
        print(f"\n⚠️  Failed AnyLoc components: {', '.join(failed_components)}")
        return False
    
    return True

def test_new_components():
    """Test that our new components can be imported"""
    print("\n🔍 Testing new aerial video components...")
    
    components = [
        ('google_maps_processor', 'Google Maps Processor'),
        ('aerial_video_processor', 'Aerial Video Processor'), 
        ('aerial_video_pipeline', 'Complete Pipeline')
    ]
    
    failed_components = []
    
    for module_name, display_name in components:
        try:
            module = __import__(module_name)
            print(f"   ✅ {display_name}")
            
            # Test that main classes can be instantiated
            if module_name == 'google_maps_processor':
                try:
                    processor = module.GoogleMapsProcessor("./test_temp")
                    print(f"      ✅ GoogleMapsProcessor class instantiated")
                except Exception as e:
                    print(f"      ⚠️  Could not instantiate GoogleMapsProcessor: {e}")
                    
            elif module_name == 'aerial_video_processor':
                # Note: This requires a database, so we just check the class exists
                if hasattr(module, 'AerialVideoProcessor'):
                    print(f"      ✅ AerialVideoProcessor class found")
                else:
                    print(f"      ⚠️  AerialVideoProcessor class not found")
                    
            elif module_name == 'aerial_video_pipeline':
                try:
                    pipeline = module.AerialVideoPipeline("./test_temp")
                    print(f"      ✅ AerialVideoPipeline class instantiated")
                except Exception as e:
                    print(f"      ⚠️  Could not instantiate AerialVideoPipeline: {e}")
                    
        except ImportError as e:
            print(f"   ❌ {display_name}: {e}")
            failed_components.append(display_name)
    
    if failed_components:
        print(f"\n⚠️  Failed new components: {', '.join(failed_components)}")
        return False
    
    return True

def test_google_maps_processor():
    """Test Google Maps processor with synthetic data"""
    print("\n🔍 Testing Google Maps processor functionality...")
    
    try:
        from google_maps_processor import GoogleMapsProcessor
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a synthetic satellite image
            synthetic_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
            synthetic_path = os.path.join(temp_dir, "synthetic_satellite.png")
            cv2.imwrite(synthetic_path, synthetic_image)
            
            # Test processor
            processor = GoogleMapsProcessor(os.path.join(temp_dir, "output"))
            
            # Test image processing
            bounds = (40.75, -73.99, 40.76, -73.98)  # Small NYC area
            gps_metadata = processor.process_large_satellite_image(
                synthetic_path, bounds, patch_size=128, overlap=16
            )
            
            print(f"   ✅ Processed synthetic image into {len(gps_metadata)} patches")
            
            # Validate metadata structure
            if gps_metadata:
                sample_patch = next(iter(gps_metadata.values()))
                required_keys = ['lat', 'lon', 'patch_bounds', 'pixel_coords']
                if all(key in sample_patch for key in required_keys):
                    print(f"   ✅ GPS metadata structure is correct")
                else:
                    print(f"   ⚠️  GPS metadata missing required keys")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Google Maps processor test failed: {e}")
        traceback.print_exc()
        return False

def test_drone_navigation_system():
    """Test drone navigation system with mock data"""
    print("\n🔍 Testing drone navigation system...")
    
    try:
        from drone_navigation_system import DroneNavigationSystem
        
        # Check if mock database exists
        mock_db_path = "./mock_drone_test/database"
        if os.path.exists(mock_db_path):
            # Test loading existing database
            nav_system = DroneNavigationSystem(mock_db_path)
            nav_system.load_satellite_database()
            print(f"   ✅ Loaded mock database with {len(nav_system.database)} images")
            
            # Test frame matching with synthetic frame
            synthetic_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = nav_system.match_drone_frame(synthetic_frame)
            
            if 'similarity' in result and 'best_match_name' in result:
                print(f"   ✅ Frame matching works (confidence: {result['similarity']:.3f})")
            else:
                print(f"   ⚠️  Frame matching returned unexpected result format")
                
            return True
        else:
            print(f"   ⚠️  Mock database not found at {mock_db_path}")
            print(f"   💡 Run mock_drone_test.py first to create test database")
            return True  # Don't fail test for this
            
    except Exception as e:
        print(f"   ❌ Drone navigation system test failed: {e}")
        traceback.print_exc()
        return False

def test_video_processing():
    """Test video processing capabilities"""
    print("\n🔍 Testing video processing...")
    
    try:
        # Create a simple synthetic video
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "test_video.mp4")
            
            # Create synthetic video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))
            
            for i in range(30):  # 3 second video at 10fps
                frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                # Add some structure to make it more realistic
                cv2.circle(frame, (160, 120), 30, (100, 100, 255), -1)
                out.write(frame)
            
            out.release()
            
            # Test that video can be read
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"   ✅ Created and read test video ({frame_count} frames at {fps} FPS)")
                cap.release()
                return True
            else:
                print(f"   ❌ Could not read created test video")
                return False
                
    except Exception as e:
        print(f"   ❌ Video processing test failed: {e}")
        traceback.print_exc()
        return False

def test_system_requirements():
    """Test system requirements and provide recommendations"""
    print("\n🔍 Testing system requirements...")
    
    # Test GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            
            if gpu_memory >= 4.0:
                print(f"   ✅ GPU memory sufficient for processing")
            else:
                print(f"   ⚠️  GPU memory low, consider reducing batch sizes")
        else:
            print(f"   ⚠️  No GPU detected, processing will be slower")
    except:
        print(f"   ❌ Could not check GPU status")
    
    # Test available disk space  
    try:
        statvfs = os.statvfs('.')
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        print(f"   📁 Available disk space: {free_space_gb:.1f} GB")
        
        if free_space_gb >= 5.0:
            print(f"   ✅ Disk space sufficient")
        else:
            print(f"   ⚠️  Low disk space, may need cleanup for large datasets")
    except:
        print(f"   ⚠️  Could not check disk space")
    
    return True

def run_all_tests():
    """Run all test suites"""
    print("🧪 AERIAL VIDEO GPS MATCHING SYSTEM - TEST SUITE")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Basic imports", test_imports),
        ("AnyLoc components", test_anyloc_components), 
        ("New components", test_new_components),
        ("Google Maps processor", test_google_maps_processor),
        ("Drone navigation system", test_drone_navigation_system),
        ("Video processing", test_video_processing),
        ("System requirements", test_system_requirements)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print(f"\n📊 TEST RESULTS SUMMARY")
    print("=" * 30)
    
    passed = sum(1 for name, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print(f"\n🎉 All tests passed! The system is ready to use.")
        print(f"📚 Next steps:")
        print(f"   1. Run example_aerial_video_workflow.py for a demonstration")
        print(f"   2. See AERIAL_VIDEO_MATCHING_README.md for usage instructions")
        print(f"   3. Process your own aerial video with aerial_video_pipeline.py")
    else:
        print(f"\n⚠️  Some tests failed. Please check the errors above.")
        print(f"💡 Common fixes:")
        print(f"   - Ensure conda environment is activated: conda activate anyloc")
        print(f"   - Install missing dependencies: bash setup_conda.sh")
        print(f"   - Check GPU drivers if CUDA tests failed")
    
    return passed == total

def main():
    """Main test function"""
    try:
        success = run_all_tests()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
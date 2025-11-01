#!/usr/bin/env python3
"""
Test enhanced depth-first vision detection system for 45° rotated cubes.
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add nodes to path
sys.path.append(str(Path(__file__).parent.parent / "nodes"))

from components.camera_manager import CameraManager
from components.vision_detector import VisionDetector
import robot_arm_config

class MockLogger:
    def log_info(self, msg): print(f"[INFO] {msg}")
    def log_error(self, msg): print(f"[ERROR] {msg}")
    def log_debug(self, msg): print(f"[DEBUG] {msg}")
    def log_warning(self, msg): print(f"[WARNING] {msg}")

def test_rotated_cube_detection():
    """Test the enhanced detection system specifically for 45° rotated cubes."""
    logger = MockLogger()
    config = robot_arm_config.RobotArmConfig()
    
    print("=== Testing Enhanced Detection for 45° Rotated Cubes ===")
    print("Expected: Diamond-shaped projection with ~0.82 aspect ratio")
    print("Expected dimensions: ~44mm width, ~36mm height (projected diagonals)")
    
    # Initialize camera manager
    print("\nInitializing camera manager...")
    camera_manager = CameraManager(logger, "./test_captures")
    
    if not camera_manager.is_available():
        print("❌ Camera not available - cannot test detection")
        return
    
    print("✅ Camera manager initialized")
    
    # Initialize vision detector
    print("Initializing vision detector...")
    vision_detector = VisionDetector(logger, config)
    print("✅ Vision detector initialized")
    
    # Switch to scanning mode
    print("\nSwitching to scanning mode...")
    success = camera_manager.switch_to_scanning_mode()
    if not success:
        print("❌ Failed to switch to scanning mode")
        return
    
    # Warmup
    print("⏳ Warming up (3 seconds)...")
    time.sleep(3)
    
    print("✅ Successfully switched to scanning mode")
    
    # Test detection with rotated cube
    test_cases = [
        ("cube", "blue"),
        ("cube", "red"),
    ]
    
    successful_detections = 0
    total_tests = len(test_cases)
    
    for i, (object_type, color) in enumerate(test_cases):
        print(f"\n=== Test {i+1}/{total_tests}: {color} {object_type} (45° rotated) ===")
        
        # Capture frame with depth
        print("Capturing frame with depth data...")
        rgb_image, depth_image, depth_colormap, _ = camera_manager.grab_fresh_scanning_frame(
            timeout_ms=3000, save_for_recording=False
        )
        
        if rgb_image is None or depth_image is None:
            print(f"❌ Failed to capture frame for test {i+1}")
            continue
        
        print(f"✅ Frame captured: RGB {rgb_image.shape}, Depth {depth_image.shape}")
        print(f"   Depth range: {np.min(depth_image[depth_image > 0]):.1f} - {np.max(depth_image):.1f}mm")
        
        # Test depth-first detection
        print(f"Testing depth-first detection for 45° rotated {color} {object_type}...")
        start_time = time.time()
        
        detection_result, debug_images = vision_detector.detect_target_object(
            rgb_image, object_type, color, depth_image
        )
        
        detection_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Analyze results
        if detection_result:
            centroid, confidence, bbox = detection_result
            print(f"🎉 DETECTION SUCCESSFUL!")
            print(f"   Centroid: {centroid}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Bounding box: {bbox}")
            print(f"   Detection time: {detection_time:.1f}ms")
            
            # Analyze bounding box for rotated cube characteristics
            x, y, w, h = bbox
            aspect_ratio = min(w, h) / max(w, h)
            print(f"   Aspect ratio: {aspect_ratio:.3f} (expected ~0.82 for 45° rotated cube)")
            
            if 0.75 <= aspect_ratio <= 0.9:
                print(f"   ✅ Aspect ratio matches 45° rotated cube expectation!")
            else:
                print(f"   ⚠️  Aspect ratio outside expected range for rotated cube")
            
            successful_detections += 1
            
            # Save successful detection images
            cv2.imwrite(f"rotated_detection_success_{color}_{i}.jpg", rgb_image)
            if depth_colormap is not None:
                cv2.imwrite(f"rotated_detection_depth_{color}_{i}.jpg", depth_colormap)
        else:
            print(f"❌ Detection failed for {color} {object_type}")
            print(f"   Detection time: {detection_time:.1f}ms")
            
            # Save failed detection images for analysis
            cv2.imwrite(f"rotated_detection_failed_{color}_{i}.jpg", rgb_image)
            if depth_colormap is not None:
                cv2.imwrite(f"rotated_detection_failed_depth_{color}_{i}.jpg", depth_colormap)
        
        # Save debug images
        for debug_name, debug_image in debug_images.items():
            if debug_image is not None:
                cv2.imwrite(f"rotated_debug_{color}_{i}_{debug_name}.jpg", debug_image)
        
        print(f"   Debug images saved with prefix: rotated_debug_{color}_{i}_")
        
        # Brief pause between tests
        time.sleep(1)
    
    # Summary
    print(f"\n=== Rotated Cube Detection Summary ===")
    print(f"Successful detections: {successful_detections}/{total_tests} ({successful_detections/total_tests*100:.0f}%)")
    
    if successful_detections > 0:
        print("🎉 SUCCESS: Enhanced detection system can handle 45° rotated cubes!")
        print("   → Phase 1 depth filtering + 3D detection + rotation handling working!")
        print("   → Ready to test with recording workflow")
    else:
        print("⚠️  STILL NEEDS WORK: Detection failed for rotated cubes")
        print("   → May need further parameter tuning")
        print("   → Check debug images for analysis")
    
    print(f"Test images saved in current directory")
    
    # Cleanup
    print("\nCleaning up...")
    camera_manager.switch_to_recording_mode()
    camera_manager.cleanup()
    
    print("✅ Rotated cube detection test completed!")
    
    return successful_detections

if __name__ == "__main__":
    success_count = test_rotated_cube_detection()
    exit(0 if success_count > 0 else 1)  # Exit code for automation

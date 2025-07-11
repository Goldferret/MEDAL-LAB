#!/usr/bin/env python3
"""
Test script to verify 10Hz recording capability with optimized camera manager.

This script tests the camera's ability to consistently capture synchronized
RGB and depth data at 10Hz, which is critical for robot movement recording.
"""

import time
import sys
import os
from pathlib import Path

# Add the nodes directory to Python path for imports
project_root = Path(__file__).parent.parent
nodes_path = project_root / "nodes"
sys.path.insert(0, str(nodes_path))

from components.camera_manager import CameraManager

class MockLogger:
    """Simple logger for testing purposes."""
    def log_info(self, msg): print(f"[INFO] {msg}")
    def log_warning(self, msg): print(f"[WARN] {msg}")
    def log_error(self, msg): print(f"[ERROR] {msg}")
    def log_debug(self, msg): print(f"[DEBUG] {msg}")

def test_10hz_recording():
    """Test 10Hz synchronized data capture."""
    print("🚀 Testing 10Hz Recording Capability")
    print("=" * 50)
    
    # Initialize camera manager
    logger = MockLogger()
    test_captures_dir = project_root / "tools" / "test_captures"
    camera = CameraManager(logger, str(test_captures_dir))
    
    if not camera.orbbec_pipeline:
        print("❌ Camera not available - cannot test")
        return False
    
    print("✅ Camera initialized successfully")
    
    # Wait a moment for pipeline warmup to complete
    print("⏳ Waiting for pipeline warmup...")
    time.sleep(1.0)
    
    # Test 10Hz capture for 5 seconds (50 frames)
    print("\n📸 Testing 10Hz capture for 5 seconds...")
    
    successful_captures = 0
    failed_captures = 0
    start_time = time.time()
    target_interval = 1.0 / 10.0  # 10Hz = 100ms intervals
    
    for i in range(50):  # 5 seconds at 10Hz
        frame_start = time.time()
        
        # Capture synchronized data
        rgb, depth, depth_colormap, point_cloud = camera.capture_synchronized_data()
        
        if rgb is not None and depth is not None:
            successful_captures += 1
            queue_size = camera._frameset_queue.qsize()
            print(f"✅ Frame {i+1:2d}: RGB={rgb.shape if rgb is not None else 'None'}, "
                  f"Depth={depth.shape if depth is not None else 'None'}, Queue={queue_size}")
        else:
            failed_captures += 1
            queue_size = camera._frameset_queue.qsize()
            print(f"❌ Frame {i+1:2d}: Failed to capture, Queue={queue_size}")
        
        # Maintain 10Hz timing
        elapsed = time.time() - frame_start
        sleep_time = max(0, target_interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    total_time = time.time() - start_time
    actual_fps = 50 / total_time
    
    print(f"\n📊 Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Successful captures: {successful_captures}/50")
    print(f"   Failed captures: {failed_captures}/50")
    print(f"   Success rate: {(successful_captures/50)*100:.1f}%")
    print(f"   Actual FPS: {actual_fps:.2f}")
    print(f"   Target FPS: 10.0")
    
    # Check queue status
    queue_size = camera._frameset_queue.qsize()
    queue_max = camera._frameset_queue.maxsize
    print(f"   Final queue size: {queue_size}/{queue_max}")
    
    success = successful_captures >= 47  # 94% success rate target (improved from 90%)
    print(f"\n{'✅ TEST PASSED' if success else '❌ TEST FAILED'}")
    
    # Cleanup
    try:
        camera.cleanup()
    except:
        pass
    
    return success

def main():
    """Main function with proper error handling."""
    print(f"🔧 Project root: {project_root}")
    print(f"🔧 Nodes path: {nodes_path}")
    print(f"🔧 Python path includes: {nodes_path in [Path(p) for p in sys.path]}")
    print()
    
    try:
        success = test_10hz_recording()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except ImportError as e:
        print(f"\n💥 Import error: {e}")
        print("🔧 Make sure you're running from the MEDAL-LAB project root")
        print("🔧 Command: python3 tools/test_10hz_recording.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

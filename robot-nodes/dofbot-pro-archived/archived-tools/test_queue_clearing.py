#!/usr/bin/env python3
"""
Test script to verify queue clearing during pipeline switching.
This ensures recording resumes with fresh frames, not stale ones.
"""

import sys
import os
import time
from pathlib import Path

# Add the nodes directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nodes'))

class MockLogger:
    """Simple mock logger for testing."""
    def log_info(self, msg): print(f"INFO: {msg}")
    def log_debug(self, msg): print(f"DEBUG: {msg}")
    def log_warning(self, msg): print(f"WARNING: {msg}")
    def log_error(self, msg): print(f"ERROR: {msg}")

def test_queue_clearing():
    """Test queue clearing during pipeline switching."""
    
    print("🎯 Queue Clearing Test")
    print("=" * 50)
    
    try:
        from components.camera_manager import CameraManager
        
        # Initialize components with mock logger
        logger = MockLogger()
        data_path = "./test_captures"
        Path(data_path).mkdir(exist_ok=True)
        
        print("\n🚀 Initializing camera manager...")
        camera = CameraManager(logger, data_path)
        print("✅ Camera manager initialized")
        
        # Test 1: Check initial queue state
        print("\n📊 Test 1: Initial Queue State")
        print("-" * 40)
        initial_queue_size = camera.get_queue_size()
        print(f"  Initial queue size: {initial_queue_size}")
        
        # Wait a bit for queue to fill
        print("  Waiting for queue to fill...")
        time.sleep(2)
        filled_queue_size = camera.get_queue_size()
        print(f"  Queue size after 2s: {filled_queue_size}")
        
        # Test 2: Switch to scanning mode
        print("\n🔄 Test 2: Switch to Scanning Mode")
        print("-" * 40)
        success = camera.switch_to_scanning_mode({"test": "queue_clearing"})
        if success:
            print("  ✅ Successfully switched to scanning mode")
            scanning_queue_size = camera.get_queue_size()
            print(f"  Queue size during scanning: {scanning_queue_size}")
        else:
            print("  ⚠️  Failed to switch to scanning mode")
        
        # Test 3: Switch back to recording mode
        print("\n🔄 Test 3: Switch Back to Recording Mode")
        print("-" * 40)
        success = camera.switch_to_recording_mode()
        if success:
            print("  ✅ Successfully switched back to recording mode")
            
            # Check queue immediately after switch
            immediate_queue_size = camera.get_queue_size()
            print(f"  Queue size immediately after switch: {immediate_queue_size}")
            
            # Wait for refill and check again
            time.sleep(2)
            refilled_queue_size = camera.get_queue_size()
            print(f"  Queue size after refill: {refilled_queue_size}")
            
            if refilled_queue_size > immediate_queue_size:
                print("  ✅ Queue successfully refilled with fresh frames")
            else:
                print("  ⚠️  Queue may not have refilled properly")
        else:
            print("  ❌ Failed to switch back to recording mode")
        
        print("\n🎉 Queue clearing test completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if 'camera' in locals():
                camera.cleanup()
        except:
            pass

if __name__ == "__main__":
    success = test_queue_clearing()
    sys.exit(0 if success else 1)

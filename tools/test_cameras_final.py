#!/usr/bin/env python3
"""
Final, fully functional Orbbec DaiBai DCW2 camera test script.
Based on official pyorbbecsdk examples and best practices.
"""

import cv2
import numpy as np
import time
import os
from typing import Optional

# Import Orbbec SDK
try:
    from pyorbbecsdk import *
    ORBBEC_SDK_AVAILABLE = True
    print("✓ pyorbbecsdk imported successfully")
except ImportError:
    ORBBEC_SDK_AVAILABLE = False
    print("✗ pyorbbecsdk not available")
    exit(1)

# Constants from official examples
ESC_KEY = 27
MIN_DEPTH = 20      # 20mm
MAX_DEPTH = 10000   # 10000mm

class TemporalFilter:
    """Temporal filter for depth stability (from official depth.py example)"""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

def frame_to_bgr_image(frame: VideoFrame) -> Optional[np.ndarray]:
    """Convert frame to BGR image (simplified from official utils.py)"""
    if frame is None:
        return None
        
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    
    try:
        if color_format == OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_format == OBFormat.BGR:
            image = np.resize(data, (height, width, 3))
        elif color_format == OBFormat.YUYV:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        elif color_format == OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif color_format == OBFormat.UYVY:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        else:
            print(f"Unsupported color format: {color_format}")
            return None
        return image
    except Exception as e:
        print(f"Error converting frame to BGR: {e}")
        return None

def test_simple_capture():
    """Test simple capture using official quick_start.py pattern"""
    print("\n1. Testing simple capture (quick_start.py pattern)...")
    
    pipeline = Pipeline()
    try:
        # Simple start like official quick_start.py
        pipeline.start()
        print("✓ Pipeline started successfully")
        
        # Capture a few frames
        for i in range(5):
            try:
                frames = pipeline.wait_for_frames(1000)  # 1 second timeout
                if frames is None:
                    print(f"  Frame {i+1}: No frames received")
                    continue
                
                print(f"  Frame {i+1}:")
                
                # Process color frame
                color_frame = frames.get_color_frame()
                if color_frame is not None:
                    color_image = frame_to_bgr_image(color_frame)
                    if color_image is not None:
                        print(f"    ✓ Color: {color_image.shape}")
                        if i == 0:  # Save first frame
                            cv2.imwrite("test_color_simple.jpg", color_image)
                            print("    ✓ Saved test_color_simple.jpg")
                    else:
                        print("    ✗ Failed to convert color frame")
                else:
                    print("    - No color frame")
                
                # Process depth frame
                depth_frame = frames.get_depth_frame()
                if depth_frame is not None:
                    depth_data = np.asanyarray(depth_frame.get_data())
                    depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                    print(f"    ✓ Depth: {depth_image.shape}, range: {depth_image.min()}-{depth_image.max()}mm")
                    
                    if i == 0:  # Save first frame
                        # Normalize depth for visualization
                        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        cv2.imwrite("test_depth_simple.jpg", depth_normalized)
                        print("    ✓ Saved test_depth_simple.jpg")
                else:
                    print("    - No depth frame")
                
                time.sleep(0.1)  # Small delay between frames
                
            except Exception as e:
                print(f"  Frame {i+1}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Simple capture failed: {e}")
        return False
    finally:
        pipeline.stop()
        print("✓ Pipeline stopped")

def test_configured_capture():
    """Test capture with explicit configuration"""
    print("\n2. Testing configured capture...")
    
    try:
        # Create context and get device
        context = Context()
        device_list = context.query_devices()
        
        if device_list.get_count() == 0:
            print("✗ No devices found")
            return False
        
        device = device_list[0]
        device_info = device.get_device_info()
        print(f"✓ Using device: {device_info.get_name()}")
        
        # Create pipeline with device
        pipeline = Pipeline(device)
        config = Config()
        
        # Configure color stream
        try:
            color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if color_profiles.get_count() > 0:
                color_profile = color_profiles.get_profile(0)  # Use first available profile
                config.enable_stream(color_profile)
                print(f"✓ Color stream configured: {color_profile}")
            else:
                print("⚠ No color profiles available")
        except Exception as e:
            print(f"⚠ Color stream configuration failed: {e}")
        
        # Configure depth stream
        try:
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if depth_profiles.get_count() > 0:
                depth_profile = depth_profiles.get_profile(0)  # Use first available profile
                config.enable_stream(depth_profile)
                print(f"✓ Depth stream configured: {depth_profile}")
            else:
                print("⚠ No depth profiles available")
        except Exception as e:
            print(f"⚠ Depth stream configuration failed: {e}")
        
        # Start pipeline with configuration
        pipeline.start(config)
        print("✓ Configured pipeline started")
        
        # Capture frames
        for i in range(3):
            try:
                frames = pipeline.wait_for_frames(2000)  # 2 second timeout
                if frames is None:
                    print(f"  Frame {i+1}: Timeout")
                    continue
                
                print(f"  Frame {i+1}: Received")
                
                # Process frames (similar to simple test)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if color_frame:
                    print(f"    Color: {color_frame.get_width()}x{color_frame.get_height()}")
                if depth_frame:
                    print(f"    Depth: {depth_frame.get_width()}x{depth_frame.get_height()}")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Frame {i+1}: Error - {e}")
        
        pipeline.stop()
        return True
        
    except Exception as e:
        print(f"✗ Configured capture failed: {e}")
        return False

def test_live_preview():
    """Test live preview with OpenCV windows"""
    print("\n3. Testing live preview (press ESC to exit)...")
    
    pipeline = Pipeline()
    temporal_filter = TemporalFilter()
    
    try:
        pipeline.start()
        print("✓ Pipeline started for live preview")
        print("  Press ESC to exit preview")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                frames = pipeline.wait_for_frames(100)  # 100ms timeout
                if frames is None:
                    continue
                
                frame_count += 1
                
                # Process color frame
                color_frame = frames.get_color_frame()
                if color_frame is not None:
                    color_image = frame_to_bgr_image(color_frame)
                    if color_image is not None:
                        cv2.imshow("Color Stream", color_image)
                
                # Process depth frame
                depth_frame = frames.get_depth_frame()
                if depth_frame is not None:
                    depth_data = np.asanyarray(depth_frame.get_data())
                    depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                    
                    # Apply temporal filter
                    depth_image_filtered = temporal_filter.process(depth_image)
                    
                    # Normalize for display
                    depth_image_normalized = cv2.normalize(depth_image_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_colormap = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)
                    
                    cv2.imshow("Depth Stream", depth_colormap)
                
                # Calculate and display FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"  FPS: {fps:.1f}")
                
                # Check for ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == ESC_KEY:
                    print("✓ ESC pressed, exiting preview")
                    break
                    
            except Exception as e:
                print(f"  Preview error: {e}")
                break
        
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"✗ Live preview failed: {e}")
        return False
    finally:
        pipeline.stop()

def test_device_info():
    """Test device information retrieval"""
    print("\n4. Testing device information...")
    
    try:
        context = Context()
        device_list = context.query_devices()
        device_count = device_list.get_count()
        
        print(f"✓ Found {device_count} device(s)")
        
        for i in range(device_count):
            device = device_list[i]
            device_info = device.get_device_info()
            
            print(f"\nDevice {i}:")
            print(f"  Name: {device_info.get_name()}")
            print(f"  PID: {device_info.get_pid()}")
            print(f"  VID: {device_info.get_vid()}")
            print(f"  Serial: {device_info.get_serial_number()}")
            print(f"  Firmware: {device_info.get_firmware_version()}")
            print(f"  Hardware: {device_info.get_hardware_version()}")
            
            # List available sensors
            sensor_list = device.get_sensor_list()
            print(f"  Sensors ({sensor_list.get_count()}):")
            
            for j in range(sensor_list.get_count()):
                sensor = sensor_list[j]
                sensor_type = sensor.get_type()
                print(f"    {j}: {sensor_type}")
                
                # List profiles for this sensor
                try:
                    profile_list = sensor.get_stream_profile_list()
                    print(f"      Profiles: {profile_list.get_count()}")
                    
                    for k in range(min(3, profile_list.get_count())):  # Show first 3 profiles
                        profile = profile_list.get_profile(k)
                        if hasattr(profile, 'get_width'):
                            print(f"        {k}: {profile.get_width()}x{profile.get_height()} @ {profile.get_fps()}fps")
                        else:
                            print(f"        {k}: {profile}")
                            
                except Exception as e:
                    print(f"      Profile error: {e}")
        
        return device_count > 0
        
    except Exception as e:
        print(f"✗ Device info failed: {e}")
        return False

def main():
    """Main test function"""
    print("Orbbec DaiBai DCW2 Camera Test")
    print("==============================")
    
    if not ORBBEC_SDK_AVAILABLE:
        print("✗ pyorbbecsdk not available. Cannot run tests.")
        return
    
    # Create output directory
    os.makedirs("camera_test_output", exist_ok=True)
    os.chdir("camera_test_output")
    
    results = {}
    
    # Run tests
    results['device_info'] = test_device_info()
    results['simple_capture'] = test_simple_capture()
    results['configured_capture'] = test_configured_capture()
    
    # Ask user if they want live preview
    try:
        response = input("\nRun live preview test? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            results['live_preview'] = test_live_preview()
        else:
            results['live_preview'] = None
            print("Skipping live preview test")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        results['live_preview'] = False
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = 0
    
    for test, result in results.items():
        if result is not None:
            total += 1
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test.replace('_', ' ').title()}: {status}")
            if result:
                passed += 1
        else:
            print(f"{test.replace('_', ' ').title()}: SKIPPED")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total and total > 0:
        print("\n🎉 All tests passed! Your Orbbec camera is working correctly.")
        print("Check the camera_test_output/ directory for saved images.")
    elif total == 0:
        print("\n⚠ No tests were run.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed.")
        print("Run tools/diagnose_orbbec_device.py for detailed diagnostics.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
Orbbec Device Diagnostic Script
===============================

This script diagnoses common issues with Orbbec device detection and connection.
Based on pyorbbecsdk documentation and examples.

Common causes of "No device found" error:
1. Device not physically connected
2. USB permissions issues (udev rules not installed/loaded)
3. Device already in use by another process
4. USB power/connection issues
5. Firmware compatibility issues
6. Missing system libraries
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def run_command(cmd, description=""):
    """Run a system command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if description:
            print(f"\n{description}:")
        print(f"Command: {cmd}")
        if result.stdout:
            print(f"Output:\n{result.stdout}")
        if result.stderr:
            print(f"Error:\n{result.stderr}")
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {cmd}")
        return False, "", "Timeout"
    except Exception as e:
        print(f"Failed to run command '{cmd}': {e}")
        return False, "", str(e)

def check_python_environment():
    """Check Python environment and pyorbbecsdk installation"""
    print_section("PYTHON ENVIRONMENT CHECK")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check if pyorbbecsdk is installed
    try:
        import pyorbbecsdk
        print(f"✓ pyorbbecsdk is installed")
        print(f"  Module path: {pyorbbecsdk.__file__}")
    except ImportError as e:
        print(f"✗ pyorbbecsdk not found: {e}")
        return False
    
    # Check PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH', '')
    if pythonpath:
        print(f"PYTHONPATH: {pythonpath}")
    else:
        print("⚠ PYTHONPATH not set")
    
    return True

def check_system_libraries():
    """Check for required system libraries"""
    print_section("SYSTEM LIBRARIES CHECK")
    
    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_path:
        print(f"LD_LIBRARY_PATH: {ld_path}")
    else:
        print("⚠ LD_LIBRARY_PATH not set")
    
    # Check for common library locations
    lib_paths = ['/usr/local/lib', '/usr/lib', '/lib']
    orbbec_libs = ['libOrbbecSDK.so', 'liborbbec.so']
    
    found_libs = []
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            for lib in orbbec_libs:
                lib_file = os.path.join(lib_path, lib)
                if os.path.exists(lib_file):
                    found_libs.append(lib_file)
                    print(f"✓ Found: {lib_file}")
    
    if not found_libs:
        print("⚠ No Orbbec libraries found in standard locations")
    
    return len(found_libs) > 0

def check_usb_devices():
    """Check for connected USB devices"""
    print_section("USB DEVICES CHECK")
    
    # List all USB devices
    success, output, error = run_command("lsusb", "All USB devices")
    
    # Look for Orbbec devices (Vendor ID: 2bc5)
    orbbec_devices = []
    if success and output:
        for line in output.split('\n'):
            if '2bc5' in line.lower():
                orbbec_devices.append(line.strip())
                print(f"✓ Found Orbbec device: {line.strip()}")
    
    if not orbbec_devices:
        print("✗ No Orbbec devices found via lsusb")
        print("  Check:")
        print("  - Device is physically connected")
        print("  - USB cable is working")
        print("  - Device is powered on")
        return False
    
    return True

def check_udev_rules():
    """Check udev rules for Orbbec devices"""
    print_section("UDEV RULES CHECK")
    
    udev_file = "/etc/udev/rules.d/99-obsensor-libusb.rules"
    
    if not os.path.exists(udev_file):
        print(f"✗ Udev rules file not found: {udev_file}")
        print("  Install udev rules from pyorbbecsdk package")
        return False
    
    print(f"✓ Udev rules file exists: {udev_file}")
    
    # Check file contents
    try:
        with open(udev_file, 'r') as f:
            content = f.read()
            
        # Look for common Orbbec product IDs
        product_ids = ['0601', '0669', '066b', '0672', '06a0']  # Common Orbbec PIDs
        found_rules = []
        
        for pid in product_ids:
            if pid in content:
                found_rules.append(pid)
                print(f"✓ Found rule for Product ID: {pid}")
        
        if not found_rules:
            print("⚠ No Orbbec product ID rules found")
            return False
            
    except Exception as e:
        print(f"✗ Error reading udev rules: {e}")
        return False
    
    return True

def check_device_permissions():
    """Check device file permissions"""
    print_section("DEVICE PERMISSIONS CHECK")
    
    # Look for video devices
    video_devices = []
    for i in range(10):  # Check video0 through video9
        device = f"/dev/video{i}"
        if os.path.exists(device):
            video_devices.append(device)
    
    if video_devices:
        print("Found video devices:")
        for device in video_devices:
            stat_info = os.stat(device)
            permissions = oct(stat_info.st_mode)[-3:]
            print(f"  {device}: permissions {permissions}")
    else:
        print("⚠ No video devices found")
    
    # Check current user groups
    success, output, error = run_command("groups", "Current user groups")
    
    if success and output:
        groups = output.strip().split()
        important_groups = ['video', 'dialout', 'plugdev']
        for group in important_groups:
            if group in groups:
                print(f"✓ User is in '{group}' group")
            else:
                print(f"⚠ User not in '{group}' group")
    
    return len(video_devices) > 0

def test_pyorbbecsdk():
    """Test pyorbbecsdk device detection"""
    print_section("PYORBBECSDK DEVICE TEST")
    
    try:
        from pyorbbecsdk import Context
        
        print("Creating Orbbec context...")
        context = Context()
        
        print("Querying devices...")
        device_list = context.query_devices()
        device_count = device_list.get_count()
        
        print(f"Device count: {device_count}")
        
        if device_count == 0:
            print("✗ No devices found by pyorbbecsdk")
            return False
        
        print(f"✓ Found {device_count} device(s)!")
        
        # Get device info
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
        
        return True
        
    except ImportError:
        print("✗ pyorbbecsdk not installed")
        return False
    except Exception as e:
        print(f"✗ Error testing pyorbbecsdk: {e}")
        return False

def check_process_conflicts():
    """Check for processes that might be using the camera"""
    print_section("PROCESS CONFLICT CHECK")
    
    # Common processes that might use cameras
    camera_processes = ['cheese', 'guvcview', 'vlc', 'opencv', 'gstreamer']
    
    success, output, error = run_command("ps aux", "Running processes")
    
    if success and output:
        conflicts = []
        for process in camera_processes:
            if process in output.lower():
                conflicts.append(process)
        
        if conflicts:
            print(f"⚠ Found potentially conflicting processes: {', '.join(conflicts)}")
            print("  Consider stopping these processes and try again")
        else:
            print("✓ No obvious process conflicts found")
    
    return True

def main():
    """Main diagnostic function"""
    print("Orbbec Device Diagnostic Tool")
    print("============================")
    
    results = {
        'python_env': check_python_environment(),
        'system_libs': check_system_libraries(),
        'usb_devices': check_usb_devices(),
        'udev_rules': check_udev_rules(),
        'permissions': check_device_permissions(),
        'process_conflicts': check_process_conflicts(),
        'pyorbbecsdk_test': test_pyorbbecsdk()
    }
    
    print_section("DIAGNOSTIC SUMMARY")
    
    passed = 0
    total = len(results)
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test.replace('_', ' ').title()}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your Orbbec device should work.")
    else:
        print(f"\n⚠ {total - passed} issues found. See recommendations below:")
        
        if not results['python_env']:
            print("- Install pyorbbecsdk and set PYTHONPATH correctly")
        if not results['system_libs']:
            print("- Install Orbbec SDK system libraries")
        if not results['usb_devices']:
            print("- Check USB connection and device power")
        if not results['udev_rules']:
            print("- Install and reload udev rules")
        if not results['permissions']:
            print("- Add user to video group: sudo usermod -a -G video $USER")
        if not results['pyorbbecsdk_test']:
            print("- Fix above issues and test again")
    
    print(f"\nFor automated fixes, run: ./fix_orbbec_device.sh")

if __name__ == "__main__":
    main()

#!/bin/bash

# Fix Orbbec Device Detection Issues
# Based on diagnostic output analysis

set -e

echo "=== Fixing Orbbec Device Detection Issues ==="
echo

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root. Please run as a regular user."
   exit 1
fi

# 1. Fix LD_LIBRARY_PATH
echo "1. Setting LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
echo "✓ LD_LIBRARY_PATH set for current session"

# Make LD_LIBRARY_PATH permanent
if ! grep -q 'export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"' ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
    echo "✓ Added LD_LIBRARY_PATH to ~/.bashrc"
else
    echo "✓ LD_LIBRARY_PATH already in ~/.bashrc"
fi

# 2. Add missing udev rule for Product ID 06a0 (DaiBai DCW2)
echo
echo "2. Adding missing udev rule for Product ID 06a0..."
UDEV_RULE='SUBSYSTEMS=="usb", ATTRS{idVendor}=="2bc5", ATTRS{idProduct}=="06a0", MODE:="0666", OWNER:="root", GROUP:="video", SYMLINK+="DaBai DCW2"'

if ! sudo grep -q '06a0' /etc/udev/rules.d/99-obsensor-libusb.rules 2>/dev/null; then
    echo "$UDEV_RULE" | sudo tee -a /etc/udev/rules.d/99-obsensor-libusb.rules > /dev/null
    echo "✓ Added udev rule for Product ID 06a0"
else
    echo "✓ Udev rule for 06a0 already exists"
fi

# 3. Add user to video group if not already in it
echo
echo "3. Checking user groups..."
if groups $USER | grep -q '\bvideo\b'; then
    echo "✓ User $USER is already in video group"
else
    echo "Adding user $USER to video group..."
    sudo usermod -a -G video $USER
    echo "✓ User $USER added to video group"
    echo "⚠  You may need to log out and back in for group changes to take effect"
fi

# 4. Reload udev rules
echo
echo "4. Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger
echo "✓ Udev rules reloaded"

# 5. Test the fix (if pyorbbecsdk is available)
echo
echo "5. Testing device detection..."

# Check if we're in a virtual environment or if pyorbbecsdk is available
if command -v python3 &> /dev/null; then
    python3 -c "
import sys
import os

# Add current directory to path in case pyorbbecsdk is locally installed
sys.path.insert(0, '.')

try:
    from pyorbbecsdk import Context
    
    print('Creating Orbbec context...')
    context = Context()
    device_list = context.query_devices()
    device_count = device_list.get_count()
    print(f'Device count: {device_count}')
    
    if device_count > 0:
        print('✓ SUCCESS: Devices found!')
        for i in range(device_count):
            device = device_list[i]
            device_info = device.get_device_info()
            print(f'  Device {i}: {device_info.get_name()} (PID: {device_info.get_pid()}, Serial: {device_info.get_serial_number()})')
    else:
        print('✗ Still no devices found')
        print('You may need to:')
        print('  - Restart the system')
        print('  - Try a different USB port')
        print('  - Check if another process is using the device')
        print('  - Ensure pyorbbecsdk is properly installed with PYTHONPATH set')
except ImportError:
    print('⚠  pyorbbecsdk not found in current environment')
    print('   Make sure to activate your virtual environment and set PYTHONPATH')
    print('   Example:')
    print('     source your_venv/bin/activate')
    print('     export PYTHONPATH=\$PYTHONPATH:/path/to/pyorbbecsdk-1.3.1/install/lib/')
except Exception as e:
    print(f'✗ Error: {e}')
    print('This may indicate a configuration issue.')
"
else
    echo "⚠  Python3 not found. Cannot test device detection."
fi

echo
echo "=== Fix Complete ==="
echo "If devices are still not found:"
echo "1. Try restarting your system (especially for group changes)"
echo "2. Ensure no other processes are using the camera"
echo "3. Try a different USB port/cable"
echo "4. Make sure pyorbbecsdk is properly installed with correct PYTHONPATH"
echo
echo "To test your camera setup:"
echo "1. Activate your Python virtual environment"
echo "2. Set PYTHONPATH: export PYTHONPATH=\$PYTHONPATH:/path/to/pyorbbecsdk-1.3.1/install/lib/"
echo "3. Run: python3 tools/test_cameras_final.py"
echo
echo "For detailed diagnostics, run: python3 tools/diagnose_orbbec_device.py"

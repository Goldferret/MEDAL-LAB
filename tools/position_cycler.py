#!/usr/bin/env python3
"""
Robot Position Cycler Tool

This tool cycles the robot through scan positions for HSV calibration.
Press Enter to move to the next position in the sequence.

Position sequence: [X, 110, 5, 0, 90] where X = [180, 135, 90, 45, 0]

Usage:
    python3 position_cycler.py

Controls:
    Enter: Move to next position
    Q: Quit
"""

import sys
import os
from pathlib import Path

# Add the nodes directory to the Python path
nodes_dir = Path(__file__).parent.parent / "nodes"
sys.path.insert(0, str(nodes_dir))

try:
    from Arm_Lib import Arm_Device
    import time
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Make sure you're running this from the MEDAL-LAB/tools directory")
    sys.exit(1)

class PositionCycler:
    def __init__(self):
        """Initialize the position cycler tool."""
        # Position sequence for servo 1 (base rotation)
        self.servo1_positions = [180, 135, 90, 45, 0]
        self.current_index = 0
        
        # Fixed positions for other servos [servo1, servo2, servo3, servo4, servo5, servo6]
        # Note: We only use first 5 servos, servo6 is not used in this application
        self.base_angles = [0, 110, 5, 0, 90, 90]  # servo1 will be replaced, servo6 set to 90
        
        # Movement time in milliseconds
        self.movement_time = 2000  # 2 seconds for smooth movement
        
        # Initialize robot arm device
        try:
            self.arm = Arm_Device()
            print("✅ Robot arm device initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize robot arm device: {e}")
            print("Make sure the robot is connected and powered on")
            sys.exit(1)
        
        # Test robot connection by checking hardware version
        try:
            version = self.arm.Arm_get_hardversion()
            if version:
                print(f"✅ Robot connected - Hardware version: {version}")
            else:
                print("⚠️ Could not read hardware version, but proceeding...")
        except Exception as e:
            print(f"⚠️ Warning: Could not verify robot connection: {e}")
        
        # Enable torque for movement
        try:
            print("🔧 Enabling robot torque...")
            self.arm.Arm_serial_set_torque(1)  # 1 = enable torque
            print("✅ Robot torque enabled")
        except Exception as e:
            print(f"❌ Failed to enable torque: {e}")
            sys.exit(1)
        
        print("\n🤖 Robot Position Cycler Ready!")
        print("📋 Position sequence:")
        for i, pos in enumerate(self.servo1_positions):
            print(f"  {i+1}. [{pos}, 110, 5, 0, 90]")
        print(f"\n🎯 Starting at position 1: [{self.servo1_positions[0]}, 110, 5, 0, 90]")
    
    def get_current_position_angles(self):
        """Get the joint angles for the current position."""
        target_angles = self.base_angles.copy()
        target_angles[0] = self.servo1_positions[self.current_index]  # Update servo 1
        return target_angles[:5]  # Return only first 5 servos
    
    def move_to_current_position(self):
        """Move robot to the current position in the sequence."""
        target_angles = self.get_current_position_angles()
        current_servo1 = self.servo1_positions[self.current_index]
        
        print(f"\n🔄 Moving to position {self.current_index + 1}/{len(self.servo1_positions)}")
        print(f"   Target: [{current_servo1}, 110, 5, 0, 90]")
        
        try:
            # Move all servos to target position
            # Arm_serial_servo_write6(s1, s2, s3, s4, s5, s6, time)
            self.arm.Arm_serial_servo_write6(
                target_angles[0],  # servo 1 (base)
                target_angles[1],  # servo 2 (shoulder) 
                target_angles[2],  # servo 3 (elbow)
                target_angles[3],  # servo 4 (wrist)
                target_angles[4],  # servo 5 (gripper rotation)
                90,                # servo 6 (gripper open/close) - fixed at 90
                self.movement_time
            )
            
            # Wait for movement to complete (movement_time + buffer)
            wait_time = (self.movement_time / 1000.0) + 0.5  # Convert to seconds + 0.5s buffer
            print(f"⏳ Moving... (waiting {wait_time:.1f} seconds)")
            time.sleep(wait_time)
            
            print(f"✅ Reached position {self.current_index + 1}: [{current_servo1}, 110, 5, 0, 90]")
            
        except Exception as e:
            print(f"❌ Error moving to position: {e}")
            return False
        
        return True
    
    def next_position(self):
        """Move to the next position in the sequence."""
        self.current_index = (self.current_index + 1) % len(self.servo1_positions)
        return self.move_to_current_position()
    
    def run(self):
        """Run the position cycler tool."""
        print("\n📋 Controls:")
        print("  Enter: Move to next position")
        print("  Q + Enter: Quit")
        print("\n🚀 Press Enter to start moving through positions...")
        
        # Move to first position
        if not self.move_to_current_position():
            return
        
        try:
            while True:
                # Wait for user input
                current_servo1 = self.servo1_positions[self.current_index]
                next_index = (self.current_index + 1) % len(self.servo1_positions)
                next_servo1 = self.servo1_positions[next_index]
                
                user_input = input(f"\n[Current: {current_servo1}°] Press Enter for next position ({next_servo1}°) or Q to quit: ").strip().lower()
                
                if user_input == 'q':
                    print("👋 Exiting position cycler...")
                    break
                elif user_input == '':
                    # Move to next position
                    if not self.next_position():
                        break
                else:
                    print("❓ Unknown command. Press Enter to continue or Q to quit.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Exiting...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources and disable torque."""
        try:
            if hasattr(self, 'arm'):
                print("🔧 Disabling robot torque...")
                self.arm.Arm_serial_set_torque(0)  # 0 = disable torque
                print("✅ Robot torque disabled")
                print("🧹 Robot arm cleaned up")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")

def main():
    """Main function."""
    print("🤖 Robot Position Cycler Tool")
    print("=" * 40)
    
    try:
        cycler = PositionCycler()
        cycler.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

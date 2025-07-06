#!/usr/bin/env python3
from Arm_Lib import Arm_Device
from time import sleep
from arm_controller import ArmController

""" 
A simple program which prints all servo angles of the robot arm 
after physically adjusting the arm to a desired position by hand
"""

class ArmActions:
    def __init__(self, controller: ArmController):
        self.controller = controller

if __name__ == "__main__":
    # Initialize hardware connection
    arm_controller = ArmController()
    actions = ArmActions(arm_controller)
    
    try:
        input("Press enter to release torque...")
        print(actions.controller.set_torque(False))
        
        input("Move arm to desired position, then press enter...")
        print("Servo state:",actions.controller.read_current_state())
        print(actions.controller.set_torque(True))
        
        input("Press enter to return robot to home position and shutdown script...")
        actions.controller.move_home(2000)
        sleep(2)
        
    except KeyboardInterrupt:
        print("\nEmergency shutdown initiated!")
        actions.controller.move_home(2000)
        sleep(4)
        print("Arm safely homed.")
        
    finally:
        print(actions.controller.set_torque(False))
        print("Script completed.")  
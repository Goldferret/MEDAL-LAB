#!/usr/bin/env python3
# encoding: utf-8
import threading
import rospy
from Arm_Lib import Arm_Device
from dofbot_pro_info.msg import *
from std_msgs.msg import Bool
import time
from sensor_msgs.msg import  JointState
import numpy as np
from math import pi

class ArmDriver:
    def __init__(self):
       self.Arm = Arm_Device()
       self.sub_Arm = rospy.Subscriber("TargetAngle", ArmJoint, self.Armcallback, queue_size=1000)
       self.sub_Buzzer = rospy.Subscriber("Buzzer", Bool, self.Buzzercallback,queue_size=1000)
       self.staPublisher = rospy.Publisher('joint_states', JointState, queue_size=1000)
       self.ArmPubUpdate = rospy.Publisher("ArmAngleUpdate", ArmJoint, queue_size=1000)
       self.joints = [90, 145, 0, 45, 90, 30] 
       self.cur_joints = [90.0, 90.0, 90.0, 00.0, 90.0,30] 
       self.Prefix = rospy.get_param("~prefix", "")
       self.RA2DE = 180 / pi
    
    def Armcallback(self,msg):
        if not isinstance(msg, ArmJoint): 
            print("----------")
            return
        arm_joint = ArmJoint()
        print("msg.joints: ",msg.joints)
        print("msg.joints: ",msg.run_time)
        if len(msg.joints) != 0:
            arm_joint.joints = self.cur_joints
            for i in range(2):
                print("--------------------------")
                self.Arm.Arm_serial_servo_write6(msg.joints[0], msg.joints[1],msg.joints[2],msg.joints[3],msg.joints[4],msg.joints[5],time=msg.run_time)
                self.cur_joints = list(msg.joints)
                self.ArmPubUpdate.publish(arm_joint)
			#time.sleep(0.01)
        else:
            arm_joint.id = msg.id
            arm_joint.angle = msg.angle
            for i in range(2):
                print("msg.id: ",msg.id)
                self.Arm.Arm_serial_servo_write(msg.id, msg.angle, msg.run_time)
                self.cur_joints[msg.id - 1] = msg.angle
                self.ArmPubUpdate.publish(arm_joint)
        self.joints_states_update()
			
    def Buzzercallback(self,msg):
        if not isinstance(msg, Bool): 
            return
        if msg.data==True:
            print("beep on")	
            self.Arm.Arm_Buzzer_On()
        else:
            self.Arm.Arm_Buzzer_Off()
            print("beep off")	

    def read_current_joint(self):
        # Try to read all servos, but don't let failures block forever
        read_success = True
        for i in range(6):
            time.sleep(.01)
            value = self.Arm.Arm_serial_servo_read(i+1)
            if value is not None:
                self.cur_joints[i] = value
            else:
                read_success = False
                break  # No point reading remaining servos if we won't publish
            time.sleep(.01)
        
        # Only publish if we got valid reads
        if read_success:
            self.joints_states_update()
        else:
            timestamp = rospy.Time.now()
            print(f"[{timestamp.to_sec():.3f}] Servo read failed")
            
        # NOTE: Increased to 2 seconds for calibration to reduce servo bus contention
        # Polling interval for reading servo positions
        time.sleep(0.5)

    def pub_cur_joints(self):
        while not rospy.is_shutdown():
            time.sleep(0.05)
            self.read_current_joint()

        
    def joints_states_update(self):
        # Validate all servo reads succeeded
        if None in self.cur_joints:
            rospy.logwarn_throttle(5, "Skipping joint_states publish - servo read failed")
            return
            
        state = JointState()
        state.header.stamp = rospy.Time.now()
        state.header.frame_id = "joint_states"
        if len(self.Prefix) == 0:
            state.name = ["Arm1_Joint", "Arm2_Joint", "Arm3_Joint", "Arm4_Joint", "Arm5_Joint", "grip_joint"]
        else:
            state.name = [self.Prefix + "/Arm1_Joint", self.Prefix + "/Arm2_Joint",
                          self.Prefix + "/Arm3_Joint", self.Prefix + "/Arm4_Joint",
                          self.Prefix + "/Arm5_Joint", self.Prefix + "/grip_joint"]
        joints = self.cur_joints[0:]
        joints[5] = np.interp(joints[5], [30, 180], [0, 90])  # Original mapping
        mid = np.array([90, 90, 90, 90, 90, 90])
        array = np.array(np.array(joints) - mid)
        DEG2RAD = np.array([pi / 180])
        position_src = list(np.dot(array.reshape(-1, 1), DEG2RAD))
        state.position = position_src
        try:
            self.staPublisher.publish(state)
        except Exception as e:
            rospy.logerr_throttle(5, f"Publish error: {e}")
			

if __name__ == '__main__':
    rospy.init_node('Arm_Driver_Node', anonymous=True)
    arm_driver = ArmDriver()
    # NOTE: Commented out home position command for calibration
    # Robot will stay in current position when arm_driver starts
    # arm_driver.Arm.Arm_serial_servo_write6(90.0, 90.0, 90.0, 0.0, 90.0, 180.0, 3000)
    rospy.sleep(1)  # Give publisher time to initialize
    
    # Start continuous joint state publishing in background thread
    joint_pub_thread = threading.Thread(target=arm_driver.pub_cur_joints)
    joint_pub_thread.daemon = True
    joint_pub_thread.start()
    
    rospy.spin()

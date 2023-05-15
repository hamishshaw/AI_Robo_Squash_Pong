import pybullet as p
import os
import math


class Arm:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'arm.urdf')
        self.arm = client.loadURDF(fileName=f_name,
                   basePosition=[-3, 0, 0])
        # to set joint positions
        self.joint_pos = [0, 1]
        # sets angle for both joints
        self.steering_angle = [0, 0]

    def get_ids(self):
        return self.arm
        # used to move the arm, action is 2d array of move joint by increment
    def apply_action(self, action):
        #move joint 0
        self.steering_angle[0] += action[0]
        self.client.setJointMotorControlArray(self.arm, self.joint_pos,
                                   controlMode=p.POSITION_CONTROL,
                                    targetPositions=[self.steering_angle[0]] * 2)         
        #move join 1
        #self.steering_angle[1] = action[1]
        #self.client.setJointMotorControlArray(self.arm, [1],
        #                        controlMode=p.POSITION_CONTROL,
        #                        targetPositions=[self.steering_angle[1]] * 2)  

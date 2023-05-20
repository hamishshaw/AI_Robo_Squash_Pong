import gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal 
from simple_driving.resources.puck import Puck 
from simple_driving.resources.wall import Wall 
import random 
import matplotlib.pyplot as plt
import time
from simple_driving.resources.arm import Arm


RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'fp_camera', 'tp_camera']}

    def __init__(self, isDiscrete=True, renders=False):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-40, -40, -1, -1, -5, -5, -10, -10], dtype=np.float32),
            high=np.array([40, 40, 1, 1, 5, 5, 10, 10], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 2
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.puck = None 
        self.wall = None 
        self.threshold_velocity = 10 
        self.done = False
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self._envStepCounter = 0
        self.arm = None
        self.arm_joint_pos = None
        self.prev_dist_to_goal = None

    def step(self, action):
        # THis is the puck kicker to ensure it keeps moving
        linear_velocity, _ = p.getBaseVelocity(self.puck.puck)
        current_velocity_magnitude = abs(linear_velocity[0]) + abs(linear_velocity[1]) + abs(linear_velocity[2])
        if current_velocity_magnitude < self.threshold_velocity:
            # Calculate the corrective force based on the desired threshold and current velocity
            corrective_force_magnitude = (self.threshold_velocity - current_velocity_magnitude) * 100  # puckmass = 1

            if current_velocity_magnitude != 0:
                # Get the direction of the current velocity
                direction = (linear_velocity[0] / current_velocity_magnitude,
                            linear_velocity[1] / current_velocity_magnitude,
                            linear_velocity[2] / current_velocity_magnitude)
                corrective_force = (corrective_force_magnitude * direction[0],
                                corrective_force_magnitude * direction[1],
                                corrective_force_magnitude * direction[2])

                # Apply the corrective force to the puck
                p.applyExternalForce(self.puck.puck, -1, forceObj=corrective_force, posObj=[0, 0, 0], flags=p.WORLD_FRAME)
            # else: ## add kicker to start the process again
            #     # If the velocity is exactly zero, set the direction to (0, 0, 0)
            #     direction = (1, 0, 0)

            # Calculate the corrective force vector in the same direction as the current velocity        
        # Feed action to the car and get observation of car's state
        if (self._isDiscrete):
            # move the arm based of action, these are the choices
            # joint 0 = + joint 1 = none
            # joint 0 = - joint 1 = none
            # joint 0 = none joint 1 = +
            # joint 0 = none joint 1 = - 
            # joint 0 = +  joint 1  = +
            # joint 0 = -  joint 1 = -
            # joint 0 = none   joint 1 = none   
            joint_0 = [1, -1, 0, 0, 1, -1, -1, 1, 0]
            joint_1 = [0, 0, 1, -1, 1, -1, 1, -1, 0]

            self.arm_joint_pos[0] += joint_0[action]*0.05
            self.arm_joint_pos[1] += joint_1[action]*0.05

        # this adjusts the joint angles of the arm.
        self.arm.apply_action(self.arm_joint_pos)
        end_aff_pos = self.getExtendedObservation()

        for i in range(self._actionRepeat):
          self._p.stepSimulation()
          if self._renders:
            time.sleep(self._timeStep)

          puckpos, puckorn = self._p.getBasePositionAndOrientation(self.puck.puck)
          
          if self._termination():
            self.done = True
            break
          self._envStepCounter += 1
        

        #dist_to_goal = math.sqrt((puckpos[1] - end_aff_pos[1])**2 +  (puckpos[0] - end_aff_pos[0])**2)
        dist_to_goal = puckpos[1] - end_aff_pos[1]
        #print(f"end affector at x = {end_aff_pos[0]}, y = {end_aff_pos[1]} ")
        #print(end_aff_pos)
        #reward = self.prev_dist_to_goal-dist_to_goal * -.1

        reward = -abs(dist_to_goal)
        reward += abs(2 + end_aff_pos[0])/5
        #self.prev_dist_to_goal = dist_to_goal
        #if dist_to_goal < 0.15 and dist_to_goal > -0.15:
        #    reward = 1 
        #if end_aff_pos[0] > - 2: 
        #    reward += 0.1
        #reward = (1 - abs(dist_to_goal))
        #reward += (2 + end_aff_pos[0])*0.5



        total_dist_to_goal = math.sqrt((puckpos[0]- end_aff_pos[0])**2 + (puckpos[1]- end_aff_pos[1])**2)
        # Done by reaching goal
        if total_dist_to_goal < 0.3 :
            #print("nice one")
            reward = 10
        if puckpos[0] < -2.2 or (puckpos[1] > 2.1) or (puckpos[1]<-2.1):
            #print("reached goal")
            reward = -50
            self.done = True
            self.reached_goal = True

       
        # return the arm end affector pos and puck location
        arm_info = [end_aff_pos[0],end_aff_pos[1],puckpos[0],puckpos[1]]

        #print(str(puckpos[0]) + " x," + str(puckpos[1]) + "y, Position")
        #print(f"end affector at x = {end_aff_pos[0]}, y = {end_aff_pos[1]}")
        return arm_info, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self._p)
        self._envStepCounter = 0
        self.arm = Arm(self._p)
        self.arm_joint_pos = [0,0]

        self.goal = (10,-10)
        self.done = False
        self.reached_goal = False

        initalPuckPos = (0,0)  #X,Y coords of initial placement
        # initalPuckMomentum = (1,0) # Both numbers add to 1 for 500 units of initial kick first is X kick second is Y kick
        # Randomised initial direction going backwards
        #
        # _____________________________________
        #
        b = random.uniform(-1, 1)
        a = 1 - abs(b)
        initalPuckMomentum = (a,b)
        
        self.puck = Puck(self._p,initalPuckPos,initalPuckMomentum) 
        self.wall = Wall (self._p,(0,0),self.puck) 
        p.setTimeStep(1.0 / 240.0)  # Set a smaller time step for more accurate simulation 
        p.setPhysicsEngineParameter(numSolverIterations=10) 
        

        # Get observation to return
        arm_end_aff_pos = self.getExtendedObservation()
        self.prev_dist_to_goal =   math.sqrt((initalPuckPos[0]- arm_end_aff_pos[0])**2 + (initalPuckPos[1]- arm_end_aff_pos[1])**2)
        # TODO return puck pos and end affector position
        arm_info = [arm_end_aff_pos[0],arm_end_aff_pos[1],initalPuckPos[0],initalPuckPos[1]]
        return np.array(arm_info, dtype=np.float32)

    def render(self, mode='human'):
        if mode == "fp_camera":
            # Base information
            arm_id = self.arm.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                       nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in
                        self._p.getBasePositionAndOrientation(arm_id)]
            pos[2] = 0.2

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # Display image
            # frame = self._p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
            # frame = np.reshape(frame, (100, 100, 4))
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
            # self.rendered_img.set_data(frame)
            # plt.draw()
            # plt.pause(.00001)

        elif mode == "tp_camera":
            # car_id = self.car.get_ids()
            base_pos = (0,0,0)#, orn = self._p.getBasePositionAndOrientation(car_id)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos, 
                                                                    distance=3.46410161028, 
                                                                    yaw=-90.0, 
                                                                    pitch=-90, 
                                                                    roll=0, 
                                                                    upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        
        # TODO returns the end affector locaiton
        # some basic forward kinematics
        basepos = [-2.5 , 0]
        x = basepos[0] + 1*math.cos(self.arm_joint_pos[0]) + math.cos(sum(self.arm_joint_pos))
        y = basepos[1] + 1*math.sin(self.arm_joint_pos[0]) + math.sin(sum(self.arm_joint_pos))
        
        observation = [x,y]
        return observation

    def _termination(self):
        return self._envStepCounter > 2000

    def close(self):
        self._p.disconnect()

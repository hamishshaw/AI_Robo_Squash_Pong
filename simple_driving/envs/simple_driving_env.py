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
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.puck = None 
        self.wall = None 
        self.threshold_velocity = 10 
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self._envStepCounter = 0
        self.arm = None

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
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            throttle = fwd[action]
            steering_angle = steerings[action]
            action = [throttle, steering_angle]
        # self.car.apply_action(action)
        self.arm.apply_action([0.05, 0.05])
        for i in range(self._actionRepeat):
          self._p.stepSimulation()
          if self._renders:
            time.sleep(self._timeStep)

        #   carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        #   goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
          puckpos, puckorn = self._p.getBasePositionAndOrientation(self.puck.puck)
          car_ob = self.getExtendedObservation()
          
          if self._termination():
            self.done = True
            break
          self._envStepCounter += 1
        
        # Compute reward as L2 change in distance to goal
        # dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  # (car_ob[1] - self.goal[1]) ** 2))
        # dist_to_goal = math.sqrt(((carpos[0] - goalpos[0]) ** 2 +
        #                           (carpos[1] - goalpos[1]) ** 2))
        dist_to_goal = puckpos[0] - (-2)
        # reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        reward = dist_to_goal
        self.prev_dist_to_goal = dist_to_goal

        # Done by reaching goal
        if puckpos[0] < -2 and not self.reached_goal:
            #print("reached goal")
            reward = -50
            self.done = True
            self.reached_goal = True

        ob = car_ob
        print(str(puckpos[0]) + " x," + str(puckpos[1]) + "y, Position")
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self._p)
        self.car = Car(self._p)
        self._envStepCounter = 0
        self.arm = Arm(self._p)
        # Set the goal to a random target
        # x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
        #      self.np_random.uniform(-9, -5))
        # y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
        #      self.np_random.uniform(-9, -5))
        self.goal = (10,-10)
        self.done = False
        self.reached_goal = False

        # Visual element of the goal
        self.goal_object = Goal(self._p, self.goal)

        initalPuckPos = (0,0)  #X,Y coords of initial placement
        # initalPuckMomentum = (1,0) # Both numbers add to 1 for 500 units of initial kick first is X kick second is Y kick
        # Randomised initial direction going backwards
        #
        # _____________________________________
        #
        b = random.uniform(-1, 1)
        a = 1 - abs(b)
        initalPuckMomentum = (a,b)
        #
        
        self.puck = Puck(self._p,initalPuckPos,initalPuckMomentum) 
        self.wall = Wall (self._p,(0,0),self.puck) 
        p.setTimeStep(1.0 / 240.0)  # Set a smaller time step for more accurate simulation 
        p.setPhysicsEngineParameter(numSolverIterations=10) 
        

        # Get observation to return
        carpos = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((carpos[0] - self.goal[0]) ** 2 +
                                           (carpos[1] - self.goal[1]) ** 2))
        car_ob = self.getExtendedObservation()
        return np.array(car_ob, dtype=np.float32)

    def render(self, mode='human'):
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                       nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in
                        self._p.getBasePositionAndOrientation(car_id)]
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
        # self._observation = []  #self._racecar.getObservation()
        carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        goalPosInCar, goalOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, goalpos, goalorn)

        observation = [goalPosInCar[0], goalPosInCar[1]]
        return observation

    def _termination(self):
        return self._envStepCounter > 2000

    def close(self):
        self._p.disconnect()

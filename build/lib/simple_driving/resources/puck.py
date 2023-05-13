import pybullet as p
import os


class Puck:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'ai_simplepuck.urdf')
        self.puck = client.loadURDF(fileName=f_name, basePosition=[base[0], base[1], 0.05])
        p.changeDynamics(self.puck,linkIndex=-1,lateralFriction = 0, restitution = 0.95, rollingFriction = 99999) 
        initial_linear_momentum = [300, 300, 0]
        p.applyExternalForce(self.puck, linkIndex=-1, forceObj=initial_linear_momentum, posObj=[0, 0, 0], flags=p.WORLD_FRAME)
            #https://www.programcreek.com/python/example/122142/pybullet.changeDynamics
            #https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
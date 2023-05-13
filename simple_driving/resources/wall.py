import pybullet as p
import os


class Wall:
   def __init__(self, client, base, puck):
        f_name = os.path.join(os.path.dirname(__file__), 'ai_simplewall.urdf')
        self.wall = client.loadURDF(fileName=f_name, basePosition=[base[0], base[1], 0])
        p.changeDynamics(self.wall, linkIndex=-1, restitution=1.0)  # Set wall's restitution to 1.0
        p.changeDynamics(self.wall, linkIndex=-1, lateralFriction=0)
        # Get the collision shape ID for the second link
        num_joints = p.getNumJoints(self.wall)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.wall, i)
            if joint_info[12].decode('UTF-8') == 'wall1':
                wall_collision_id = joint_info[0]
                break

        # Enable collision detection between the wall and the puck
        p.setCollisionFilterGroupMask(self.wall, -1, 0, 0)  # Enable collisions with all groups
        p.setCollisionFilterPair(self.wall, puck.puck, -1, -1, 1)  # Enable collisions with the puck
        p.setInternalSimFlags(1)


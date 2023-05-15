import gym
import math
import simple_driving
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# from matplotlib.widgets import Button # not used as was for adding reset button that didnt work
from scipy.cluster.vq import kmeans2

# adding for complex colour detection
import cv2

def extract_colour_pixels(image, colour):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Convert the target color to the corresponding HSV values
    colour_hsv = cv2.cvtColor(np.uint8([[colour]]), cv2.COLOR_RGB2HSV)[0][0]

    # Define the range of acceptable hue values
    hue_range = 10

    # Define the lower and upper bounds for the color range
    lower_colour = np.array([colour_hsv[0] - hue_range, 50, 50])
    upper_colour = np.array([colour_hsv[0] + hue_range, 255, 255])

    # Create a mask based on the color range
    mask = cv2.inRange(hsv_image, lower_colour, upper_colour)

    # Bitwise-AND the mask and the original image to extract color pixels
    colour_image = cv2.bitwise_and(image, image, mask=mask)

    return colour_image


def get_non_black_pixels(image):
    non_black_pixels = np.argwhere(np.any(image != [0, 0, 0], axis=2))
    return [(float(x), float(y)) for y, x in non_black_pixels]

def kmeans(inputpoints, k):
  centroid, label = kmeans2(inputpoints, k,minit = 'points', missing = 'warn')
  return (centroid, label)

def map_pixels_to_coordinates(pixel_indices,RENDER_WIDTH,RENDER_HEIGHT):
    coordinates = []
    for pixel_index in pixel_indices:

        pixel_index_y = 4*(   (pixel_index[0]-  ((RENDER_WIDTH-RENDER_HEIGHT)/2))    / RENDER_HEIGHT)            - 2 
        pixel_index_x = 4*(pixel_index[1]/ RENDER_HEIGHT) - 2

        # x = (2 * pixel_index_x / RENDER_WIDTH) - 1
        # y = (2 * pixel_index_y / RENDER_HEIGHT) - 1
        if abs(pixel_index_x)>2:
            print ("errorMapping x")
        if abs(pixel_index_y)>2:
            print ("errorMapping y")

        coordinates.append((-pixel_index_x, -pixel_index_y)) # negatives are due to camera importation
    
    return coordinates
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera') 
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True,render_mode='tp_camera') 
state, info = env.reset()
green_colour = [0, 236, 0] #IMPORTANT TO RECONSIES THE COLOUR OF THE TARGET
n_clusters = 1 # number of pucks in simulation

frames = []
redframes = []
startframe = env.render()

frames.append(startframe)
tempcolourframe = extract_colour_pixels(startframe,green_colour)
redframes.append(tempcolourframe)
renderwidth = startframe.shape[1]
renderheight = startframe.shape[0]
# print(startframe.shape)
# print(startframe.shape[0])
# print(startframe.shape[1])
points =  get_non_black_pixels(tempcolourframe)
for i in range(200):
    action = env.action_space.sample()
    state, reward, done, _, info = env.step(action)
    tempframe = env.render()
    frames.append(tempframe)
    
    tempcolourframe = extract_colour_pixels(tempframe,green_colour)
    temppoints = get_non_black_pixels(tempcolourframe)
    if not (len(temppoints) == 0):
        points = temppoints
    # print(points)
    # print(n_clusters)
    centroids, assignment = kmeans(points, n_clusters)
    centroids = map_pixels_to_coordinates(centroids,renderwidth,renderheight)
    print(str(centroids[0][0]) + " x," + (str(centroids[0][1])) + "y, Calculated Pos")
    # print(centroids)
    # print(centroids[0])
    # print(centroids[0][0])
    # print(assignment)
    redframes.append(tempcolourframe)
    if done:
        break

env.close()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.axis("off")
ax2.axis("off")
im1 = ax1.imshow(frames[0])
im2 = ax2.imshow(redframes[0])

def update(frame):
    im1.set_data(frames[frame])
    im2.set_data(redframes[frame])
    return im1, im2

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, repeat=True)
plt.show()
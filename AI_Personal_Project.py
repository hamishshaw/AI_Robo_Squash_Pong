import gym
import simple_driving
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def extract_colour_pixels(image, colour):
    # Convert the colour to the corresponding RGB values
    colour_rgb = np.array(colour) * 255

    # Define the lower and upper bounds for the colour range
    lower_colour = np.maximum(colour_rgb - 1, 0).astype(np.uint8)
    upper_colour = np.minimum(colour_rgb + 1, 255).astype(np.uint8)

    # Mask the colour pixels
    colour_mask = np.logical_and(np.all(image >= lower_colour, axis=2), np.all(image <= upper_colour, axis=2))

    # Create an image with only the colour pixels
    colour_image = np.zeros_like(image)
    colour_image[colour_mask] = image[colour_mask]

    return colour_image


# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera') 
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True,render_mode='tp_camera') 
state, info = env.reset()


frames = []
redframes = []

frames.append(env.render())

for i in range(200):
    action = env.action_space.sample()
    state, reward, done, _, info = env.step(action)
    tempframe = env.render()
    frames.append(tempframe)
    # print(tempframe.shape)
    green_colour = [0, 236, 0]
    tempcolourframe = extract_colour_pixels(tempframe,green_colour)
    redframes.append(tempcolourframe)
    if done:
        break

env.close()

# Display the frames as a video
fig, ax = plt.subplots()
ax.axis("off")
im = ax.imshow(frames[0])

def update(frame):
    im.set_data(frame)
    return im,

ani = animation.FuncAnimation(fig, update, frames=frames, interval=200)
# plt.show()

fig, ax = plt.subplots()
ax.axis("off")
im = ax.imshow(redframes[0])

def update(redframes):
    im.set_data(redframes)
    return im,

ani = animation.FuncAnimation(fig, update, frames=redframes, interval=200)
plt.show()
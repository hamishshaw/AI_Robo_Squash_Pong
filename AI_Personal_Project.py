import gym
import simple_driving
import matplotlib.pyplot as plt
import matplotlib.animation as animation


env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera') 
state, info = env.reset()
frames = []
frames.append(env.render())

for i in range(200):
    action = env.action_space.sample()
    state, reward, done, _, info = env.step(action)
    frames.append(env.render())
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

ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
plt.show()

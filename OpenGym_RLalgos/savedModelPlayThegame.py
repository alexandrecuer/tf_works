import numpy as np
import gym
import random
import math
import tensorflow as tf
import matplotlib.pylab as plt

model = tf.keras.models.load_model('moutainCar.h5')
print(model.layers[0].input_shape)
model.summary()

env_name = 'MountainCar-v0'
env = gym.make(env_name)
numStates = env.env.observation_space.shape[0]

num_episodes = 100

steps=0
reward_store = []
max_x_store = []

for episode in range(num_episodes):
    state = env.reset()
    if episode % 10 == 0:
        print('Episode {} of {}'.format(episode+1, num_episodes))
    tot_reward = 0
    max_x = -100
    while True:
        env.render()
        rspd=state.reshape(1,numStates)
        predictionBrute = model(rspd)
        action = np.argmax(predictionBrute)
        nextState, reward, done, info = env.step(action)

        if nextState[0] >= 0.5:
            reward += 100
            print("Top of the hill reached after {} timesteps".format(steps))
        elif nextState[0] >= 0.25:
            reward += 20
        elif nextState[0] >= 0.1:
            reward += 10
        if nextState[0] > max_x:
            max_x = nextState[0]

        steps+=1
        # update state
        state = nextState

        tot_reward += reward

        if done:
            reward_store.append(tot_reward)
            max_x_store.append(max_x)
            break

    print("step {} Total reward {}".format(steps,tot_reward))

env.close()
plt.plot(reward_store)
plt.show()
plt.close("all")
plt.plot(max_x_store)
plt.show()

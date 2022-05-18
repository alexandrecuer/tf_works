import numpy as np
from matplotlib import pyplot as plt

import gym
env = gym.make("CartPole-v0")

import tensorflow as tf
from tensorflow import keras
import datetime as dt

# pour lancer tensorboard : tensorboard --logdir=TensorBoard/PolicyGradient
game = "CartPole"
DIR =  "TensorBoard/PolicyGradient"
STORE_PATH = "{}/{}".format(DIR,game)
GAMMA = 0.99
tb = False

state_size = env.observation_space.shape[0]
num_actions = env.action_space.n

def get_action(network, state, num_actions):
    # state as returned by Gym is a numpy array so we can use reshape
    softmax_out = network(state.reshape((1, -1)))
    selected_action = np.random.choice(num_actions, p=softmax_out.numpy()[0])
    #selected_action = np.argmax(softmax_out)
    return selected_action

def watch_agent():
  state = env.reset()
  rewards = []
  for t in range(200):
    env.render()
    action = get_action(network, state, num_actions)
    state, reward, done, _ = env.step(action)
    rewards.append(reward)
    if done:
        print("Reward:", sum([r for r in rewards]))
        break

def update_network(network, rewards, actions, states):
    """
    policy gradient REINFORCE vanilla algorithm
    """
    reward_sum = 0
    discounted_rewards = []
    for reward in rewards[::-1]:  # reverse buffer r
        reward_sum = reward + GAMMA * reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()
    discounted_rewards = np.array(discounted_rewards)
    # standardise the rewards
    #discounted_rewards -= np.mean(discounted_rewards)
    #discounted_rewards /= np.std(discounted_rewards)
    discounted_rewards /= discounted_rewards.max()
    states = np.vstack(states)
    with tf.GradientTape() as tape:
        predictions = network(states)
        # solution 1 - peu élégante
        #indices = np.array([[1 if a==i else 0 for i in range(2)] for a in actions])
        #actions_probs = tf.reduce_sum(predictions * indices, axis=1)
        # solution 2
        indices = [[i,actions[i]] for i in range(len(actions))]
        actions_probs = tf.gather_nd(predictions, indices)
        # en numpy, on ferait ceci, mais on perd la référence au gradient
        # actions_probs = predictions.numpy()[range(predictions.shape[0]), actions])
        loss = tf.reduce_sum(-tf.math.log(actions_probs) * discounted_rewards)
    #by default, the ressources held by a gradient tape are released as soon as GradienTape.gradient() method is called
    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

import os
import readline
import glob
def simplePathCompleter(text,state):
    """
    tab completer pour les noms de fichiers, chemins....
    """
    line   = readline.get_line_buffer().split()

    return [x for x in glob.glob(text+'*')][state]

def pickName(name = None, autocomplete = True):
    """
    vérifie un chemin ou un nom de fichier fourni en argument ou saisi en autocomplétion par l'utilisateur
    """
    if name is None and autocomplete:
        readline.set_completer_delims('\t')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(simplePathCompleter)
        name = input("nom du réseau ?")
        if not name:
            name = "RL.h5"

    savedModel = False
    if os.path.isdir(name):
        if os.path.isfile("{}/saved_model.pb".format(name)):
            savedModel = True
    else:
        if ".h5" not in name:
            name = "{}.h5".format(name)
        if os.path.isfile(name):
            savedModel = True

    return name, savedModel

agent_path, saved = pickName()
if saved :
    network = tf.keras.models.load_model(agent_path, compile = False, custom_objects={'Functional':tf.keras.models.Model})
    for i in range(10):
        watch_agent()
    import sys
    sys.exit(0)
else :
    network = keras.Sequential([
        keras.layers.Dense(256, input_shape=(state_size,), activation='relu'),# kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(num_actions, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3)


state = env.reset()
action = get_action(network, state, num_actions)
network.summary()
input("press a key")

num_episodes = 500
now = dt.datetime.now().strftime('%d%m%Y%H%M')
train_writer = tf.summary.create_file_writer("{}/{}".format(STORE_PATH, now))
score = []
for episode in range(num_episodes):
    state = env.reset()
    rewards = []
    states = []
    actions = []
    tot_reward = 0
    while True:
        action = get_action(network, state, num_actions)
        new_state, reward, done, _ = env.step(action)
        tot_reward += reward
        states.append(state)
        rewards.append(tot_reward)
        actions.append(action)
        if done:
            score.append(tot_reward)
            update_network(network, rewards, actions, states)
            if episode % 50 == 0 and episode>0:
                    print('Trajectory {}\tAverage Score: {:.2f}'.format(episode, np.mean(score[-50:-1])))
            if tb:
                with train_writer.as_default():
                    tf.summary.scalar('reward', tot_reward, step=episode)
            break
        state = new_state

def running_mean(x):
    N=50
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

score = np.array(score)
avg_score = running_mean(score)

plt.figure(figsize=(15,7))
plt.ylabel("Trajectory Duration",fontsize=12)
plt.xlabel("Training Epochs",fontsize=12)
plt.plot(score, color='gray' , linewidth=1)
plt.plot(avg_score, color='blue', linewidth=3)
plt.scatter(np.arange(score.shape[0]),score, color='green' , linewidth=0.3)
plt.show()

for i in range(10):
    watch_agent()
save = input("save ? Y=yes")
if save == "Y":
    network.save("{}/{}_ReinforceAgent_{}".format(DIR, game, now))

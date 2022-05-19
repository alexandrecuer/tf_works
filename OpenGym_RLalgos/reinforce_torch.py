import numpy as np
import torch
import gym
from matplotlib import pyplot as plt

import tensorflow as tf
import datetime as dt
game = "CartPole"
DIR =  "TensorBoard/PolicyGradient"
STORE_PATH = "{}/{}".format(DIR,game)

env = gym.make('CartPole-v0')
print('observation space:', env.observation_space)
print('action space:', env.action_space)
print('threshold: ', env.spec.reward_threshold)
reward_start = 0

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

HIDDEN_SIZE = 256

model = torch.nn.Sequential(
             torch.nn.Linear(obs_size, HIDDEN_SIZE),
             torch.nn.ReLU(),
             torch.nn.Linear(HIDDEN_SIZE, n_actions),
             torch.nn.Softmax(dim=0)
     )

print (model)

def watch_agent():
  state = env.reset()
  rewards = []
  for t in range(200):
    env.render()
    pred = model(torch.from_numpy(state).float())
    action = np.random.choice(np.array([0,1]), p=pred.data.numpy())
    state, reward, done, _ = env.step(action)
    rewards.append(reward)
    if done:
        print("Reward:", sum([r for r in rewards]))
        break
  #env.close()

#watch_agent()
#input("press")

learning_rate = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Horizon = 500
MAX_TRAJECTORIES = 500
gamma = 0.99
score = []

tb = False

if tb:
    now = dt.datetime.now().strftime('%d%m%Y%H%M')
    train_writer = tf.summary.create_file_writer("{}/{}".format(STORE_PATH, now))

for trajectory in range(MAX_TRAJECTORIES):
    curr_state = env.reset()
    done = False
    transitions = []
    tot_reward = reward_start

    for t in range(Horizon):
        act_prob = model(torch.from_numpy(curr_state).float())
        action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy())
        prev_state = curr_state
        """
        curr_state, _, done, info = env.step(action)
        transitions.append((prev_state, action, t+1))
        """
        curr_state, reward, done, _ = env.step(action)
        tot_reward += reward
        transitions.append((prev_state, action, tot_reward))
        if done:
            break

    score.append(tot_reward)

    reward_batch = [r for (s,a,r) in transitions]

    batch_Gvals =[]
    for i in range(len(transitions)):
        new_Gval=0
        power=0
        for j in range(i,len(transitions)):
             new_Gval=new_Gval+((gamma**power)*reward_batch[j])
             power+=1
        batch_Gvals.append(new_Gval)

    expected_returns_batch=torch.FloatTensor(batch_Gvals)

    expected_returns_batch /= expected_returns_batch.max()

    state_batch = torch.Tensor([s for (s,a,r) in transitions])
    action_batch = torch.Tensor([a for (s,a,r) in transitions])

    pred_batch = model(state_batch)
    #prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze()
    prob_batch = pred_batch.gather(dim=1,index=action_batch.long().unsqueeze(-1)).squeeze()
    loss = - torch.sum(torch.log(prob_batch) * expected_returns_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if tb:
        with train_writer.as_default():
            tf.summary.scalar('reward', tot_reward, step=trajectory)
            tf.summary.scalar('avg loss', loss.item(), step=trajectory)

    if trajectory % 50 == 0 and trajectory>0:
            print('Trajectory {}\tAverage Score: {:.2f}'.format(trajectory, np.mean(score[-50:-1])))

    # on réinitialise la récompense totale
    tot_reward = reward_start

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

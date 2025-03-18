import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import torch
from torch import nn
from DeepQLearning import DeepQLearning
import csv

env = gym.make('CartPole-v1')
#env.seed(0)
np.random.seed(0)

## pytorch model
model = nn.Sequential(
            nn.Linear(env.observation_space.shape[0],512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,env.action_space.n)
        )

criterion = nn.MSELoss()# MSE
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)# ADAM


params = {
    "env":env,
    "gamma" : 0.99,
    "epsilon" : 1.0,
    "epsilon_min" : 0.01,
    "epsilon_dec" : 0.99,
    "episodes" : 400,
    "batch_size" : 128,
    "memory" : deque(maxlen=10000), #talvez usar uma memoria mais curta
    "model" : model,
    "criterion" : criterion,
    "optimizer" : optimizer,
    "max_steps" : 500,
}

DQN = DeepQLearning(**params)
rewards = np.array(DQN.train())
avgs = []
for i in range(10,len(rewards)):
    avgs.append(rewards[i-10:i].mean())



import matplotlib.pyplot as plt
plt.plot(avgs)
plt.xlabel('Episodes')
plt.ylabel('# Rewards')
plt.title('# Average Reward vs Episodes')
plt.savefig("results/cartpole_DeepQLearning.jpg")     
plt.close()

with open('results/cartpole_DeepQLearning_rewards.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    episode=0
    for reward in rewards:
        writer.writerow([episode,reward])
        episode+=1

# model.save('data/model_cart_pole.keras')




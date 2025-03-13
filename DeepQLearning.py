import numpy as np
import random
# from keras.activations import relu, linear
import gc
# import keras
import torch

class DeepQLearning:

    #
    # Implementacao do algoritmo proposto em 
    # Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013
    # https://arxiv.org/abs/1312.5602
    #

    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, criterion,optimizer, max_steps):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = memory
        self.model = model
        self.criterion = criterion # loss function
        self.optimizer = optimizer # optimizer --> RESEARCH WHAT IT DOES LATER
        self.max_steps = max_steps
        self.device = torch.device("cuda")
        self.model.to(self.device)

    def convert(self, state):
        return torch.tensor(state, dtype=torch.float32).to(self.device)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        state_tensor = self.convert(state).unsqueeze(0)
        with torch.no_grad():
            action = self.model(state_tensor)
        return torch.argmax(action).item()

    # cria uma memoria longa de experiencias
    def experience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal)) 

    def experience_replay(self):
        # soh acontece o treinamento depois da memoria ser maior que o batch_size informado
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size) #escolha aleatoria dos exemplos
            # everything to tensors on gpu
            states = torch.tensor([i[0] for i in batch], dtype=torch.float32, device=self.device).squeeze(1)
            actions = torch.tensor([i[1] for i in batch], dtype=torch.long, device=self.device)
            rewards = torch.tensor([i[2] for i in batch], dtype=torch.float32, device=self.device)
            next_states = torch.tensor([i[3] for i in batch], dtype=torch.float32, device=self.device)
            terminals = torch.tensor([i[4] for i in batch], dtype=torch.float32, device=self.device)
            

            next_values = self.predict_on_batch(next_states)
            next_max = torch.max(next_values, dim=1).values


            targets = rewards + self.gamma * next_max * (1 - terminals)


            
            targets_full = self.predict_on_batch(states)
            indexes = np.array([i for i in range(self.batch_size)])

            targets_full[indexes, actions] = targets


            self.fit_model(states, targets_full)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec

    def predict_on_batch(self, states):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(states).squeeze(1)
        return predictions

    def fit_model(self, states, targets_full):
        self.model.train()
        
        self.optimizer.zero_grad()
        output = self.model(states)  # Forward pass
        loss = self.criterion(output, targets_full)  # Compute loss
        loss.backward()  # Backpropagate
        self.optimizer.step()  # Update weights

    def train(self):
        rewards = []
        for i in range(self.episodes+1):
            (state,_) = self.env.reset()
            state = np.reshape(state, (1, self.env.observation_space.shape[0]))
            score = 0
            steps = 0
            #
            # no caso do Cart Pole eh possivel usar a condicao de done do episodio
            #
            done = False
            while not done:
                steps += 1
                action = self.select_action(state)
                next_state, reward, terminal, truncated, _ = self.env.step(action)
                if terminal or truncated or (steps>self.max_steps):
                    done = True          
                score += reward
                next_state = np.reshape(next_state, (1, self.env.observation_space.shape[0]))
                self.experience(state, action, reward, next_state, terminal)
                state = next_state
                self.experience_replay()
                if done:
                    print(f'Episódio: {i+1}/{self.episodes}. Score: {score}')
                    break
            rewards.append(score)
            gc.collect()
            # keras.backend.clear_session()

        return rewards

import gymnasium as gym
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt



class Agent():
    def __init__(self,epsilon,e_dec,e_min,alpha,gamma,env,algo="q-learning",max_episodes=1000,max_ticks=500,training=False,human=False,cont=False): # iniital epsilon, e_dec, alpha gamma are not optional
        self.epsilon = epsilon
        self.e_dec = e_dec
        self.e_min = e_min
        self.alpha = alpha
        self.gamma = gamma
        if algo == "q-learning":
            self.algo = 0
        else: # currently sarsa
            self.algo = 1
        
        self.env = env
        self.training = training
        self.qdb = []
        self.max_episodes = max_episodes
        self.max_ticks = max_ticks
        self.cont = cont

        if self.cont: # mountain car continuous env requires multiple dimensions
            num_states = (self.env.observation_space.high - self.env.observation_space.low)*np.array([10, 100])
            num_states = np.round(num_states, 0).astype(int) + 1
            
            self.qdb = np.zeros([num_states[0], num_states[1], self.env.action_space.n])
            
        else:
            
            self.qdb = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        self.reward_hist = []


    def select_action(self,state):
        rand = random.randint(0,100)
        actions = self.qdb[state]

        ep = max(self.epsilon,self.e_min)
        if rand > ep*100 or not self.training:
            return np.argmax(actions)
        
        return self.env.action_space.sample()
    
    
    def transform_state(self, state):
        state_adj = (state - self.env.observation_space.low)*np.array([10, 100])
        return tuple(np.round(state_adj, 0).astype(int)) # needs to be a tuple in order to index np arrays
    
    def q_learning(self,state,action,next_state,reward): # state action q q learning
        self.qdb[state][action] = self.qdb[state][action] + self.alpha*(reward+self.gamma*max(self.qdb[next_state])-self.qdb[state][action])

    def sarsa(self,state,action,next_state,next_action,reward): # state action sarsa
        self.qdb[state][action] = self.qdb[state][action] + self.alpha*(reward+self.gamma*self.qdb[next_state][next_action]-self.qdb[state][action]) 


    def execute(self):
        m_e = self.max_episodes
        for ep in range(m_e):
            state, info  = self.env.reset()
            if self.cont:
                state = self.transform_state(state)
            action = self.select_action(state)
            ep_reward = 0
            for t in range(self.max_ticks):

                # sampling the action space
                next_state,reward,terminated,truncated,info = self.env.step(action)
                if self.cont:
                    next_state = self.transform_state(next_state)

                next_action = self.select_action(next_state)

                if self.training:
                    if self.algo == 0: ## q-learning
                        self.q_learning(state,action,next_state,reward)
                    else: ## sarsa
                        self.sarsa(state,action,next_state,next_action,reward)

                ep_reward += reward
                state = next_state
                action = next_action

                if terminated or truncated:
                    break

            self.epsilon *= self.e_dec
            next_state, info = self.env.reset()
            if self.cont:
                next_state = self.transform_state(next_state)
            self.reward_hist.append(ep_reward)

        if not self.training:
            self.env.close()
            
        return self.reward_hist


    
    def write_training(self, path):
        with open(path, "w+") as f:
            # Write shape information as first line
            shape_str = " ".join(str(dim) for dim in self.qdb.shape)
            f.write(f"SHAPE {shape_str}\n")
            
            # Flatten array and write as single line
            flat_qdb = self.qdb.flatten()
            qdb_line = " ".join(f"{value:.6f}" for value in flat_qdb)
            f.write(qdb_line + "\n")
            
            # Write rewards
            rewards_line = " ".join(f"{reward:.6f}" for reward in self.reward_hist)
            f.write("REWARDS " + rewards_line + "\n")

    def read_training(self, path, inplace=True, read_results=False):
        db = None
        rewards = []
        try:
            with open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                
                # Read shape information
                if not lines[0].startswith("SHAPE"):
                    raise ValueError("File format error: missing SHAPE information")
                
                shape_str = lines[0].replace("SHAPE", "").strip()
                shape = tuple(int(dim) for dim in shape_str.split())
                
                # Read flattened Q-values
                qdb_values = [float(x) for x in lines[1].split() if x.strip()]
                
                # Convert to numpy array and reshape
                import numpy as np
                db = np.array(qdb_values, dtype=float).reshape(shape)
                
                # Read rewards if requested
                if read_results:
                    rewards_idx = next((i for i, line in enumerate(lines) if line.startswith("REWARDS")), -1)
                    if rewards_idx != -1:
                        rewards_line = lines[rewards_idx].replace("REWARDS", "").strip()
                        rewards = [float(x) for x in rewards_line.split() if x.strip()]
        
        except Exception as e:
            print(f"Error reading training file: {e}")
            return None if not inplace else None
        
        if inplace:
            self.qdb = db
            if read_results:
                self.reward_hist = rewards
        else:
            return (db, rewards) if read_results else db
        
    def kill(self):
        self.env.close()


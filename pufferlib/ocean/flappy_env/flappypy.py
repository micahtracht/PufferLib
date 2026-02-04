import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import pufferlib
import pufferlib.vector

NOOP = 0
UP = 1
DOWN = 2

EMPTY = 0
AGENT = 1
WALL = 2
DANGER = 3

class FlappyBird(pufferlib.PufferEnv):
    def __init__(self, render_mode='ansi', buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(0, 3, (2,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(3)
        self.render_mode = render_mode
        self.num_agents = 1
        
        super().__init__(buf)
    
    def reset(self, seed=0):
        self.observations[0, :] = EMPTY
        self.observations[0, 0] = AGENT # initialize bird to first position
        self.h = 0 # height is 0. Larger height means further down.
        self.observations[0, 1] = WALL # initialize wall to other square
        self.tick = 0
        
        return self.observations, {}

    def step(self, actions):
        atn = actions[0] # get action for agent 0 (the only agent)
        
        self.terminals[0] = False
        self.rewards[0] = 0
        
        if atn == DOWN:
            self.h += 1
        elif atn == UP:
            self.h -= 1
        
        info = []
        if (self.h < 0 or self.h > 1 
            or self.observations[0, self.h] == WALL
            or self.observations[0, self.h] == DANGER):
            self.terminals[0] = True
            self.rewards[0] = -1.0
            info = [{'reward': -1.0}]
            self.reset()
        elif self.tick > 1000:
            self.terminals[0] = True # max episode len is 1000
            self.rewards[0] = 1.0
            info = [{'reward': 1.0}]
            self.reset()
        else:
            self.observations[0, :] = EMPTY
            self.observations[0, self.h] = AGENT
            self.tick += 1
        
        random = np.random.randint(0, 2)
        if self.observations[0, random] == AGENT:
            self.observations[0, random] = DANGER
        else:
            self.observations[0, random] = WALL
        
        return self.observations, self.rewards, self.terminals, self.truncations, info
    
    def render(self):
        for i in range(2):
            if self.observations[0, i] == AGENT:
                print('A', end=" ")
            if self.observations[0, i] == WALL:
                print('W', end=" ")
        print()
    
    def close(self): # no idea what this does, I just copied the code.
        pass

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(int(np.prod(envs.single_observation_space.shape)), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, envs.single_action_space.n)
        )


def collect_episode(agent, env):
    log_probs = []
    rewards = []
    obs, _ = env.reset()
    done = False
    while not done: # revise condition
        logits = agent.actor(torch.tensor(obs, dtype=torch.float32))
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        
        obs, reward, terminals, _, _ = env.step(action.numpy())
        rewards.append(reward[0])
        
        done = terminals[0]
    
    return log_probs, rewards

def calculate_returns(rewards, gamma=0.99):
    returns = [0] * len(rewards)
    
    curr_ret = 0
    idx = len(rewards) - 1
    for r in reversed(rewards):
        curr_ret *= gamma # multiply everything before by gamma (apply discount)
        curr_ret += r # add current reward
        returns[idx] = curr_ret # direct assignment to avoid creating extra space when reversing
        idx -= 1
    
    return returns

def calculate_loss(log_probs, returns):
    loss = 0
    for i in range(len(returns)):
        loss += log_probs[i] * torch.tensor(returns[i], dtype=torch.float32) # don't call .item() on log_probs, will detach it from compute graph -> needed for backprop
    return -loss

def main():
    learning_rate = 2.5e-4
    num_envs = 1
    num_episodes = 5000
    gamma=0.99
    
    # PufferLib Vectorization
    envs = pufferlib.vector.make(
        FlappyBird,
        num_envs=num_envs,
        backend=pufferlib.vector.Serial
    )
    
    agent = Agent(envs)
    opt = torch.optim.AdamW(agent.parameters(), lr=learning_rate)
    running_reward = 0
    for i in range(num_episodes):
        log_probs, rewards = collect_episode(agent, envs)
        returns = calculate_returns(rewards, gamma=gamma)
        loss = calculate_loss(log_probs, returns)
        
        total_reward = sum(rewards)
        running_reward = 0.05 * total_reward + 0.95 * running_reward
        if i % 20 == 0:
            print(i, running_reward)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    


    envs.close()
    
if __name__ == "__main__":
    main()
    
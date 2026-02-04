'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.target import binding

class Target(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=1080, height=720, num_agents=8,
            num_goals=4, render_mode=None, log_interval=128, size=11, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(2*(num_agents+num_goals) + 4,), dtype=np.float32) # set observation space to have low of 0, high of 1, and 2 * number of agents and goals plus four possible entires. Needs 2 observations for each agent (x and y delta) and goal, as well as for heading, rewards, and it's own x and y
        self.single_action_space = gymnasium.spaces.MultiDiscrete([9, 5]) # can pick actions in a discrete space. First action is in [0, 8] (9 possible), second is in [0, 4] (5 possible)

        self.render_mode = render_mode # set other info
        self.num_agents = num_envs*num_agents
        self.log_interval = log_interval

        super().__init__(buf) # init the shared buffer object
        c_envs = []
        for i in range(num_envs): # for each environemnt we:
            c_env = binding.env_init( # initialize the environment
                self.observations[i*num_agents:(i+1)*num_agents], # pass view of giant observation block of memory to the c function for it to work with
                self.actions[i*num_agents:(i+1)*num_agents], # do the same for actions
                self.rewards[i*num_agents:(i+1)*num_agents], # and rewrads
                self.terminals[i*num_agents:(i+1)*num_agents], # and terminals
                self.truncations[i*num_agents:(i+1)*num_agents], # and truncations
                seed, width=width, height=height, # set the seed, width, and height
                num_agents=num_agents, num_goals=num_goals) # set num agetns and goals
            c_envs.append(c_env)

        self.c_envs = binding.vectorize(*c_envs) # required in all pufferlib: set c_envs to the vectorized binding.

    def reset(self, seed=0): # just use vec_reset, reset the tick, and return whatever was written to observations
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs) # increment tick, get actions, then call vec_step to actually increment the simulation

        info = []
        if self.tick % self.log_interval == 0: # if it's time to log info, then log it.
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0) # simple wrapper

    def close(self):
        binding.vec_close(self.c_envs) # simple wrapper

if __name__ == '__main__': # test SPS
    N = 512

    env = Target(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(env.single_action_space.nvec, size=(CACHE, 2))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_agents
        i += 1

    print('Target SPS:', int(steps / (time.time() - start)))

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.shooter import binding

class Shooter(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=1080, height=720, num_goals=4, 
            render_mode=None, log_interval=100, buf=None, seed=0): # maybe issue w how num envs are initialized? Maybe should be num agents? Probs not bc not multiagent
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        
        self.render_mode = render_mode
        self.num_agents = num_envs # tells thing how many actions to sample and observation pairs there are
        '''
        Key part of library here:
        num_agents is how many actions we need to sample.
        We simply feed them the observations, which are handled quite cleanly by our C code.
        In multiagent environments, it'll always work out because we'll write the right observations for each agent.
        If there's a mismatch, C will probably through a memory error, making it easy to debug.
        '''
        self.log_interval = log_interval
        
        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                # ASSUME: 1 agent per env
                self.observations[i:i+1],
                self.actions[i:i+1],
                self.rewards[i:i+1],
                self.terminals[i:i+1],
                self.truncations[i:i+1],
                seed+i, width=width, height=height,
                num_agents=1
            )
            c_envs.append(c_env)
        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        
        info = []
        if self.terminals.any():
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)
        
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0) # maybe do not include 0? Unsure, params might mismatch.
    
    def close(self):
        binding.vec_close(self.c_envs)
    
if __name__ == '__main__': # test SPS
    N = 512

    env = Shooter(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(env.single_action_space.n, size=(CACHE, N))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_agents
        i += 1

    print('Shooter SPS:', int(steps / (time.time() - start)))


    
    
    

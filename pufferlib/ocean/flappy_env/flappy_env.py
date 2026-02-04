import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.flappy_env import binding

class Flappy(pufferlib.PufferEnv):
    def __init__(self, render_mode='ansi', buf=None, log_interval=128, num_envs = 1, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(0, 3, (2,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(3)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.num_envs = num_envs
        self.log_interval = log_interval
        
        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards, 
            self.terminals, self.truncations, self.num_envs, seed)
    
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []
    
    def step(self, actions):
        self.tick += 1
        
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        
        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))
        
        return (self.observations, self.rewards, 
            self.terminals, self.truncations, info)
    
    def render(self):
        binding.vec_render(self.c_envs, 0)
    
    def close(self):
        binding.vec_close(self.c_envs)
    

if __name__ == "__main__":
    N = 2
    
    env = Flappy(num_envs=N)
    env.reset()
    steps = 0
    
    CACHE = 1024
    actions = np.random.randint(0, 3, (CACHE, N))
    
    i = 0
    import time
    start = time.perf_counter()
    while time.perf_counter() - start < 10:
        env.step(actions[i % CACHE])
        steps += 1
        i += 1
    
    print(f'sps: {(steps / (time.perf_counter() - start)):.3f}')
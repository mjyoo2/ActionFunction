import gym


class WrappedEnvClass(gym.Env):
    def __init__(self, wrappedEnv, num_seq):
        self.wrapped_env = wrappedEnv
        self.num_seq = num_seq
        self.action_space = (num_seq, ) + self.wrapped_env.action_space.shape
        self.observation_space = wrappedEnv.observation_space

    def reset(self):
        return self.wrapped_env.reset()

    def step(self, action):
        r_cnt = 0
        for a in action:
            s, r, d, i = self.wrapped_env.step(a)
            r_cnt += r
            if d:
                return s, r_cnt, d, i

    def render(self, mode='human'):
        self.wrapped_env.render(mode)


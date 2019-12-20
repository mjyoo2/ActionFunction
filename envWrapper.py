import gym
import numpy as np


class WrappedEnvClass(gym.Env):
    """
    Wrap an environment for action as a sequential structure
    The environment receives sequence of action and perform stepwise. i.e., n-step planning
    """
    def __init__(self, wrappedEnv, num_seq):
        """
        :param wrappedEnv: the environment which will be wrapped
        :param num_seq: the number of sequence to perform steps. it may be tuple type with length 2 or just integer
        if num_seq is tuple with length 2, the sequence is variable length with lower bound, and upper bound
        else num_seq is static integer
        """
        self.wrapped_env = wrappedEnv
        if type(num_seq) is tuple:
            assert len(num_seq) == 2
            self._num_seq = lambda: np.random.randint(low=num_seq[0], high=num_seq[1])
        else:
            self._num_seq = lambda: num_seq
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

    @property
    def num_seq(self):
        return self._num_seq()

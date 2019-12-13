import envWrapper
import numpy as np
import gym


class FunctionEnv(envWrapper.WrappedEnvClass):
    def __init__(self, wrappedEnv, num_seq, neuro_structure):
        super().__init__(wrappedEnv, num_seq)
        assert type(neuro_structure) is tuple
        self.neuro_structure = neuro_structure
        a = 1
        for i in range(len(self.neuro_structure)):
            a *= self.neuro_structure[i]
        self.action_space = gym.spaces.Box(low=-3 ,high=3, shape=(a, ))
        self.last_state = None
        self.step_cnt = 0

    def reset(self):
        self.last_state = super().reset()
        return np.copy(self.last_state)

    def step(self, action):

        temp_neuron = action.reshape(self.neuro_structure)
        r_cnt = 0
        done = False
        info = {}
        self.step_cnt += 1

        for _ in range(self.num_seq):
            a = self.run_neuro(self.last_state, temp_neuron)
            s, r, done, info = self.wrapped_env.step(a)
            self.last_state = s
            r_cnt += r
            if done:
                return s, r_cnt, done, info


        return self.last_state, r_cnt, done , info

    def run_neuro(self, state, neuron):
        temp_action = np.matmul(state, neuron)
        np.tanh(temp_action)
        return np.argmax(temp_action)


import envWrapper
import numpy as np
import gym


class FunctionEnv(envWrapper.WrappedEnvClass):
    def __init__(self, wrappedEnv, num_seq, neuro_structure, ):
        super().__init__(wrappedEnv, num_seq)
        assert type(neuro_structure) is tuple
        self.neuro_structure = self.parse_neuro_structure(neuro_structure)
        self.partition_table = self.build_action_partion_table()
        a = self.partition_table[-1]
        self.action_space = gym.spaces.Box(low=-3 ,high=3, shape=(a, ))
        self.last_state = None
        self.step_cnt = 0

    def reset(self):
        self.last_state = super().reset()
        return np.copy(self.last_state)

    def step(self, action):
        action = np.sinh(action)
        temp_neuron = self.build_neurons(action)
        r_cnt = 0
        done = False
        info = {}
        self.step_cnt += 1
        seq = self.num_seq
        for _ in range(seq):
            a = self.run_neuro_discrete(self.last_state, temp_neuron)
            s, r, done, info = self.wrapped_env.step(a)
            self.last_state = s
            r_cnt += r
            if done:
                return s, r_cnt, done, info
        return self.last_state, r_cnt, done , info

    @staticmethod
    def run_neuro_discrete(state, neurons):
        x = np.copy(state)
        for n in range(len(neurons) - 1):
            layer = neurons[n]
            x = np.matmul(x, layer)
            x = np.maximum(x, 0, x)
        x = np.matmul(x, neurons[-1])
        x = np.tanh(x)
        return np.argmax(x)

    def parse_neuro_structure(self, action_structure) -> list:
        assert len(action_structure) >= 2
        if len(action_structure) == 2:
            return [action_structure]
        else:
            shapes = []
            for i in range(len(action_structure) - 1):
                shapes.append((action_structure[i], action_structure[i + 1]))
            return shapes

    def build_action_partion_table(self):
        cnt = 0
        table = [0]
        for sh in self.neuro_structure:
            size = sh[0] * sh[1]
            cnt += size
            table.append(cnt)
        return table

    def build_neurons(self, action):
        neurons = []
        for i in range(len(self.partition_table) - 1):
            a = action[self.partition_table[i]: self.partition_table[i + 1]]
            a = a.reshape(self.neuro_structure[i])
            neurons.append(a)
        return neurons




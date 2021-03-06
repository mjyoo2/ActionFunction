import envWrapper
import numpy as np
import tensorflow as tf
from stable_baselines.common.distributions import make_proba_dist_type
from stable_baselines.common import tf_util
import gym

class replaybuffer():
    def __init__(self, maxlen):
        self.data = []
        self.maxlen = maxlen


    def len(self):
        return len(self.data)


    def add(self, new_data):
        self.data.append(new_data)
        reward = new_data['reward']
        gamma = 0.99
        for i in range(self.len() - 1, 0, -1):
            self.data[i]['reward'] = reward * gamma
            gamma = gamma * 0.99
        if self.len() > self.maxlen:
            del self.data[0]
        return


    def clear(self):
        self.data = []


    def get_data(self):
        return self.data


class FunctionEnv(envWrapper.WrappedEnvClass):
    def __init__(self, wrappedEnv, num_seq, neuro_structure, ):
        super().__init__(wrappedEnv, num_seq)
        assert type(neuro_structure) is tuple
        self.sess = tf.Session()
        self.neuro_structure = self.parse_neuro_structure(neuro_structure)
        self.partition_table = self.build_action_partion_table()
        a = self.partition_table[-1]
        self.action_space = gym.spaces.Box(low=-3 ,high=3, shape=(a, ))
        self.last_state = None
        self.step_cnt = 0
        self.replay_buffer = replaybuffer(maxlen=512)

        self._pdtype = make_proba_dist_type(self.action_space)
        self._proba_distribution = None
        self.action_ph = None
        self._policy_proba = None
        self.pg_loss = None
        self.params = None
        self.obs = tf.placeholder(tf.float32, shape=(None, 2))

        self.policy = self.init_network_continuous(self.obs, 'net')
        self.sess.run(tf.global_variables_initializer())


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
            self.replay_buffer.add({'state': s, 'reward': r, 'done': done, 'action': a})
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

    def init_network_continuous(self, input, name):
        with tf.variable_scope(name):
            model = tf.layers.dense(input, 8, activation=tf.nn.relu)
            model = tf.layers.dense(model, self.action_space.shape[0], activation=tf.nn.sigmoid)

        self._proba_distribution, _, _ = \
            self._pdtype.proba_distribution_from_latent(model, model, init_scale=0.01)

        self.action_ph = self._pdtype.sample_placeholder([None], name='action_ph')
        self._policy_proba = [self._proba_distribution.mean, self._proba_distribution.std]
        self.params = tf_util.get_trainable_vars('net')
        self.pg_loss = tf.gradients(self._proba_distribution.neglogp(self.action_ph), self.params)
        return model

    def predict(self, observation):
        action = self.sess.run([self.policy],{self.obs: observation})
        return action

    def get_gradeints(self):
        gradient_set = []
        dataset = self.replay_buffer.get_data()
        for replay in dataset:
            G = replay['reward']
            gradient = self.sess.run([self.pg_loss], feed_dict={self.obs: replay['state'], self.action_ph: replay['action']})
            gradient_set.append(gradient * G)
        return gradient_set

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



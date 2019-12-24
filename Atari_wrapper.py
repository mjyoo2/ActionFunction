import gym
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

class AtariWrapEnv(gym.Env):
    def __init__(self, game_id, frame_skip = 4, render_mode=True):
        self.game_id = game_id
        self.atari = gym.make(game_id, obs_type='image', frameskip=1)
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.real_state = []
        self.action_space = self.atari.action_space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84, self.frame_skip))

        self.num_timesteps = 0
        self.sum_reward = 0

    def step(self, action):
        done = False
        reward = 0
        self.real_state = []

        for _ in range(self.frame_skip):
            if done:
                self.real_state.append(np.zeros((84, 84)))
                continue
            state, temp_reward, done, _ = self.atari.step(action)
            state = resize(rgb2gray(state), (84, 84))
            self.real_state.append(state)
            reward += temp_reward
            if self.render_mode:
                self.render()

        output = self.real_state[0]
        for i in range(self.frame_skip - 1):
            output = (np.concatenate([output, self.real_state[i + 1]], axis=2))

        self.sum_reward += reward
        self.num_timesteps += 1

        if done:
            info = {'episode': {'r': self.sum_reward, 'l': self.num_timesteps}, 'game_reward': reward}
        else:
            info = {'episode': None, 'game_reward': reward}

        return output, reward, done, info


    def reset(self):
        for i in range(self.frame_skip - 1):
            self.real_state.append(np.zeros((84, 84)))
        state = self.atari.reset()
        state = resize(rgb2gray(state), (84, 84))
        self.real_state.append(state)

        self.sum_reward = 0
        self.num_timesteps = 0

        output = self.real_state[0]
        for i in range(self.frame_skip - 1):
            output = (np.concatenate([output, self.real_state[i + 1]], axis=2))

        return output


    def render(self):
        self.atari.render()
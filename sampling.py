from Atari_wrapper import AtariWrapEnv
from stable_baselines import DQN
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

if __name__ == '__main__':
    env = AtariWrapEnv(sampling_mode=True)
    model = DQN(CnnPolicy, env)
    while True:
        done = False
        state = env.reset()
        while not done:
            action, _ = model.predict(state)
            state, _, done, info = env.step(action)
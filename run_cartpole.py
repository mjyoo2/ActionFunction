from  FunctionWrappedEnv import FunctionEnv
import gym
from stable_baselines import SAC
from stable_baselines.sac.policies import LnMlpPolicy

if __name__ == "__main__":
    cartpole = gym.make("CartPole-v1")
    print(cartpole.observation_space)
    neuro_structure=  (4, 2)
    env = FunctionEnv(cartpole, 5, neuro_structure)
    model = SAC(LnMlpPolicy, env, verbose=0, full_tensorboard_log=True,
                tensorboard_log="D:\논문\ActionLearningTensorBoard")
    model.learn(1000000)

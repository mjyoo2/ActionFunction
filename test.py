from Atari_wrapper import AtariWrapEnv
import gym

if __name__ =='__main__':
    env = gym.make('SpaceInvaders-v0')
    env.reset()
    for _ in range(1000): env.step(env.action_space.sample())
    env.render('human')
    env.close()

    # env = AtariWrapEnv('Breakout-v0')
    # state = env.reset()


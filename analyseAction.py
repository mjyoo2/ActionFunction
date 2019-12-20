import numpy as np

if __name__ == "__main__":
    actions = np.load("action.npy")
    obs = np.load("obs.npy")

    for i in range(len(obs) - 1):
        obs_norm = np.linalg.norm(obs[i+ 1] - obs[i])
        action_norm = np.linalg.norm(actions[i+1] - actions[i])
        print("observation norm", obs_norm)
        print("action norm", action_norm)
        print("diff", action_norm/obs_norm)
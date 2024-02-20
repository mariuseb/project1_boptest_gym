from boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper, NormalizedObservationWrapper, DiscretizedActionWrapper, DiscretizedObservationWrapper
from stable_baselines3 import A2C, DQN
from examples.test_and_plot import test_agent
from copy import deepcopy
import gymnasium as gym
import torch

# BOPTEST case address
#url = 'http://127.0.0.1:5001'
url = 'http://bacssaas_boptest:5000'

# Instantite environment
#env = gym.make("MountainCar-v0")
env = gym.make("CartPole-v0")

# Add wrappers to normalize state and action spaces (Optional)
#env = NormalizedObservationWrapper(env)
#env = NormalizedActionWrapper(env)  
#env = DiscretizedActionWrapper(env, n_bins_act=200)
#env = DiscretizedObservationWrapper(env, n_bins_obs=100)

# Instantiate and train an RL algorithm

#model = A2C('MlpPolicy', env, verbose=1)


model = DQN('MlpPolicy',
            env,
            learning_starts=1,
            exploration_fraction=0.1,
            learning_rate=1E-3,
            verbose=1)


"""
model = DQN('MlpPolicy',
            env,
            verbose=1,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.07,
            target_update_interval=10,
            max_grad_norm=100,
            learning_starts=1,
            buffer_size=1000,
            batch_size=1,
            learning_rate=4e-4,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=2
            )
"""

#model.learn(total_timesteps=int(1e5))
#model.learn(total_timesteps=int(1000))
#model.learn(total_timesteps=int(1000))
before = deepcopy(model.get_parameters()["policy"])

model.learn(total_timesteps=int(1E6)) # progress_bar=True)
model.save("dqn_testcase1_mountain_car")

after = model.get_parameters()["policy"]
for k, v in after.items():
    print("After: ")
    print(v)
    print("Before: ")
    print(before[k])
print(model)
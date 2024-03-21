from boptestGymEnv import BoptestGymEnv, \
 			  NormalizedActionWrapper, \
 			  NormalizedObservationWrapper, \
 			  DiscretizedActionWrapper, \
 			  DiscretizedObservationWrapper
from stable_baselines3 import A2C, DQN
from examples.test_and_plot import test_agent
from copy import deepcopy
import torch

# BOPTEST case address
#url = 'http://127.0.0.1:5001'
url = 'http://bacssaas_boptest:5000'

# Instantiate environment
env = BoptestGymEnv(None,
                    url                   = url,
                    actions               = ['oveAct_u'],
                    observations          = {'TRooAir_y':(280.,310.)}, 
                    #observations          = {'TRooAir_y':(280.,310.), 'TDryBul':(280.,310.)}, 
                    random_start_time     = True,
                    max_episode_length    = 24*3600,
                    predictive_period     = 3600,
                    warmup_period         = 0,
                    step_period           = 900)

# Add wrappers to normalize state and action spaces (Optional)
env = NormalizedObservationWrapper(env)
env = NormalizedActionWrapper(env)  
env = DiscretizedActionWrapper(env, n_bins_act=200)
#env = DiscretizedObservationWrapper(env, n_bins_obs=100)
env.reset()
# Instantiate and train an RL algorithm

#model = A2C('MlpPolicy', env, verbose=1)
model = DQN('MlpPolicy',
            env,
            exploration_initial_eps=0.3,
            learning_starts=1,
            exploration_fraction=0.25,
            learning_rate=5E-4,
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

model.learn(total_timesteps=int(8760*4)) # progress_bar=True)
#model = DQN.load("dqn_testcase1")
model.save("dqn_testcase1")
# Test trained agent
observations, actions, rewards, kpis = test_agent(env, model, 
                                                  start_time=0, 
                                                  episode_length=7*24*3600,
                                                  #episode_length=4*900,
                                                  #episode_length=1,
                                                  warmup_period=0,
                                                  plot=True)
after = model.get_parameters()["policy"]


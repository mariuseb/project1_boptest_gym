from boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper, NormalizedObservationWrapper
from stable_baselines3 import A2C, DQN
from examples.test_and_plot import test_agent

# BOPTEST case address
#url = 'http://127.0.0.1:5001'
url = 'http://bacssaas_boptest:5000'

# Instantite environment
env = BoptestGymEnv(url                   = url,
                    actions               = ['oveHeaPumY_u'],
                    observations          = {'reaTZon_y':(280.,310.)}, 
                    random_start_time     = True,
                    max_episode_length    = 24*3600,
                    warmup_period         = 24*3600,
                    step_period           = 900)

# Add wrappers to normalize state and action spaces (Optional)
env = NormalizedObservationWrapper(env)
env = NormalizedActionWrapper(env)  

# Instantiate and train an RL algorithm
model = A2C('MlpPolicy', env, verbose=1)
#model = A2C('MlpPolicy', env, verbose=1)
#model.learn(total_timesteps=int(1e5))
#model.learn(total_timesteps=int(1000))
#model.learn(total_timesteps=int(1000))
model.learn(total_timesteps=int(1e4))

# Test trained agent
observations, actions, rewards, kpis = test_agent(env, model, 
                                                  start_time=0, 
                                                  episode_length=7*24*3600,
                                                  #episode_length=1,
                                                  warmup_period=24*3600,
                                                  plot=True)
print(env)
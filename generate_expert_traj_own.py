'''
Created on May 13, 2021

@author: Javier Arroyo

Generates an expert trajectory for pretraining.

'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from project1_boptest_gym.boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper, NormalizedObservationWrapper, DiscretizedActionWrapper, DiscretizedObservationWrapper
#from boptestGymEnv import DiscretizedActionWrapper
# from stable_baselines3.gail.dataset.record_expert import generate_expert_traj
from stable_baselines3 import A2C, DQN
from gymnasium.core import Wrapper
from project1_boptest_gym.examples import train_RL

class ExpertModelCont(A2C):
    '''Simple proportional controller for this emulator that works as an 
    expert to pretrain the RL algorithms. Case with continuous actions. 
    The generated expert dataset works e.g. with A2C and SAC.   
    
    '''
    def __init__(self, env, TSet=22+273.15, k=1):
        super(ExpertModelCont, self).__init__(env=env,policy='MlpPolicy')

        self.env   = env
        self.TSet  = TSet
        self.k     = k 
    
    def predict(
                self,
                obs,
                state=None,
                episode_start=None,
                deterministic=True
                ):
        #self.env.envs[0]
        #self.env.envs[0].measurement_vars
        # Find index
        #i_obs = self.env.envs[0].observations.index('reaTZon_y')
        i_obs = self.env.observations.index('TRooAir_y')
        # Rescale
        l = self.env.lower_obs_bounds[i_obs]
        u = self.env.upper_obs_bounds[i_obs]
        TZon = l + ((1+obs[0][i_obs])*(u-l)/2)
        # Compute control between -1 and 1 since env is normalized
        return np.array([min(1,max(-1,self.k*(self.TSet-TZon)))]), None
    
    def get_env(self):
        return self.env.envs[0]

class ExpertModelDisc(DQN):
    '''Simple proportional controller for this emulator that works as an 
    expert to pretrain the RL algorithms. The generated expert dataset 
    works e,g, with DQN. 
    
    '''
    def __init__(self, env, 
                 n_bins_act = 10, 
                 TSet=22+273.15, k=1):
        #self.env        = DiscretizedActionWrapper(env,n_bins_act=n_bins_act)
        self.env = env
        self.n_bins_act = n_bins_act
        self.TSet       = TSet
        self.k          = k 
        self.act_vals   = np.arange(n_bins_act+1)
    
    #def predict(self, obs, deterministic=True):
    def predict(
                self,
                obs,
                state=None,
                episode_start=None,
                deterministic=True
                ):
        """
        self.env
        self.env.measurement_vars
        # Find index
        #i_obs = self.env.observations.index('reaTZon_y')
        i_obs = self.env.observations.index('TRooAir_y')
        # Rescale
        l = self.env.lower_obs_bounds[i_obs]
        u = self.env.upper_obs_bounds[i_obs]
        TZon = l + ((1+obs[i_obs])*(u-l)/2)
        """
        #obs = state
        i_obs = self.env.observations.index('TRooAir_y')
        l = self.env.lower_obs_bounds[i_obs]
        u = self.env.upper_obs_bounds[i_obs]
        TZon = obs[i_obs]
        TZon = l + ((1 + obs[i_obs])*(u-l)/2)
        #TZon = (obs[i_obs] - l)/(u-l)
        # Compute control 
        value = self.k*(self.TSet-TZon)
        # Transform from [-1,1] to [0,10] since env is discretized
        #value = 5*value + 5
        #bias = int(self.n_bins_act/2) # is two-sided or not?
        #scaling = self.k/bias
        #value = int(value/bias + bias)
        # Bound result
        #value = min(self.n_bins_act, max(0,value))
        
        value = self.normalize_action(value)
        return np.array([self.find_nearest_action(value)]), None
    
    def normalize_action(self, value):
        bias = int(self.n_bins_act/2) # is two-sided or not?
        #value = int(int(value/bias) + bias)
        value = int(round(value/bias + bias, 0))
        # Bound result
        return min(self.n_bins_act, max(0,value))
        
    def get_env(self):
        return self.env
    
    def find_nearest_action(self, value):
        idx = (np.abs(self.act_vals - value)).argmin()
        return self.act_vals[idx]


if __name__ == "__main__":
    #n_days = 0.1
    cont_disc = 'cont'
    #cont_disc = 'disc'
    url = 'http://bacssaas_boptest:5000'
    env = BoptestGymEnv(url                   = url,
                    actions               = ['oveAct_u'],
                    observations          = {'TRooAir_y':(280.,310.)}, 
                    #observations          = {'TRooAir_y':(280.,310.), 'TDryBul':(280.,310.)}, 
                    random_start_time     = False,
                    max_episode_length    = 24*3600,
                    predictive_period     = 3600,
                    warmup_period         = 0,
                    step_period           = 900)
    env.reset()
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)  
    env = DiscretizedActionWrapper(env, n_bins_act=200)
    
    # Set expert trajectory to start the first day of February    
    start_year      = '2021-01-01 00:00:00'
    start           = '2021-02-01 00:00:00'
    start_time      = (pd.Timestamp(start)-pd.Timestamp(start_year)).total_seconds()    
    if isinstance(env,Wrapper): 
        env.unwrapped.random_start_time   = False
        env.unwrapped.start_time          = start_time
    else:
        env.random_start_time   = False
        env.start_time          = start_time
    
    # Instantiate expert model. Distinguish between continuous or discrete
    if cont_disc == 'cont':
        expert_model = ExpertModelCont(env)
    elif cont_disc == 'disc':
        expert_model = ExpertModelDisc(env)
    
    # Generate data and save in a numpy archive named `expert_traj.npz`
    print('Generating expert data...')
    from imitation.data import rollout
    from imitation.data.wrappers import RolloutInfoWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv

    
    """
    model = DQN('MlpPolicy', env, verbose=1, gamma=0.99,
                learning_rate=5e-4, batch_size=24, 
                buffer_size=365*24, learning_starts=24, train_freq=1)
                #tensorboard_log=log_dir)   
    """
    model = A2C('MlpPolicy', env, verbose=1, gamma=0.99, seed=123456,
                learning_rate=7e-4, n_steps=4, ent_coef=1)
                #tensorboard_log=log_dir)
                
    expert_model.observation_space = model.observation_space
    expert_model.action_space = model.action_space
    rollouts = rollout.generate_trajectories(
        policy=expert_model,
        venv=DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        sample_until=rollout.make_sample_until(None, 1),
        rng=np.random.default_rng(seed=42)
    )
    transitions = rollout.flatten_trajectories(rollouts)

    print(
        f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
    After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
    The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
    """
    )

    print('implementing behavior cloning...')
    from imitation.algorithms import bc
    rng = np.random.default_rng(0)
    bc_trainer = bc.BC(
        #policy=model,
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng
    )

    from stable_baselines3.common.evaluation import evaluate_policy

    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 1)
    print(f"Reward before training: {reward_before_training}")

    bc_trainer.train(n_epochs=10)
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 1)
    print(f"Reward after training: {reward_after_training}")

    traj_name = os.path.join('trajectories',
                             'expert_traj_{}_{}'.format(cont_disc, 1.0))
    print(traj_name)
    # generate_expert_traj(expert_model,
    #                      traj_name,
    #                      n_episodes=1)
    #

    # plt.savefig(traj_name+'.pdf', bbox_inches='tight')
    
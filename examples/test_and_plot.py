'''
Common functionality to test and plot an agent

'''

import matplotlib.pyplot as plt
from scipy import interpolate
from gymnasium.core import Wrapper
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
import json
import os


def test_agent(env, model, start_time, episode_length, warmup_period,
               log_dir=os.getcwd(), model_name='last_model', 
               save_to_file=False, plot=False):
    ''' Test model agent in env.
    
    '''
        
    # Set a fixed start time
    if isinstance(env,Wrapper): 
        env.unwrapped.random_start_time   = False
        env.unwrapped.start_time          = start_time
        env.unwrapped.max_episode_length  = episode_length
        env.unwrapped.warmup_period       = warmup_period
    else:
        env.random_start_time   = False
        env.start_time          = start_time
        env.max_episode_length  = episode_length
        env.warmup_period       = warmup_period
    
    # Reset environment
<<<<<<< HEAD
    obs, _ = env.reset()
=======
    obs, info = env.reset()
>>>>>>> 99da257 (test)
    
    # Simulation loop
    done = False
    observations = [obs]
    actions = []
    rewards = []
    print('Simulating...')
<<<<<<< HEAD
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
=======
    #while done is False:
    for i in range(int(episode_length/900)):
        #action, _ = model.predict(obs, deterministic=True)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
>>>>>>> 99da257 (test)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        done = (terminated or truncated)

    kpis = env.get_kpis()
    
    if save_to_file:
        os.makedirs(os.path.join(log_dir, 'results_tests_'+model_name+'_'+env.scenario['electricity_price']), exist_ok=True)
        with open(os.path.join(log_dir, 'results_tests_'+model_name+'_'+env.scenario['electricity_price'], 'kpis_{}.json'.format(str(int(start_time/3600/24)))), 'w') as f:
            json.dump(kpis, f)
    
    if plot:
        plot_results(env, rewards, save_to_file=save_to_file, log_dir=log_dir, model_name=model_name)
    
    # Back to random start time, just in case we're testing in the loop
    if isinstance(env,Wrapper): 
        env.unwrapped.random_start_time = True
    else:
        env.random_start_time = True
    
    return observations, actions, rewards, kpis

<<<<<<< HEAD
def plot_results(env, rewards, points=['reaTZon_y','reaTSetHea_y','reaTSetCoo_y','oveHeaPumY_u',
                                       'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'],
                 log_dir=os.getcwd(), model_name='last_model', save_to_file=False):
    

=======
def plot_results(env, rewards, points=None,
                 log_dir=os.getcwd(), model_name='last_model', save_to_file=False):
    
    """
    TODO:
        - modularize plotting
    """
    
    df_res = pd.DataFrame()
>>>>>>> 99da257 (test)
    if points is None:
        points = list(env.all_measurement_vars.keys()) + \
                 list(env.all_input_vars.keys())
        
<<<<<<< HEAD
    # Retrieve all simulation data
    # We use env.start_time+1 to ensure that we don't return the last 
    # point from the initialization period to don't confuse it with 
    # actions taken by the agent in a previous episode. 
    res = requests.put('{0}/results'.format(env.url), 
                        json={'point_names':points,
                              'start_time':env.start_time+1, 
                              'final_time':3.1536e7}).json()['payload']

    df = pd.DataFrame(res)
    df = create_datetime_index(df)
=======
    #for point in points:
        # Retrieve all simulation data
        # We use env.start_time+1 to ensure that we don't return the last 
        # point from the initialization period to don't confuse it with 
        # actions taken by the agent
    res = requests.put('{0}/results'.format(env.url), 
                        json={'point_names':points,
                                'start_time':env.start_time, 
                                'final_time':env.start_time + 3.1536e6}).json()["payload"]
    """
    df_res = pd.concat((df_res,pd.DataFrame(data=res[point], 
                                            index=res['time'],
                                            columns=[point])), axis=1)
    """
    df_res = pd.DataFrame.from_dict(res)
    df_res.index.name = 'time'
    #df_res.reset_index(inplace=True)
    df_res = reindex(df_res)
    
    # Retrieve boundary condition data. 
    # Only way we have is through the forecast request. 
    scenario = env.scenario
    requests.put('{0}/initialize'.format(env.url), 
                 data={'start_time':df_res['time'].iloc[0],
                       'warmup_period':0}).json()['payload']
    
    # Store original forecast parameters
    #forecast_parameters_original = requests.get('{0}/forecast_parameters'.format(env.url)).json()['payload']
    forecast_parameters_original = {"horizon": 3.1536e6,
                                    "interval": 900}
    tf = forecast_parameters_original["horizon"]
    dt = forecast_parameters_original["interval"]
    # Set forecast parameters for test. Take 10 points per step. 
    #forecast_parameters = {'horizon':env.max_episode_length/env.step_period, 
    #                       'interval':env.step_period}
    #requests.put('{0}/forecast_parameters'.format(env.url),
    #             data=forecast_parameters)
    forecast = requests.put('{0}/forecast'.format(env.url), 
                            data={'point_names': list(env.all_predictive_vars.keys()),
                                  'horizon': tf,
                                  'interval': dt
                                  }).json()['payload']
    # Back to original parameters, just in case we're testing during training
    requests.put('{0}/forecast_parameters'.format(env.url),
                 data=forecast_parameters_original)
        
    df_for = pd.DataFrame(forecast)
    df_for = reindex(df_for)
    df_for.drop('time', axis=1, inplace=True)
    
    df = pd.concat((df_res,df_for), axis=1)

    df = create_datetime(df)
    df = df.loc[~df.index.isna()]
    
>>>>>>> 99da257 (test)
    df.dropna(axis=0, inplace=True)
    scenario = env.scenario

    if save_to_file:
        df.to_csv(os.path.join(log_dir, 'results_tests_'+model_name+'_'+scenario['electricity_price'], 
                  'results_sim_{}.csv'.format(str(int(res['time'][0]/3600/24)))))
<<<<<<< HEAD
    
    # Project rewards into results index
    rewards_time_days = np.arange(df['time'][0], 
                                  env.start_time+env.max_episode_length,
=======
        
    rewards_time_days = np.arange(df_res['time'].iloc[0], 
                                  #env.start_time+env.max_episode_length,
                                  env.start_time + df_res['time'].iloc[-1],
                                  #env.start_time+env.max_episode_length+env.step_period,
>>>>>>> 99da257 (test)
                                  env.step_period)/3600./24.
    f = interpolate.interp1d(rewards_time_days, rewards, kind='zero',
                             fill_value='extrapolate')
    res_time_days = np.array(df['time'])/3600./24.
    rewards_reindexed = f(res_time_days)
    
    if not plt.get_fignums():
<<<<<<< HEAD
        # no window(s) are open, so open a new window. 
        _, axs = plt.subplots(4, sharex=True, figsize=(8,6))
=======
        # no window(s) open
        # fig = plt.figure(figsize=(10,8))
        _, axs = plt.subplots(4,1, sharex=True, figsize=(8,6))
>>>>>>> 99da257 (test)
    else:
        # There is a window open, so get current figure. 
        # Combine this with plt.ion(), plt.figure()
        fig = plt.gcf()
        axs = fig.subplots(nrows=4, ncols=1, sharex=True)
            
    x_time = df.index.to_pydatetime()

<<<<<<< HEAD
    axs[0].plot(x_time, df['reaTZon_y']  -273.15, color='darkorange',   linestyle='-', linewidth=1, label='_nolegend_')
    axs[0].plot(x_time, df['reaTSetHea_y'] -273.15, color='gray',       linewidth=1, label='Comfort setp.')
    axs[0].plot(x_time, df['reaTSetCoo_y'] -273.15, color='gray',       linewidth=1, label='_nolegend_')
    axs[0].set_yticks(np.arange(15, 31, 5))
    axs[0].set_ylabel('Operative\ntemperature\n($^\circ$C)')
    
    axs[1].plot(x_time, df['oveHeaPumY_u'],   color='darkorange',     linestyle='-', linewidth=1, label='_nolegend_')
    axs[1].set_ylabel('Heat pump\nmodulation\nsignal\n( - )')
=======
    #axs[0].plot(x_time, df['reaTZon_y']  -273.15, color='darkorange',   linestyle='-', linewidth=1, label='_nolegend_')
    axs[0].plot(x_time, df['TRooAir_y']  -273.15, color='darkorange',   linestyle='-', linewidth=1, label='_nolegend_')
    axs[0].plot(x_time, df['LowerSetp[1]'] -273.15, color='gray',       linewidth=1, label='Comfort setp.')
    axs[0].plot(x_time, df['UpperSetp[1]'] -273.15, color='gray',       linewidth=1, label='_nolegend_')
    axs[0].set_yticks(np.arange(15, 31, 5))
    axs[0].set_ylabel('Operative\ntemperature\n($^\circ$C)')
    
    axt = axs[0].twinx()
    axt.plot(x_time, df['PriceElectricPowerHighlyDynamic'], color='dimgray', linestyle='dotted', linewidth=1, label='Price')
    axs[0].plot([],[], color='dimgray', linestyle='-', linewidth=1, label='Price')
    
    axt.set_ylim(0,0.3)
    axt.set_yticks(np.arange(0, 0.31, 0.1))
    axt.set_ylabel('(EUR/kWh)')   
    axt.set_ylabel('Price\n(EUR/kWh)')
    
    axs[1].plot(x_time, df['oveAct_u'],   color='darkorange',     linestyle='-', linewidth=1, label='_nolegend_')
    #axs[1].set_ylabel('Heat pump\nmodulation\nsignal\n( - )')
    axs[1].set_ylabel('Heating power [W]')
>>>>>>> 99da257 (test)
    
    axs[2].plot(x_time, rewards_reindexed, 'b', linewidth=1, label='rewards')
    axs[2].set_ylabel('Rewards\n(-)')
    
    axs[3].plot(x_time, df['weaSta_reaWeaTDryBul_y'] - 273.15, color='royalblue', linestyle='-', linewidth=1, label='_nolegend_')
    axs[3].set_ylabel('Ambient\ntemperature\n($^\circ$C)')
    axs[3].set_yticks(np.arange(-5, 16, 5))
    axt = axs[3].twinx()
    
<<<<<<< HEAD
    axt.plot(x_time, df['weaSta_reaWeaHDirNor_y'], color='gold', linestyle='-', linewidth=1, label='$\dot{Q}_rad$')
    axt.set_ylabel('Solar\nirradiation\n($W$)')
=======
    #axt.plot(x_time, df['HDirNor'], color='gold', linestyle='-', linewidth=1, label='$\dot{Q}_rad$')
    #axt.set_ylabel('Solar\nirradiation\n($W$)')
>>>>>>> 99da257 (test)
    
    axs[3].plot([],[], color='darkorange',  linestyle='-', linewidth=1, label='RL')
    axs[3].plot([],[], color='dimgray',     linestyle='dotted', linewidth=1, label='Price')
    axs[3].plot([],[], color='royalblue',   linestyle='-', linewidth=1, label='$T_a$')
    axs[3].plot([],[], color='gold',        linestyle='-', linewidth=1, label='$\dot{Q}_{rad}$')
    axs[3].legend(fancybox=True, ncol=6, bbox_to_anchor=(1.06, -0.3)) 
    
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.tight_layout()
    
    if save_to_file:
        dir_name = os.path.join(log_dir, 'results_tests_'+model_name+'_'+scenario['electricity_price'])
        fil_name = os.path.join(dir_name,'results_sim_{}.pdf'.format(str(int(res['time'][0]/3600/24))))
        os.makedirs(dir_name, exist_ok=True)
        plt.savefig(fil_name, bbox_inches='tight')
    
    if not save_to_file:
        # showing and saving to file are incompatible
        plt.pause(0.001)
        plt.show()  

    
def reindex(df, interval=60, start=None, stop=None):
    '''
    Define the index. Make sure last point is included if 
    possible. If interval is not an exact divisor of stop,
    the closest possible point under stop will be the end 
    point in order to keep interval unchanged among index.
    
    ''' 
    
    if start is None:
        start = df['time'][0]
    if stop is None:
        stop  = df['time'][-1]  
    index = np.arange(start,stop+0.1,interval).astype(int)
    df_reindexed = df.reindex(index)
    
    # Avoid duplicates from FMU simulation. Duplicates lead to 
    # extrapolation errors
    df.drop_duplicates('time',inplace=True)
    
    for key in df_reindexed.keys():
        # Use linear interpolation 
        f = interpolate.interp1d(df['time'], df[key], kind='linear',
                                 fill_value='extrapolate')
        df_reindexed.loc[:,key] = f(index)
        
    return df_reindexed


def create_datetime_index(df):
    '''
    Create a datetime index for the data
    
    '''
    
    datetime = []
    for t in df['time']:
        datetime.append(pd.Timestamp('2023/1/1') + pd.Timedelta(t,'s'))
    df['datetime'] = datetime
    df.set_index('datetime', inplace=True)
    
    return df
    
    
from boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper, NormalizedObservationWrapper, DiscretizedActionWrapper, DiscretizedObservationWrapper
from generate_expert_traj import ExpertModelDisc
from stable_baselines3 import A2C, DQN
from examples.test_and_plot import test_agent
from copy import deepcopy
import torch
import collections
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

#env = DiscretizedActionWrapper(env, n_bins_act=200)
#env = DiscretizedObservationWrapper(env, n_bins_obs=100)

# Instantiate and train an RL algorithm

#model = A2C('MlpPolicy', env)
#model = DQN('MlpPolicy', env, verbose=1)

#import gym

#Hyperparameters
learning_rate = 1E-4
gamma         = 0.99
buffer_limit  = 50000
batch_size    = 1000

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 201)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            #return random.randint(0,1)
            #return int(round(random.randint(-10000,10000), -3))
            #return random.randint(0,20)
            #return random.randint(-1,1)
            return random.randint(0,200)
        else : 
            return out.argmax().item()
        
    def predict(self, obs, deterministic=True):
        obs = torch.tensor(obs)
        out = self.forward(obs)
        return torch.tensor(out.argmax().item()), dict()
    
def train(q, q_target, memory, optimizer):
    
    for i in range(1):
        
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        #q_a = q_out.gather(0, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        #loss = F.mse_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return loss

def main():
    # BOPTEST case address
    #url = 'http://127.0.0.1:5001'
    url = 'http://bacssaas_boptest:5000'

    # Instantite environment
    env = BoptestGymEnv(url                   = url,
                        actions               = ['oveAct_u'],
                        #observations          = {'TRooAir_y':(280.,310.), 'TDryBul':(280.,310.)}, 
                        observations          = {'TRooAir_y':(280.,310.)}, 
                        random_start_time     = False,
                        max_episode_length    = 48*3600,
                        predictive_period     = 900,
                        warmup_period         = 0,
                        step_period           = 900)
    
    
    # Add wrappers to normalize state and action spaces (Optional)
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)
    env = DiscretizedActionWrapper(env, n_bins_act=200)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())

    """
    Generate one expert demonstration episode of 1000 timesteps.
    """
    
    env.reset()
    Tset = 295.15
    K = 200
    #s = np.array([293.15, 293.15, 293.15])
    #s = np.array([0, 0, 0])
    #s = np.array([293.15])
    s = np.array([0])
    x = np.array([])
    u = np.array([])
    
    """
    Try stepping through env w/ embedded (P-) control.
    
    Try stepping through with poorly tuned control first,
    then nicely tuned.
    """
    
    filename = 'replay_buffer.pkl'
    
    if not os.path.exists(filename):    
        memory = ReplayBuffer()
        days = 1
        #K = 24*days*4
        K = 1000
        expert = ExpertModelDisc(env, 
                                n_bins_act=200,
                                TSet=Tset,
                                k=2000)
        for n in range(K):
            a, _ = expert.predict(s)
            u = np.append(u, a)
            a = torch.tensor(a)
            s_prime, r, done, truncated, info = env.step(a)
            x = np.append(x, s_prime[0])
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime
            
        """
        expert = ExpertModelDisc(env, 
                                n_bins_act=200,
                                TSet=Tset,
                                k=2000)
        
        for n in range(K):
            a, _ = expert.predict(s)
            u = np.append(u, a)
            a = torch.tensor(a)
            s_prime, r, done, truncated, info = env.step(a)
            x = np.append(x, s_prime[0])
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime
        """   
           
        fig, ax = plt.subplots(1,1, sharex=True)
        ax.plot(range(K), x)
        ax1 = ax.twinx()
        ax1.plot(range(K), u, color="r")
        plt.show()

        with open('replay_buffer.pkl', 'wb') as handle:
            pickle.dump(memory.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('replay_buffer.pkl', 'rb') as handle:
            memory = ReplayBuffer()
            memory.buffer = pickle.load(handle)
    
    """
    for n in range(100):
        a = K*(Tset - s[0])
        u = np.append(u, a)
        a = int(a/1000 + 10)
        a = torch.tensor(a) 
        s_prime, r, done, truncated, info = env.step(a)
        x = np.append(x, s_prime[0])
        done_mask = 0.0 if done else 1.0
        memory.put((s,a,r/100.0,s_prime, done_mask))
        s = s_prime
    """
    
    print_interval = 10
    #score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    """
    Train offline, off-policy:
    """

    #for n_epi in range(100):
    for n_epi in range(10000):
        
        loss = train(q, q_target, memory, optimizer)
                  
        score = 0.0
        epsilon = 0
        if n_epi % print_interval == 0 and n_epi != 0:
            # update target network:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, loss : {:.16f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, loss, memory.size(), epsilon*100))

    print_interval = 1
    
    """
    #for n_epi in range(10000):
    for n_epi in range(100):
        #epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/10)) #Linear annealing from 8% to 1%
        s, _ = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            a = torch.tensor(a)      
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break
            
        #if memory.size() > 100:
        #    train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
            
    env.close()
    """
    model = q
    observations, actions, rewards, kpis = test_agent(env,
                                                      model, 
                                                      start_time=0, 
                                                      episode_length=3600*48,
                                                      #episode_length=1,
                                                      #warmup_period=24*3600,
                                                      warmup_period=0,
                                                      plot=True)
    print(observations)

if __name__ == '__main__':
    main()
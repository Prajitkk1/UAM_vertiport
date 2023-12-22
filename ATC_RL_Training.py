import numpy as np
import gym
import torch
# from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
import pickle
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from environment import environment
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from torch import nn
from GL_Policy import CustomBaselinePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os

class TensorboardCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix='rl_model', ep_len: int = 200, log_freq: int = 1_000, verbose: int = 0):
        super(TensorboardCallback,self).__init__(verbose)
        self.ep_len = ep_len
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
    
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        #Logging the number of tasks completed
        tasks_completed = self.training_env.get_attr('tasks_completed')
        step_time = self.training_env.get_attr('step_time')
        total_delay = self.training_env.get_attr('total_delay')
        collisions = self.training_env.get_attr('collisions')
        avoided_collisions = self.training_env.get_attr("avoided_collisions")
        good_takeoffs = self.training_env.get_attr("good_takeoffs")
        good_landings = self.training_env.get_attr("good_landings")
        avg_battery = self.training_env.get_attr("avg_battery")
        self.logger.record_mean('mean_step_time', step_time[0])
        if self.num_timesteps % self.ep_len == 0:
            self.logger.record_mean('ep_mean_tasks_completed', tasks_completed[0]) #indexing 0 since the output is a list for some reason...
            self.logger.record_mean('ep_mean_total_delay', total_delay[0]) 
            self.logger.record_mean('ep_mean_collisions', collisions[0])
            self.logger.record_mean("ep_mean_avoided_collisions", avoided_collisions[0])
            self.logger.record_mean("ep_mean_good_takeoffs", good_takeoffs[0])
            self.logger.record_mean("ep_mean_good_landings", good_landings[0])
            self.logger.record_mean("ep_mean_battery", avg_battery[0])
        if (self.num_timesteps % self.log_freq == 0): #displaying log data
            self.logger.dump(self.num_timesteps)
        if self.n_calls % self.save_freq == 0: #Saving the model 
            path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            self.model.save(path)
            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))
        return True #No need to experiment with early stopping yet

custom_callback = TensorboardCallback(ep_len = 1439,log_freq= 10000, save_freq=10000, save_path='./ATC_Model/',
                                         name_prefix='ATC_RL_Model_9_11_22') 
       
# env = DummyVecEnv([lambda: environment(5)])
# env = SubprocVecEnv([lambda: environment(5)])

env = environment(no_of_drones=4, type="regular")

model = PPO(CustomBaselinePolicy,env=env,tensorboard_log='ATC_RL_Model/',verbose=1,n_steps=20000,batch_size=10000,gamma=1,learning_rate=0.00001,device='cuda')
model.learn(total_timesteps=1_000_000,callback=custom_callback)
model.save("Final_ATC_RL_model")
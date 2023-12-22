import numpy as np
import gym
import torch
# from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
import pickle
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from environment_test import environment
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from GL_Policy import CustomGLPolicy, CustomBaselinePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
import random
import os
import csv



with open("test_results.csv", mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Agent", "Reward", "Collisions", "Delay", "Good Takeoffs", "Good Landings", "Battery", "Step Time", "Problem"])


    env = environment(no_of_drones=4, type="graph")

    ep_len = 1440
    test_eps = 50
   # model = PPO.load('ATC_Model/ATC_GRL_Model_9_11_22_1200000_steps')
    obs = env.reset()
    for i in range(1,ep_len*test_eps+1):
        #action, _ = model.predict(obs,deterministic=True)
        action = random.randint(0,10)
        if i % ep_len == 0 and i > 0:
            reward, collisions, total_delay, good_takeoffs, good_landings, avg_battery, step_time = env.step(action)
            writer.writerow(["Random_walk", reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time, "UAM-VSM"])
            print("Iteration", i // ep_len)
            obs = env.reset()
        else:
            obs, reward, done, info = env.step(action)

    # del model
    # del env

    # env = environment(no_of_drones=4, type="regular")

    # model = PPO.load('ATC_Model/ATC_RL_Model_9_14_22_910000_steps')
    # obs = env.reset()
    # for i in range(1,ep_len*test_eps+1):
    #     action, _ = model.predict(obs,deterministic=True)
    #     if i % ep_len == 0 and i > 0:
    #         reward, collisions, total_delay, good_takeoffs, good_landings, avg_battery, step_time = env.step(action)
    #         writer.writerow(["RL", reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time, "UAM-VSM"])
    #         print("Iteration",i // ep_len)
    #         obs = env.reset()
    #     else:
    #         obs, reward, done, info = env.step(action)
    
    # del model
    # del env
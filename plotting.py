# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:38:29 2022

@author: Prajit
"""


###############this plotting is for test results###########################
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import os
#import matplotlib.backends.backend_pdf
perfect_jobs = list()
perfect_job_rl = list()
perfect_job_grl = list()



# df = pd.read_csv("test_results_safe_yes.csv")
# df1 = pd.read_csv("test_results_safe.csv")
df = pd.read_csv("test_results2.csv")
df1 = pd.read_csv("test_results_random_walk_final.csv")

# name = "results/result"+str(3)+".npy"
# arr = np.load(name,allow_pickle=True)
df = df.append(df1)
# for i in range(1,len(os.listdir("all_results/rw"))):
#     name = "all_results/rw/result"+str(i)+".npy"
#     arr = np.load(name,allow_pickle=True)
#     perfect_jobs.append([arr.item().get("good_takeoffs"), arr.item().get("good_landings"),arr.item().get("collisions"), "Random-Walk"])
#     #perfect_jobs.append([arr.item().get("good_takeoffs")+ arr.item().get("good_landings"),arr.item().get("collisions"), "rl", "6"])
#    # perfect_jobs.append([arr.item().get("good_takeoffs")+ arr.item().get("good_landings"),arr.item().get("collisions"), "grl", "6"])
    

# for i in range(1,len(os.listdir("all_results/grl"))):
#     name = "all_results/grl/result"+str(i)+".npy"
#     arr = np.load(name,allow_pickle=True)
#     perfect_jobs.append([arr.item().get("good_takeoffs"), arr.item().get("good_landings"),arr.item().get("collisions"), "Graph Learning"])
#     # perfect_jobs.append([arr.item().get("good_takeoffs")+ arr.item().get("good_landings"),arr.item().get("collisions"), "rl", "4"])
#     # perfect_jobs.append([arr.item().get("good_takeoffs")+ arr.item().get("good_landings"),arr.item().get("collisions"), "grl", "4"])
    
    
# for i in range(1,len(os.listdir("results_5robots"))):
#     name = "results_4robots/result"+str(i)+".npy"
#     arr = np.load(name,allow_pickle=True)
#     perfect_jobs.append([arr.item().get("good_takeoffs")+ arr.item().get("good_landings"),arr.item().get("collisions"), "rw", "5"])
#     perfect_jobs.append([arr.item().get("good_takeoffs")+ arr.item().get("good_landings"),arr.item().get("collisions"), "rl", "5"])
#     perfect_jobs.append([arr.item().get("good_takeoffs")+ arr.item().get("good_landings"),arr.item().get("collisions"), "grl", "5"])
    

#df = pd.DataFrame(perfect_jobs, columns=["Good takeoffs", "Good landings","Collisions", "Algorithm"])
#df1 = pd.DataFrame(perfect_job_grl, columns=["good takeoffs", "good landings","collisions", "algorithm"])

df["Delay"] = df["Delay"].div(60*60).round(2)
#fig, axes = plt.subplots(1, 2, figsize=(15, 5))

data = pd.melt(df, id_vars=["Agent", "Reward", "Step Time", "Problem"], var_name="Number")


# sns.boxplot(ax = axes[0], x = df["good takeoffs"], y = df["perfect jobs"],hue=df["algorithm"], palette="Blues")
# sns.boxplot(ax = axes[1], x = df["good landings"], y = df["collisions"],hue=df["algorithm"], palette="Blues", showfliers = False)


# sns.boxplot(ax = axes[0], x = [df["good takeoffs"], df["good takeoffs"], df["good landings"]], y = df["good takeoffs"],hue=df["algorithm"], palette="Blues")
# sns.boxplot(ax = axes[1], x = df["robots"], y = df["collisions"],hue=df["algorithm"], palette="Blues", showfliers = False)

#sns.boxplot(x = "variable", y = "value", data = pd.melt(df), hue=df["algorithm"])

sns.boxplot(x = data["Number"],y=data["value"], hue=data["Agent"], showfliers=False)
plt.ylim(0, 100)
plt.savefig('test_plot_final.png', dpi = 600)  

# plt.xticks([0],'')
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:19:29 2022

@author: prajit
"""

###############this plotting is for comparing safety things###########################
"""

we can compare collisions over training episodes 
Plot 1: 
    x axis: 20, 100, 400, max episodes
    yaxis: collision count
plot 2:
    Tested scenarios
    X axis: takeoff, landings, collision, mean battery
    y axis: values
"""



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
#import matplotlib.backends.backend_pdf


#plot 1 data


#plot 2 data



all_data = list()


for i in range(1,len(os.listdir("all_results/safety_0"))):
    name = "all_results/safety_0/result"+str(i)+".npy"
    arr = np.load(name,allow_pickle=True)
    all_data.append([arr.item().get("good_takeoffs"), arr.item().get("good_landings"),arr.item().get("collisions"),arr.item().get("avg_battery"), "S_0"])



for i in range(1,len(os.listdir("all_results/safety_1"))):
    name = "all_results/safety_1/result"+str(i)+".npy"
    arr = np.load(name,allow_pickle=True)
    all_data.append([arr.item().get("good_takeoffs"), arr.item().get("good_landings"),arr.item().get("collisions"),arr.item().get("avg_battery"), "S_1"])

    
# for i in range(1,len(os.listdir("results_5robots"))):
#     name = "results_4robots/result"+str(i)+".npy"
#     arr = np.load(name,allow_pickle=True)
#     perfect_jobs.append([arr.item().get("good_takeoffs")+ arr.item().get("good_landings"),arr.item().get("collisions"), "rw", "5"])
#     perfect_jobs.append([arr.item().get("good_takeoffs")+ arr.item().get("good_landings"),arr.item().get("collisions"), "rl", "5"])
#     perfect_jobs.append([arr.item().get("good_takeoffs")+ arr.item().get("good_landings"),arr.item().get("collisions"), "grl", "5"])
    

df = pd.DataFrame(all_data, columns=["Good takeoffs", "Good landings","Collisions", "parameter"])


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

data = pd.melt(df, id_vars=["parameter"], var_name="Number")
# sns.boxplot(ax = axes[0], x = df["good takeoffs"], y = df["perfect jobs"],hue=df["algorithm"], palette="Blues")
sns.boxplot(ax = axes[1], x = data["Number"],y=data["value"], hue=data["Algorithm"], palette="Blues", showfliers = False)

#sns.boxplot(x = data["Number"],y=data["value"], hue=data["Algorithm"])
plt.savefig('collision_plot.png', dpi = 600)  

# plt.xticks([0],'')
plt.show()
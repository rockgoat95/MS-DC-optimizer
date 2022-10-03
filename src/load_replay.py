import pandas as pd
from env.Shadower_simple import shadowerEnvSimple
from env.Shadower import shadowerEnv

from torch import nn
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

from copy import deepcopy

FRAME = 5

preprocess_function = None
att_p = 111
defense_ignore = 94.17
critical_damage = 126
main_stat = 39433
sub_stat = 3303 + 4998
damage = 148
boss_damage = 340
final_damage = 43.75
boss_defense = 300 
dealing_time = 360
env = shadowerEnvSimple(FRAME, main_stat, sub_stat, damage, boss_damage, critical_damage, att_p, defense_ignore, final_damage, boss_defense, dealing_time)

policy_kwargs = dict(activation_fn=nn.Tanh,
                     net_arch=[dict(pi=[64, 64], vf=[64, 64])])


model = sb3.PPO("MlpPolicy", env, verbose=1,
                learning_rate = 3e-4,
                gamma = 0.99,
                gae_lambda = 0.9,
                clip_range = 0.1,
                )

model.load("model/Shadower3")

max_score = 0
action_list_at_max_score = []

for j in range(1):
    print(j)
    obs = env.reset()
    score = 0
    df_list = []
    action_list = []
    for i in range(3000):
        df_list.append(obs)
        action, _states = model.predict([obs], deterministic= True)
        obs, rewards, dones, info = env.step(action)
        score += rewards 
        action_list.append(action)
    if score > max_score:
        max_score = deepcopy(score)
        action_list_at_max_score = deepcopy(action_list)
        df = pd.DataFrame(df_list, columns = env.state_labels)

print(max_score)
   

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import random

image_list  = []

for i in range(7):
    image_list.append(mpimg.imread('fig/skill'+ str(i) +'.png'))


fig, ax = plt.subplots()
fig.set_size_inches(40, 10)
ax.set_xlim([0, 3000/5])
ax.set_ylim([0, 8])

for i in range(3000):
    if action_list_at_max_score[i] == 6:
        continue
    # if df['delay'][i] !=0:
    #     continue
    # if i > 0 and df.iloc[i-1,action_list_at_max_score[i]+5] != 0:
    #     continue
    imagebox = OffsetImage(image_list[action_list_at_max_score[i]], zoom =1.7)
    ab = AnnotationBbox(imagebox, (i/5, action_list_at_max_score[i]+1), frameon = False)
    ax.add_artist(ab)
    
ax.vlines(np.arange(0, 600, 95), 0 , 10)
    
ax.vlines(np.arange(0, 600, 95)+10, 0 , 10, color = 'red')
ax.vlines(np.arange(0, 600, 95)+30, 0 , 10, color = 'blue')
from env.Shadower_simple import ShadowerEnvSimple
from env.ability.Shadower import * 

import stable_baselines3 as sb3
from torch import nn

import pandas as pd

from copy import deepcopy

import gym

# 1183
# 4746 m2 inc
# 14878 wp inc
FRAME = 5
dealing_time = 360
nobless = True
doping = True

epi_num = 3000  # 최대 에피소드 설정

# 스펙계산기 이용 후 입력 

if nobless:
    ability.add(damage = 30, boss_damage = 30, critical_damage = 30)
if doping:
    ability.add(defense_ignore = 20, boss_damage = 20, total_att = 60)
    

env = ShadowerEnvSimple(5, ability, 300, 360, common_attack_rate = 0.85, reward_divider = 1e10, test = True)
env = gym.wrappers.RecordEpisodeStatistics(env) 
env.reset()

model = sb3.PPO.load("best_model/shadower", env=env)

max_score = 0
action_list_at_max_score = []


obs = env.reset()
score = 0
df_list = []
action_list = []
for i in range(3000):
    df_list.append(obs)
    action, _states = model.predict([obs], deterministic=True)
    obs, rewards, dones, info = env.step(action)
    score += rewards
    action_list.append(action)

obs_data = pd.DataFrame(df_list, columns = env.state_labels)

### plottinh dealcycle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import random

image_list = []

for i in range(7):
    image_list.append(mpimg.imread("fig/shadower/skill" + str(i) + ".png"))


fig, ax = plt.subplots()
fig.set_size_inches(20, 7)
ax.set_xlim([-20, 380])
ax.set_ylim([0, 8])

for i in range(360*FRAME):
    if action_list[i] == 7:
        continue
    # if df['delay'][i] !=0:
    #     continue
    # if i > 0 and df.iloc[i-1,action_list_at_max_score[i]+5] != 0:
    #     continue
    imagebox = OffsetImage(image_list[action_list[i]], zoom=1.2)
    ab = AnnotationBbox(
        imagebox, (i / 5+7, action_list[i] + 1), frameon=False
    )
    ax.add_artist(ab)

#리레 
for i in range(2):
    ax.fill_between([0+182*i,15+182*i], [10,10], color = 'red', alpha = 0.3)
    
# 메용 2
for i in range(2):
    ax.fill_between([0+182*i,30+182*i], [10,10], color = 'gray', alpha = 0.3)

# 웨펖    
for i in range(2):
    ax.fill_between([91+182*i,91+182*i], [10,10], color = 'yellow', alpha = 0.3)

# 레투다
for i in range(4):
    ax.fill_between([0+91*i,15+91*i], [10,10], color = 'blue', alpha = 0.3)
# 소울 컨트랙트
for i in range(4):
    ax.fill_between([0+91*i,10+91*i], [10,10], color = 'pink', alpha = 0.3)

ax.vlines(np.arange(0, 380, 30), 0, 10, color="gray", linestyles= 'dotted', alpha = 0.7)

fig.savefig('res/shadower_dealcycle.png')
ax.set_xlabel('time(seconds)', size = 15)
ax.set_yticks([])
ax.set_title('Shadower deal cycle', size = 20)

plt.show()
from env.Shadower_simple import ShadowerEnvSimple
from env.ability.Shadower import *

import wandb

import stable_baselines3 as sb3
import gym
from torch import nn

import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
import numpy as np

def replay(env : gym.Env, model, run : wandb.init , plot : bool = True, job : str = "shadower", get_obs : bool = False, FRAME : int = 5):

    obs = env.reset()
    reward = 0
    df_list = []
    action_list = []
    for i in range(env.FRAME * env.dealing_time):
        df_list.append(obs)
        action, _states = model.predict([obs], deterministic=True)
        obs, reward_, dones, info = env.step(action)
        reward += reward_
        action_list.append(action)

    obs_data = pd.DataFrame(df_list, columns=env.state_labels)

    ### plotting dealcycle

    if plot:
        image_list = []

        for i in range(len(os.listdir("fig/" + job))):
            image_list.append(mpimg.imread("fig/"+job+"/skill" + str(i) + ".png"))


        fig, ax = plt.subplots()
        fig.set_size_inches(20, 7)
        ax.set_xlim([-20, 380])
        ax.set_ylim([0, 8])

        for i in range(env.FRAME * env.dealing_time):
            if action_list[i] == 7:
                continue
            # if df['delay'][i] !=0:
            #     continue
            # if i > 0 and df.iloc[i-1,action_list_at_max_score[i]+5] != 0:
            #     continue
            imagebox = OffsetImage(image_list[action_list[i][0]], zoom=1.2)
            ab = AnnotationBbox(imagebox, (i / env.FRAME + 7, action_list[i][0] + 1), frameon=False)
            ax.add_artist(ab)

        # 리레
        for i in range(2):
            ax.fill_between([0 + 182 * i, 15 + 182 * i], [10, 10], color="red", alpha=0.3)

        # 메용 2
        for i in range(2):
            ax.fill_between([0 + 182 * i, 30 + 182 * i], [10, 10], color="gray", alpha=0.3)

        # 웨펖
        for i in range(2):
            ax.fill_between([91 + 182 * i, 91 + 182 * i], [10, 10], color="yellow", alpha=0.3)

        # 레투다
        for i in range(4):
            ax.fill_between([0 + 91 * i, 15 + 91 * i], [10, 10], color="blue", alpha=0.3)
        # 소울 컨트랙트
        for i in range(4):
            ax.fill_between([0 + 91 * i, 10 + 91 * i], [10, 10], color="pink", alpha=0.3)

        ax.vlines(np.arange(0, 380, 30), 0, 10, color="gray", linestyles="dotted", alpha=0.7)

        fig.savefig("temp/" + job + "_dealcycle.png")
        ax.set_xlabel("time(seconds)", size=15)
        ax.set_yticks([])
        ax.set_title(job +" deal cycle", size=20)
        
        image_array = mpimg.imread("temp/" + job + "_dealcycle.png")

        images = wandb.Image(image_array, caption="optimized deal cycle")

        run.log({"deal-cycle": images})
    
    if get_obs:
        return reward, obs_data
    else:
        return reward 

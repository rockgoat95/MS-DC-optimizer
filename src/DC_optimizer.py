from env.Shadower_simple import ShadowerEnvSimple
from module.schema.common import Ability
from env.ability.Shadower import * 

import stable_baselines3 as sb3
from torch import nn

from typing import Callable

import gym

import wandb

wandb.init(project="MS-DC-optimizer", entity="rockgoat95")

from wandb.integration.sb3 import WandbCallback

# 1183
# 4746 m2 inc
# 14878 wp inc
FRAME = 5
dealing_time = 400
nobless = True
doping = True

epi_num = 3000  # 최대 에피소드 설정


# 스펙계산기 이용 후 입력 


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": dealing_time * FRAME * epi_num,
    "env_name": "Shadower",
    "learning_rate" : "4e-4"
    
}


run = wandb.init(
    name = "Shadower",
    project="MS-DC-optimizer",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


if nobless:
    ability.add(damage = 30, boss_damage = 30, critical_damage = 30)
if doping:
    ability.add(defense_ignore = 20, boss_damage = 20, total_att = 60)
    

env = ShadowerEnvSimple(FRAME, ability, 300, dealing_time, common_attack_rate = 0.75, reward_divider = 2e10)
env = gym.wrappers.RecordEpisodeStatistics(env) 
env.reset()


def linear_schedule(initial_value) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value * (progress_remaining*0.95 + 0.05)
            
    return func

def descrete_schedule(epi_num) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        if (epi_num*(1-progress_remaining)) < 500:
            return 5e-4
        elif (epi_num * (1-progress_remaining)) < 1000:
            return 2e-4
        else:
            return 1e-4
            
    return func

policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[400, 200], vf=[400, 200])])


model = sb3.PPO(
    config["policy_type"],
    env,
    verbose=1,
    learning_rate=descrete_schedule(epi_num),
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    tensorboard_log=f"runs/{run.id}"
)

model.learn(total_timesteps=dealing_time * FRAME * epi_num,
    callback = WandbCallback(
        gradient_save_freq= 100,
        model_save_path=f"models/{run.id}",
        verbose=2,
        
    ))  


wandb.finish()

model.save("best_model/shadower")
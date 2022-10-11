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
from stable_baselines3.common.callbacks import EvalCallback
# 1183
# 4746 m2 inc
# 14878 wp inc
FRAME = 5
dealing_time = 400
nobless = True
doping = True

epi_num = 3000  # 최대 에피소드 설정


# 스펙계산기 이용 후 입력 

sweep_configuration = {
    'method': 'random',
    'name': 'Shadower',
    'metric': {'goal': 'maximize', 'name': 'reward'},
    'parameters': 
    {
        'batch_size': {'values': [16, 32, 64, 128]},
        'clip_range': {'values': [0.1, 0.2]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'values':['constant', 'scheduler1', 'scheduler2']}
     }
}

sweep_id = wandb.sweep(sweep_configuration, project="MS-DC-optimizer", entity = 'rockgoat95')

default_config = {
    "total_timesteps": dealing_time * FRAME * epi_num,
    'batch_size' : 64,
    'clip_range' : 0.1,
    'epochs' : 10,
    'lr' : 4e-4,
}



if nobless:
    ability.add(damage = 30, boss_damage = 30, critical_damage = 30)
if doping:
    ability.add(defense_ignore = 20, boss_damage = 20, total_att = 60)
    

def train(config = None):
    with wandb.init(config = default_config, project = "MS-DC-optimizer", entity = "rockgoat95") as run:
        env = ShadowerEnvSimple(5, ability, 300, 360, common_attack_rate = 0.85, reward_divider = 1e10)
        env.reset()


        def linear_schedule(initial_value) -> Callable[[float], float]:
            def func(progress_remaining: float) -> float:
                return initial_value * (progress_remaining*0.95 + 0.05)
                    
            return func

        def descrete_schedule(epi_num) -> Callable[[float], float]:
            def func(progress_remaining: float) -> float:
                if (epi_num*(1-progress_remaining)) < 500:
                    return 4e-4
                elif (epi_num * (1-progress_remaining)) < 1000:
                    return 2e-4
                else:
                    return 1e-4
            return func

        def descrete_schedule2(epi_num) -> Callable[[float], float]:
            def func(progress_remaining: float) -> float:
                if (epi_num*(1-progress_remaining)) < 1000:
                    return 4e-4
                elif (epi_num * (1-progress_remaining)) < 2000:
                    return 2e-4
                else:
                    return 1e-4
            return func


        policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[200, 100], vf=[200, 100])])


        if wandb.config.lr == 'constant':
            lr = 4e-4
        elif wandb.config.lr == 'scheduler1':
            lr = descrete_schedule(epi_num)
        elif wandb.config.lr == 'scheduler2':
            lr = descrete_schedule2(epi_num)

        model = sb3.PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=wandb.config.clip_range,
            batch_size = wandb.config.batch_size,
            n_epochs = wandb.config.n_epochs,
            policy_kwargs=policy_kwargs
        )

        eval_callback = EvalCallback(env, best_model_save_path="./best_models/",
                                    log_path="./logs/", eval_freq=10,
                                    deterministic=True, render=False)
        model.learn(total_timesteps=dealing_time * FRAME * epi_num,
            callback = [eval_callback])  

        obs = env.reset()
        reward = 0
        for i in range(epi_num*FRAME):
            action, _ = model.predict([obs], deterministic=True)
            obs, reward_ ,_ ,_ = env.step(action)
            reward += reward_
            
        wandb.log({"reward": reward})


wandb.agent(sweep_id, train)

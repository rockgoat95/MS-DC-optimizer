from env.Shadower_simple import ShadowerEnvSimple
from module.schema.common import Ability
from env.ability.Shadower import *

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch import nn
import torch 

from typing import Callable

import wandb

from wandb.integration.sb3 import WandbCallback

# 1183
# 4746 m2 inc
# 14878 wp inc
FRAME = 5
dealing_time = 400
nobless = True
doping = True

<<<<<<< HEAD
epi_num = 1500  # 최대 에피소드 설정
=======
epi_num = 2000  # 최대 에피소드 설정
>>>>>>> a2133d98ba4d2e5d6604e75c8fc0a840f7456dcc


# 스펙계산기 이용 후 입력

sweep_configuration = {
<<<<<<< HEAD
    'method': 'random',
    'name': 'Shadower',
    'metric': {'goal': 'maximize', 'name': 'reward'},
    'parameters': 
    {
        'batch_size': {'values': [16, 32, 64, 128]},
        'clip_range': {'values': [0.1, 0.2]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'values':['constant', 'scheduler1']}
     }
=======
    "method": "random",
    "name": "Shadower",
    "metric": {"goal": "maximize", "name": "reward"},
    "parameters": {
        "batch_size": {"values": [32, 64, 128]},
        "clip_range": {"values": [0.1, 0.2]},
        "epochs": {"values": [5, 10, 15]},
        "lr": {"values": ["constant", "scheduler1", "scheduler2"]},
    },
>>>>>>> a2133d98ba4d2e5d6604e75c8fc0a840f7456dcc
}

sweep_id = wandb.sweep(
    sweep_configuration, project="MS-DC-optimizer", entity="rockgoat95"
)

default_config = {
    "total_timesteps": dealing_time * FRAME * epi_num,
<<<<<<< HEAD
    'batch_size' : 64,
    'clip_range' : 0.1,
    'epochs' : 10,
    'lr' : 'constant',
=======
    "batch_size": 64,
    "clip_range": 0.1,
    "epochs": 10,
    "lr": 4e-4,
>>>>>>> a2133d98ba4d2e5d6604e75c8fc0a840f7456dcc
}


if nobless:
    ability.add(damage=30, boss_damage=30, critical_damage=30)
if doping:
<<<<<<< HEAD
    ability.add(defense_ignore = 20, boss_damage = 20, total_att = 60)
    
best_reward = 0

run = wandb.init(config = default_config, project = "MS-DC-optimizer", entity = "rockgoat95")
def train(config = None):
    with wandb.init(config = default_config, project = "MS-DC-optimizer", entity = "rockgoat95") as run:
        env = ShadowerEnvSimple(5, ability, 300, dealing_time, common_attack_rate = 0.75, reward_divider = 2e10)
=======
    ability.add(defense_ignore=20, boss_damage=20, total_att=60)

best_reward = 0


def train(config=None):
    with wandb.init(
        config=default_config,
        project="MS-DC-optimizer",
        entity="rockgoat95",
        sync_tensorboard=True,
    ) as run:
        env = ShadowerEnvSimple(
            5, ability, 300, dealing_time, common_attack_rate=0.75, reward_divider=2e10
        )
>>>>>>> a2133d98ba4d2e5d6604e75c8fc0a840f7456dcc
        env.reset()

        def linear_schedule(initial_value) -> Callable[[float], float]:
            def func(progress_remaining: float) -> float:
                return initial_value * (progress_remaining * 0.95 + 0.05)

            return func

        def descrete_schedule(epi_num) -> Callable[[float], float]:
            def func(progress_remaining: float) -> float:
                if (epi_num * (1 - progress_remaining)) < 500:
                    return 4e-4
                elif (epi_num * (1 - progress_remaining)) < 1000:
                    return 2e-4
                else:
                    return 1e-4

            return func

        def descrete_schedule2(epi_num) -> Callable[[float], float]:
            def func(progress_remaining: float) -> float:
                if (epi_num * (1 - progress_remaining)) < 1000:
                    return 4e-4
                elif (epi_num * (1 - progress_remaining)) < 2000:
                    return 2e-4
                else:
                    return 1e-4

            return func

        policy_kwargs = dict(
            activation_fn=nn.ReLU, net_arch=[dict(pi=[200, 100], vf=[200, 100])]
        )

        if wandb.config.lr == "constant":
            lr = 4e-4
        elif wandb.config.lr == "scheduler1":
            lr = descrete_schedule(epi_num)
        elif wandb.config.lr == "scheduler2":
            lr = descrete_schedule2(epi_num)

        
        # device = torch.device("cuda")
        device = 'cpu'
        model = sb3.PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=wandb.config.clip_range,
<<<<<<< HEAD
            batch_size = wandb.config.batch_size,
            n_epochs = wandb.config.epochs,
            policy_kwargs=policy_kwargs, 
            device= device,
        )

        # model.learn(total_timesteps=dealing_time * FRAME * epi_num,
        #             callback = WandbCallback(
        #                 gradient_save_freq=100,
        #                 model_save_path=f"models/{run.id}",
        #                 verbose=2
        #             )) 
        
        model.learn(total_timesteps=1000,
                    callback = WandbCallback(
                        gradient_save_freq=100,
                        model_save_path=f"models/{run.id}",
                        verbose=2
                    ))  

        
        env = ShadowerEnvSimple(5, ability, 300, dealing_time, common_attack_rate = 0.75, reward_divider = 2e10, test = True)
        env.reset()
        reward = 0
        for i in range(dealing_time*FRAME):
=======
            batch_size=wandb.config.batch_size,
            n_epochs=wandb.config.epochs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"runs/{run.id}",
        )

        model.learn(
            total_timesteps=dealing_time * FRAME * epi_num,
            callback=WandbCallback(
                gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2,
            ),
        )

        env = ShadowerEnvSimple(
            5,
            ability,
            300,
            dealing_time,
            common_attack_rate=0.75,
            reward_divider=2e10,
            test=True,
        )
        obs = env.reset()
        reward = 0
        for i in range(dealing_time * FRAME):
>>>>>>> a2133d98ba4d2e5d6604e75c8fc0a840f7456dcc
            action, _ = model.predict([obs], deterministic=True)
            obs, reward_, _, _ = env.step(action)
            reward += reward_
<<<<<<< HEAD
            
        if reward > best_reward:
            best_reward = reward
            model.save("best_model/model")
=======

>>>>>>> a2133d98ba4d2e5d6604e75c8fc0a840f7456dcc
        wandb.log({"reward": reward})

        if reward > best_reward:
            best_reward = reward
            model.save("best_model/shadower")


wandb.agent(sweep_id, train)


wandb.finish()

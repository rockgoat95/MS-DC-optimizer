from env.Shadower_simple import ShadowerEnvSimple
from env.ability.Shadower import ability

from module.policy import DnnGluFeatureExtractor
from module.scheduler import *
from module.replay import replay

from torch import nn

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import wandb

from wandb.integration.sb3 import WandbCallback

FRAME = 5
# 400초 
dealing_time = 400
# 노블 적용 여부 
nobless = True
# 도핑 적용 여부 
doping = True

epi_num = 2000  # 최대 에피소드 설정

# epi_num = 3

sweep_configuration = {
    "method": "random",
    "name": "Shadower",
    "metric": {"goal": "maximize", "name": "reward"},
    "parameters": {
        "batch_size": {"values": [32, 64, 128]},
        "clip_range": {"values": [0.1, 0.2]},
        "epochs": {"values": [5, 10, 15]},
        "lr": {"values": ["constant", "scheduler1", "scheduler2"]},
        "network": {"values": ["MLP", "MLP-GLU"]}
    },
}

sweep_id = wandb.sweep(
    sweep_configuration, project="MS-DC-optimizer", entity="rockgoat95"
)

default_config = {
    "total_timesteps": dealing_time * FRAME * epi_num,
    "batch_size": 64,
    "clip_range": 0.1,
    "epochs": 10,
    "lr": "constant",
    "network": "MLP-GLU"
}

if nobless:
    ability.add(damage=30, boss_damage=30, critical_damage=30)
if doping:
    ability.add(defense_ignore=20, boss_damage=20, total_att=60)

best_reward = 0

# run =  wandb.init(
#         config=default_config,
#         project="MS-DC-optimizer",
#         entity="rockgoat95",
#         sync_tensorboard=True,
#     )

def train(config=None):
    with wandb.init(
        config=default_config,
        project="MS-DC-optimizer",
        entity="rockgoat95",
        sync_tensorboard=True,
    ) as run:
        
        global best_reward
        
        env = ShadowerEnvSimple(
            FRAME, ability, 300, dealing_time, common_attack_rate=0.75, reward_divider=2e10
        )
        
        
        # def make_env(env_: str, rank: int, seed: int = 0):
        #     """
        #     Utility function for multiprocessed env.
            
        #     :param env_id: (str) the environment ID
        #     :param num_env: (int) the number of environment you wish to have in subprocesses
        #     :param seed: (int) the inital seed for RNG
        #     :param rank: (int) index of the subprocess
        #     :return: (Callable)
        #     """
        #     def _init():
        #         env = env_
        #         env.seed(seed + rank)
        #         return env
        #     set_random_seed(seed)
        #     return _init
        
        # num_cpu = 2 
        
        # env_ = SubprocVecEnv([make_env(env, i) for i in range(num_cpu)])
        
        if wandb.config.network == "MLP":
            policy_kwargs = dict(activation_fn=nn.LeakyReLU,
                            net_arch=[dict(pi=[512, 256], vf=[512, 256])])
        elif wandb.config.network == "MLP-GLU":
            policy_kwargs = dict(
                features_extractor_class=DnnGluFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            )

        if wandb.config.lr == "constant":
            lr = 4e-4
        elif wandb.config.lr == "scheduler1":
            lr = descrete_schedule(epi_num)
        elif wandb.config.lr == "scheduler2":
            lr = descrete_schedule2(epi_num)

        model = sb3.PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=wandb.config.clip_range,
            batch_size=wandb.config.batch_size,
            n_epochs=wandb.config.epochs,
            policy_kwargs=policy_kwargs,
            device = 'cpu'
        )

        model.learn(
            total_timesteps=dealing_time * FRAME * epi_num,
            callback=WandbCallback(
                gradient_save_freq=500, model_save_path=f"models/{run.id}", verbose=2,
            ),
        )

        env = ShadowerEnvSimple(
            FRAME,
            ability,
            300,
            dealing_time,
            common_attack_rate=0.75,
            reward_divider=3e10,
            test=True,
        )

        reward = replay(env, model, run, FRAME = FRAME)

        wandb.log({"reward": reward})

        if reward > best_reward:
            best_reward = reward
            model.save("best_model/shadower")


wandb.agent(sweep_id, train)


wandb.finish()

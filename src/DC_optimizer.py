from env.Shadower_simple import ShadowerEnvSimple
from module.schema.common import Ability

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from torch import nn

import wandb

wandb.init(project="MS-DC-optimizer", entity="rockgoat95")

from wandb.integration.sb3 import WandbCallback

# 1183
# 4746 m2 inc
# 14878 wp inc
FRAME = 5
dealing_time = 360
nobless = True
doping = True

epi_num = 1000  # 최대 에피소드 설정

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": dealing_time * FRAME * epi_num,
    "env_name": "Shadower",
}

run = wandb.init(
    project="MS-DC-optimizer",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


ability = Ability(
    main_stat=42207,
    sub_stat=3430 + 5336,
    damage=118,
    boss_damage=287,
    att_p=111,
    defense_ignore=93.8,
    critical_damage=98,
    final_damage=45,
    buff_indure_time=20,
    total_att=2539,
    maple_goddess2_inc=4746,
    weapon_puff_inc=14878,
)

if nobless:
    ability.add(damage=30, boss_damage=30, critical_damage=30)
if doping:
    ability.add(defense_ignore=20, boss_damage=20, total_att=60)


env = ShadowerEnvSimple(4, ability, 300, 360, reward_divider=1e10)
env.reset()


policy_kwargs = dict(
    activation_fn=nn.Tanh, net_arch=[dict(pi=[400, 300], vf=[400, 300])]
)


model = sb3.PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.9,
    clip_range=0.1,
)

model.learn(total_timesteps=dealing_time * FRAME * epi_num, callback=WandbCallback())

model.save("model/Shadower3")

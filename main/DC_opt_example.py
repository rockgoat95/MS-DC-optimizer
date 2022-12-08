from env.Shadower_simple import ShadowerEnvSimple
#저장된 스탯 불러오기 
from env.ability.Shadower import ability

from module.scheduler import *

from torch import nn

import stable_baselines3 as sb3


FRAME = 5
# 400초 
dealing_time = 400
# 노블 적용 여부 
nobless = True
# 도핑 적용 여부 
doping = True

epi_num = 30  # 최대 에피소드 설정

# 노블 45포 기준 
if nobless:
    ability.add(damage=30, boss_damage=30, critical_damage=30)
    
# 반파별, 고보킬, 익스트림레드, 길축 적용량 
if doping:
    ability.add(defense_ignore=20, boss_damage=20, total_att=60)

# 환경 설정 
env = ShadowerEnvSimple(
    FRAME, ability, 300, dealing_time, common_attack_rate=0.75, reward_divider=2e10
)

# MLP layer
policy_kwargs = dict(activation_fn=nn.LeakyReLU,
                net_arch=[dict(pi=[512, 256], vf=[512, 256])])
# learning rate
lr = 4e-4

model = sb3.PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=lr,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    batch_size=32,
    n_epochs=5,
    policy_kwargs=policy_kwargs,
    device = 'cpu'
)

model.learn(
    total_timesteps=dealing_time * FRAME * epi_num
)



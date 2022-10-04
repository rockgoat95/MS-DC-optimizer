from env.Shadower_simple import shadowerEnvSimple
from env.Shadower import shadowerEnv
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from torch import nn


# 1183
# 4746 m2 inc
# 14878 wp inc
FRAME = 5
dealing_time = 360

preprocess_function = None
att_p = 111
defense_ignore = 94.17
critical_damage = 126
main_stat = 42207
sub_stat = 3430 + 5336
damage = 118
boss_damage = 287
final_damage = 45
boss_defense = 300


env = shadowerEnvSimple(
    FRAME,
    main_stat,
    sub_stat,
    damage,
    boss_damage,
    critical_damage,
    att_p,
    defense_ignore,
    final_damage,
    boss_defense,
    dealing_time,
)


epi_num = 1000  # 최대 에피소드 설정

policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=[dict(pi=[64, 64], vf=[64, 64])])


model = sb3.PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.9,
    clip_range=0.1,
)

model.learn(total_timesteps=dealing_time * FRAME * epi_num)

model.save("model/Shadower3")

import gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GLU(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.size_in = size_in
        self.linear = nn.Linear(size_in,size_in*2)
    def forward(self, X):
        out = self.linear(X)
        return out[:,:self.size_in] * out[:,self.size_in:].sigmoid()

class CustomPolicy(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param action_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, action_dim: int):
        super(CustomPolicy, self).__init__(observation_space, action_dim)
        n_input_channels = observation_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(n_input_channels,128),
            nn.LeakyReLU(0.1),
            GLU(128),
            nn.Linear(128,64),
            nn.LeakyReLU(0.1),
            GLU(64),
            nn.Linear(64, action_dim),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.network(nn.functional.softmax(observations))

policy_kwargs = dict(
    features_extractor_class=CustomPolicy,
    features_extractor_kwargs=dict(action_dim = 2),
)
model = PPO("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)

model.learn(10)


th.Tensor(a))
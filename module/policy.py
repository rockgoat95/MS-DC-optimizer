import gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GLU(nn.Module):
    def __init__(self, size_in):
        self.size_int = size_in
        self.linear1 = nn.Linear(size_in,size_in)
        self.linear2 = nn.Linear(size_in,size_in)
    def forward(self, X):
        return (self.linear1(X) * self.linear2(X)).sigmoid()

class CustomPolicy(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, action_dim: int):
        super(CustomPolicy, self).__init__(observation_space, action_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(n_input_channels,200),
            nn.LeakyReLU(),
            nn.GLU(100),
            nn.Linear(200,100),
            nn.LeakyReLU(),
            nn.GLU(100),
            nn.Linear(100, action_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.network(observations)

policy_kwargs = dict(
    features_extractor_class=CustomPolicy,
    features_extractor_kwargs=dict(action_dim = 8),
)
model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
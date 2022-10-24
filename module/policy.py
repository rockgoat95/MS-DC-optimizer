import gym
import torch as th
import torch.nn as nn

class GLU(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.size_in = size_in
        self.linear = nn.Linear(size_in,size_in*2)
    def forward(self, X):
        out = self.linear(X)
        return out[:,:self.size_in] * out[:,self.size_in:].sigmoid()

class DnnGlu(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(
            nn.Linear(size_in,128),
            nn.LeakyReLU(0.1),
            GLU(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,64),
            nn.LeakyReLU(0.1),
            GLU(64),
            nn.LeakyReLU(0.1)
        ))
        
    def forward(self, X):
        out = self.net(X)
        return out
        

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DnnGluFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(DnnGluFeatureExtractor, self).__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.net = DnnGlu(n_input, features_dim)
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

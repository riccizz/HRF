import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from tqdm import tqdm


class LowDimData():
    def __init__(self, data_type, device):
        self.batchsize = 100000
        self.device = device
        self.data_init(data_type)
        self.pairs = torch.stack([self.x0, self.x1], dim=1)

    def data_init(self, data_type):
        if data_type == "1to2":
            self.dim = 1
            self.mean = torch.tensor([1])
            # self.mean = torch.tensor([5])
            self.means = torch.ones((2, self.dim)) * self.mean
            self.means[1] = -self.means[1]
            self.var = torch.tensor([0.02])
            # self.var = torch.tensor([0.5])
            self.covs = self.var * torch.stack([torch.eye(self.dim) for _ in range(2)])
            self.probs = torch.tensor([0.5, 0.5])
            target_mix = Categorical(self.probs)
            target_comp = MultivariateNormal(self.means, self.covs)
            self.target_model = MixtureSameFamily(target_mix, target_comp)
            self.initial_model = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
            self.x1 = self.target_model.sample([self.batchsize]).to(self.device).detach()
            self.x0 = self.initial_model.sample([self.batchsize]).to(self.device).detach()
            # print(f"means and var before norm: \n{self.means.numpy()} {self.var.numpy()}")
            # self.x1 = (self.x1-torch.mean(self.x1)) / torch.std(self.x1)
            # print(f"means and var after norm {torch.mean(self.x1).item():.3f}, var: {torch.var(self.x1).item():.3f}")
        elif data_type == "2to2":
            self.dim = 1
            self.mean = torch.tensor([1])
            self.means = torch.ones((2, self.dim)) * self.mean
            self.means[1] = -self.means[1]
            self.var = torch.tensor([0.1])
            self.covs = self.var * torch.stack([torch.eye(self.dim) for _ in range(2)])
            self.probs = torch.tensor([0.5, 0.5])
            target_mix = Categorical(self.probs)
            target_comp = MultivariateNormal(self.means, self.covs)
            self.target_model = MixtureSameFamily(target_mix, target_comp)
            initial_mix = Categorical(self.probs)
            initial_comp = MultivariateNormal(self.means, self.covs)
            self.target_model = MixtureSameFamily(target_mix, target_comp)
            self.initial_model = MixtureSameFamily(initial_mix, initial_comp)
            self.x1 = self.target_model.sample([self.batchsize]).to(self.device).detach()
            self.x0 = self.initial_model.sample([self.batchsize]).to(self.device).detach()
            # print(f"means and var before norm: \n{self.means.numpy()} {self.var.numpy()}")
            # self.x1 = (self.x1-torch.mean(self.x1)) / torch.std(self.x1)
            # print(f"means and var after norm {torch.mean(self.x1).item():.3f}, var: {torch.var(self.x1).item():.3f}")
        elif data_type == "1to5":
            self.dim = 1
            self.means = torch.ones((5, self.dim)) * 5
            for i in range(5):
                self.means[i] *= i-2
            self.var = torch.tensor([0.5])
            self.covs = self.var * torch.stack([torch.eye(self.dim) for _ in range(5)])
            self.probs = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
            target_mix = Categorical(self.probs)
            target_comp = MultivariateNormal(self.means, self.covs)
            self.target_model = MixtureSameFamily(target_mix, target_comp)
            self.initial_model = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
            self.x1 = self.target_model.sample([self.batchsize]).to(self.device).detach()
            self.x0 = self.initial_model.sample([self.batchsize]).to(self.device).detach()
            print(f"means and var before norm: \n{self.means.numpy()} {self.var.numpy()}")
            self.x1 = (self.x1-torch.mean(self.x1)) / torch.std(self.x1)
            print(f"means and var after norm {torch.mean(self.x1).item():.3f}, var: {torch.var(self.x1).item():.3f}")
        elif data_type == "2D1to6":
            self.dim = 2
            D = 10.
            self.probs = torch.tensor([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
            self.means = torch.tensor([
                [D * np.sqrt(3) / 2., D / 2.], 
                [-D * np.sqrt(3) / 2., D / 2.], 
                [0.0, -D],
                [D * np.sqrt(3) / 2., - D / 2.], [-D * np.sqrt(3) / 2., - D / 2.], [0.0, D]
            ]).float()
            self.var = torch.tensor([0.5])
            self.covs = self.var * torch.stack([torch.eye(self.dim) for _ in range(6)])
            target_mix = Categorical(self.probs)
            target_comp = MultivariateNormal(self.means, self.covs)
            self.target_model = MixtureSameFamily(target_mix, target_comp)
            self.initial_model = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
            self.x1 = self.target_model.sample([self.batchsize]).to(self.device).detach()
            self.x0 = self.initial_model.sample([self.batchsize]).to(self.device).detach()
            print(f"means and var before norm: \n{self.means.numpy()} {self.var.numpy()}")
            self.x1 = (self.x1-torch.mean(self.x1)) / torch.std(self.x1)
            print(f"means and var after norm {torch.mean(self.x1).item():.3f}, var: {torch.var(self.x1).item():.3f}")
        elif data_type == "moon":
            self.dim = 2
            n_samples_out = self.batchsize // 2
            n_samples_in = self.batchsize - n_samples_out
            outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
            outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
            inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
            inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5
            X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                        np.append(outer_circ_y, inner_circ_y)]).T
            X += np.random.rand(self.batchsize, 1) * 0.2
            self.x1 = (torch.from_numpy(X) * 3 - 1).float().to(self.device).detach()
            self.x1 = self.x1[torch.randperm(self.batchsize)]
            
            self.probs = torch.tensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])
            self.means = torch.tensor([
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
                (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            ]).float() * 5
            self.var = torch.tensor([0.1])
            self.covs = self.var * torch.stack([torch.eye(self.dim) for _ in range(8)])
            initial_mix = Categorical(self.probs)
            initial_comp = MultivariateNormal(self.means, self.covs)
            self.initial_model = MixtureSameFamily(initial_mix, initial_comp)
            self.x0 = self.initial_model.sample([self.batchsize]).to(self.device).detach()
        else:
            raise NotImplementedError



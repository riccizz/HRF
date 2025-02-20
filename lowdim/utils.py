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


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        assert(self.dim%2==0)

    def forward(self,x):
        device = x.device
        half_dim = self.dim//2
        emb = math.log(10000)/(half_dim-1)
        emb = torch.exp(torch.arange(half_dim,device=device)*(-emb))
        if x.dim()==1:
            emb = x[...,None]*emb[None,:]
        elif x.dim()==2:
            emb = x[...,None]*emb[None,None,:]
        elif x.dim()==3:
            emb = x[...,None]*emb[None,None,None,:]
        else:
            assert(False)
        emb = torch.cat((emb.sin(),emb.cos()),dim=-1)
        return emb


def load_ckpt(rf_dir, dim, model, ckpt=None):
    ckpt_dir = os.path.join(rf_dir, f"ckpt")
    if ckpt is None:
        ckpt = sorted(os.listdir(ckpt_dir), key=lambda x: int(x.split("_")[-2].split(".")[0]))[-1]
    print(f"loading {ckpt}")
    checkpoint = torch.load(os.path.join(ckpt_dir, ckpt), weights_only=True)
    model.load_state_dict(checkpoint['v_net_state_dict'])
    model.eval()
    return model


@torch.no_grad()
def sample_ode(x0, N, model):
    dt = 1./N
    traj = []
    x = x0.detach().clone()
    batchsize = x.shape[0]

    traj.append(x.detach().clone())
    for i in range(N):
        t = torch.ones((batchsize, 1), device=x0.device) * i / N
        v = model(x, t)
        x = x.detach().clone() + v * dt
        traj.append(x.detach().clone())

    return torch.stack(traj) # (N+1, batchsize, data_dim)


@torch.no_grad()
def plot_traj(traj, distance, traj_dir, file_name, title):
    plt.figure()
    n, _, d = traj.shape
    n_traj = 1000
    start_color = '#4D4D4D'
    end_color = 'blue'
    flow_color = '#748B47'
    
    if d == 1:
        plt.scatter(traj[0, :n_traj, 0].cpu().numpy(), [0 for _ in range(n_traj)], s=4, alpha=1, c=start_color, label="Start", zorder=2)
        plt.scatter(traj[-1, :n_traj, 0].cpu().numpy(), [1 for _ in range(n_traj)], s=4, alpha=1, c=end_color, label="End", zorder=3)
    else:
        plt.scatter(traj[0, :n_traj, 0].cpu().numpy(), traj[0, :n_traj, 1].cpu().numpy(), s=4, alpha=0.6, c=start_color, label="Start")
        plt.scatter(traj[-1, :n_traj, 0].cpu().numpy(), traj[-1, :n_traj, 1].cpu().numpy(), s=4, alpha=1, c=end_color, label="End", zorder=3)
    
    if d == 1:
        label = f"Flow WD={distance:.3f}"
        plt.plot(traj[:, 0, 0].cpu().numpy(), np.linspace(0, 1, n), linewidth=1, alpha=0.4, c=flow_color, label=label)
    else:
        label = f"Flow SWD={distance:.3f}"
        plt.plot(traj[:, 0, 0].cpu().numpy(), traj[:, 0, 1].cpu().numpy(), linewidth=1, alpha=0.4, c=flow_color, label=label)
    for i in range(1, n_traj):
        if d == 1:
            plt.plot(traj[:, i, 0].cpu().numpy(), np.linspace(0, 1, n), linewidth=1, alpha=0.4, c=flow_color)
        else:
            plt.plot(traj[:, i, 0].cpu().numpy(), traj[:, i, 1].cpu().numpy(), linewidth=1, alpha=0.4, c=flow_color)

    # plt.xlim(-3, 3)
    if d == 1:
        # plt.gca().get_xaxis().set_visible(False)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Space', fontsize=18)
        plt.ylabel('Time', fontsize=18)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # plt.gca().spines['bottom'].set_visible(False)
    else:
        plt.axis('off')
    plt.title(title, fontsize=16)
    plt.legend(fontsize=16, framealpha=0.5, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(traj_dir, file_name), dpi=300, bbox_inches='tight')
    plt.close()


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


class VNet(torch.nn.Module):
    def __init__(self, data_dim=2, hidden_num=128):
        super().__init__()
        dim = self.dim = 32
        self.data_time_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim,dim),
            # torch.nn.LayerNorm(dim),
        )

        self.v_time_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim,dim),
            # torch.nn.LayerNorm(dim),
        )

        self.data_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.Flatten(),
            torch.nn.GELU(),
            torch.nn.Linear(data_dim*dim,dim),
            # torch.nn.LayerNorm(dim),
        )

        self.v_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.Flatten(),
            torch.nn.GELU(),
            torch.nn.Linear(data_dim*dim,dim),
            # torch.nn.LayerNorm(dim),
        )
        
        self.fc1 = torch.nn.Linear(4*dim, hidden_num*2, bias=True)
        self.fc2 = torch.nn.Linear(hidden_num*2, hidden_num, bias=True)
        self.fc3 = torch.nn.Linear(hidden_num, data_dim, bias=True)
        self.act = torch.nn.GELU()
    
    def forward(self, vt, t_v, xt, t):
        t = self.data_time_mlp(t).squeeze()
        if len(t.shape) == 1:
            t = t.unsqueeze(0)
        t_v = self.v_time_mlp(t_v).squeeze()
        if len(t_v.shape) == 1:
            t_v = t_v.unsqueeze(0)
        xt = self.data_mlp(xt)
        vt = self.v_mlp(vt)

        x = torch.cat([vt, t_v, xt, t], dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x


class VNetD(torch.nn.Module):
    def __init__(self, data_dim=2, depth=2, hidden_num=128):
        super().__init__()
        dim = self.dim = 64

        self.time_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.Flatten(),
            torch.nn.GELU(),
            torch.nn.Linear(depth*dim,depth*dim),
            # torch.nn.LayerNorm(dim),
        )

        self.data_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.Flatten(),
            torch.nn.GELU(),
            torch.nn.Linear(data_dim*depth*dim,depth*dim),
            # torch.nn.LayerNorm(dim),
        )

        self.fc1 = torch.nn.Linear(2*depth*dim, 2*depth*hidden_num, bias=True)
        self.fc2 = torch.nn.Linear(2*depth*hidden_num, depth*hidden_num, bias=True)
        self.fc3 = torch.nn.Linear(depth*hidden_num, data_dim, bias=True)
        self.act = torch.nn.GELU()
    
    def forward(self, xt, t):
        """
        xt: (batch_size, depth, data_dim)
        t: (batch_size, depth)
        """
        t = self.time_mlp(t)
        xt = self.data_mlp(xt)
        
        x = torch.cat([xt, t], dim=1)   # N x 2*D*dim
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x


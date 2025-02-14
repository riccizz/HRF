import math
import torch

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


class Net(torch.nn.Module):
    def __init__(self, data_dim=2):
        super().__init__()
        dim = 128

        self.time_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim,dim),
            torch.nn.LayerNorm(dim),
        )

        self.data_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.Flatten(),
            torch.nn.GELU(),
            torch.nn.Linear(data_dim*dim,2*dim),
            torch.nn.LayerNorm(2*dim),
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3*dim,3*dim),
            torch.nn.GELU(),
            torch.nn.Linear(3*dim, dim),
        )

        self.out_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, data_dim),
        )
    
    def forward(self, x_t, t):
        t_emb = self.time_mlp(t).squeeze()
        x_t_emb = self.data_mlp(x_t)
        x = self.mlp(torch.cat([x_t_emb,t_emb],dim=1))
        return self.out_mlp(x)


class VNet(torch.nn.Module):
    def __init__(self, data_dim=2, hidden_num=128):
        super().__init__()
        dim = self.dim = 32
        self.t_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim,dim),
        )

        self.tau_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim,dim),
        )

        self.x_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.Flatten(),
            torch.nn.GELU(),
            torch.nn.Linear(data_dim*dim,dim),
        )

        self.v_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.Flatten(),
            torch.nn.GELU(),
            torch.nn.Linear(data_dim*dim,dim),
        )
        
        self.fc1 = torch.nn.Linear(4*dim, hidden_num*2, bias=True)
        self.fc2 = torch.nn.Linear(hidden_num*2, hidden_num, bias=True)
        self.fc3 = torch.nn.Linear(hidden_num, data_dim, bias=True)
        self.act = torch.nn.GELU()
    
    def forward(self, vt, tau, xt, t):
        t = self.t_mlp(t).squeeze()
        if len(t.shape) == 1:
            t = t.unsqueeze(0)
        tau = self.tau_mlp(tau).squeeze()
        if len(tau.shape) == 1:
            tau = tau.unsqueeze(0)
        xt = self.x_mlp(xt)
        vt = self.v_mlp(vt)

        x = torch.cat([vt, tau, xt, t], dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x

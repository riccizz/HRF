import copy
import os

import torch
from torch import distributed as dist
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchvision.utils import save_image


def sample_rf(model, sample_shape, nfe, device, integration_method="euler"):
        if integration_method == "euler":
            xt = sample_rf_euler(model, sample_shape, nfe, device)
        elif integration_method == "dopri5":
            xt, nfe = sample_rf_dopri5(model, sample_shape, device)
        else:
            raise NotImplementedError
        return xt, nfe


def sample_rf_euler(model, sample_shape, nfe, device):
    node_ = NeuralODE(model, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(sample_shape, device=device),
            t_span=torch.linspace(0, 1, nfe, device=device),
        )
        traj = traj[-1, :].view(sample_shape)
    return traj


def sample_rf_dopri5(model, sample_shape, device):
    with torch.no_grad():
        step_counter = {"steps": 0}
        def wrapped_model(t, xt):
            step_counter["steps"] += 1 
            return model(t, xt) 
        t_span = torch.linspace(0, 1, 2, device=device)
        xt = odeint(
            wrapped_model, 
            torch.randn(sample_shape, device=device), 
            t_span, 
            rtol=1e-5, 
            atol=1e-5, 
            method="dopri5", 
        )
        xt = xt[-1, :]
        
    return xt, step_counter['steps']


def sample_hrf(model, sample_shape, N, M, device, integration_method="euler"):
        if integration_method == "euler":
            xt = sample_hrf_euler(model, sample_shape, N, M, device)
            nfe = N * M
        elif integration_method == "dopri5":
            xt, nfe = sample_hrf_dopri5(model, sample_shape, N, device)
        else:
            raise NotImplementedError
        return xt, nfe


def sample_hrf_euler(model, sample_shape, N, M, device):
    with torch.no_grad():
        batchsize = sample_shape[0]
        xt = torch.randn(sample_shape, device=device)
        
        t_values = torch.arange(N, device=device) / N
        tau_values = torch.arange(M, device=device) / M

        for i in range(N):
            t = t_values[i].expand(batchsize)
            vtau = torch.randn(sample_shape, device=device)
            for j in range(M):
                tau = tau_values[j].expand(batchsize) 
                a = model(tau, vtau, t, xt)
                vtau += a / M
            xt += vtau / N

    return xt


def sample_hrf_dopri5(model, sample_shape, N, device):
    with torch.no_grad():
        batchsize = sample_shape[0]
        xt = torch.randn(sample_shape, device=device)
        t_values = torch.arange(N, device=device) / N
        step_counter = {"steps": 0}
        for i in range(N):
            t = t_values[i].expand(batchsize)
            def wrapped_model(tau, vtau):
                step_counter["steps"] += 1 
                return model(tau, vtau, t, xt)
            tau_span = torch.linspace(0, 1, 2, device=device)
            vtau = odeint(
                wrapped_model, 
                torch.randn_like(xt), 
                tau_span, 
                rtol=1e-5, 
                atol=1e-5, 
                method="dopri5", 
            )
            vtau = vtau[-1, :]
            xt += vtau / N

    return xt, step_counter['steps']


def generate_samples(model, savedir, step, shape, device, net_="normal", hrf=True):
    """Save generated images for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)

    if hrf:
        samples, _ = sample_hrf(model_, shape, 1, 100, device)
    else:
        samples, _ = sample_rf(model_, shape, 100, device)
    
    save_image(samples.clip(-1, 1) / 2 + 0.5, os.path.join(savedir, f"{net_}_generated_FM_images_step_{step}.png"), nrow=4)

    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def load_model(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)


def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    print(f"[Rank {rank}] MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    print(f"[Rank {rank}] Initializing process group...")
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )
    print(f"[Rank {rank}] Process group initialized!")


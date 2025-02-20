import os
import time

import copy
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from absl import app, flags
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from utils import load_ckpt, plot_traj
from utils import LowDimData, VNetD

@torch.no_grad()
def sample_hierarchical(model, x_t, t, cur_depth, max_depth, N_list, return_traj=False):
    x_0 = x_t[:,cur_depth,...].clone()
    local_num_steps = N_list[cur_depth]
    times = torch.linspace(0.0,1.0,local_num_steps+1,device=x_t.device)
    dts = torch.diff(times)
    if return_traj:
        traj = [x_0]
    for k in range(local_num_steps):
        current_time = times[k]
        dt = dts[k]
        t[cur_depth] = current_time
        if k == 0 and cur_depth != 0:
            x_0 = torch.randn_like(x_t[:,cur_depth,...],device=x_t.device)
            x_t[:,cur_depth,...] = x_0
        if cur_depth+1==max_depth:
            f = model(x_t,t*torch.ones((x_t.shape[0],1),device=x_t.device))
            x_t[:,cur_depth,...] += dt*f
        else:
            x_t[:,cur_depth,...] += dt*sample_hierarchical(model, x_t, t, cur_depth+1, max_depth, N_list)[1]
        if return_traj:
            traj.append(x_t[:,cur_depth,...].detach().clone())
    if return_traj:
        return x_0, x_t[:,cur_depth,...], torch.stack(traj)
    else:
        return x_0, x_t[:,cur_depth,...]


def train_hrf(data, depth, N_list, checkpoint, iterations, base_dir, seed, device, progress):
    ckpt_dir = os.path.join(base_dir, f"ckpt")
    img_dir = os.path.join(base_dir, f"fig")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    loss_curve = []

    v_net = VNetD(data_dim=data.dim, depth=depth).to(device)
    optimizer = torch.optim.AdamW(v_net.parameters(), lr=1e-3)
    # show model size
    model_size = 0
    for param in v_net.parameters():
        model_size += param.data.nelement()
    print(f"Model params number: {model_size}")
    print("Model params: %.2f M" % (model_size / 1000 / 1000))

    A = torch.tril(torch.ones((depth, depth),device=device),diagonal=-1)
    with tqdm(initial=0,total=iterations) as pbar:
        for train_i in range(iterations):
            optimizer.zero_grad()
            indices = torch.randperm(len(data.pairs))[:checkpoint['batchsize']]
            batch = data.pairs[indices]
            x0 = batch[:, 0].detach().clone()   # N x d
            x1 = batch[:, 1].detach().clone()   # N x d

            x0 = torch.cat([x0[:,None,:], torch.randn((x0.shape[0],depth-1)+x0.shape[1:],device=device)], dim=1)   # N x D x d
            t = torch.rand((x1.shape[0],depth)+(1,)*(x1.dim()-1), device=device)

            xt = (1-t)*x0 + t*(x1[:,None,...] - torch.einsum('ij,bj...->bi...', A, x0))
            pred = v_net(xt, t.squeeze(list(range(2,t.dim()))))
            target = x1 - torch.sum(x0, dim=1) # N x d

            pred = v_net(xt, t)
            loss = torch.mean((target - pred) ** 2)
            loss.backward()

            optimizer.step()
            loss_curve.append(loss.item())
            pbar.set_description(f'loss: {loss.item():.4f}')
            pbar.update(1)

            if (train_i+1) % checkpoint['save_every_steps'] == 0 or train_i == (iterations-1):
                checkpoint['v_net_state_dict'] = v_net.state_dict()
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                checkpoint['step'] = (train_i+1)
                torch.save(checkpoint, os.path.join(ckpt_dir, f"hrf_{train_i+1}_D{depth}_seed{seed}.pt"))
                
                v_net.eval()
                x0 = data.initial_model.sample([data.batchsize]).to(data.device).detach()
                x0 = torch.cat([x0[:,None,:], torch.randn((x0.shape[0],depth-1)+x0.shape[1:],device=device)], dim=1)
                t0 = torch.zeros(depth, device=device)
                xt = sample_hierarchical(v_net, x0, t0, 0, depth, N_list)[1]
                if data.dim == 1:
                    distance = wasserstein_distance(data.x1[:, 0].cpu().numpy(), xt[:, 0].cpu().numpy())
                else:
                    distance = ot.sliced_wasserstein_distance(xt, data.x1, seed=1)
                print(f"{train_i+1} WD={distance} NFE={np.prod(N_list)} {N_list}")
                
                plt.figure()
                if data.dim == 1:
                    bins = np.linspace(-2, 2, 201)
                    plt.hist(data.x1[:, 0].cpu().numpy(), bins=bins, density=True, alpha=0.8, histtype='step', linewidth=1, label=f'Target')
                    plt.hist(xt.cpu().numpy(), bins=bins, density=True, alpha=0.8, histtype='step', linewidth=1, label=f'Gen WD={distance:.3f}')
                else:
                    plt.scatter(data.x0[:5000, 0].cpu().numpy(), data.x0[:5000, 1].cpu().numpy(), label="Source", alpha=0.25, s=3)
                    plt.scatter(data.x1[:5000, 0].cpu().numpy(), data.x1[:5000, 1].cpu().numpy(), label="Target", alpha=0.25, s=3)
                    plt.scatter(xt[:5000,0].cpu().numpy(), xt[:5000,1].cpu().numpy(), label=f'Gen SWD={distance:.3f}', alpha=0.25, s=3)
                plt.xlabel("Data")
                plt.ylabel("Density")
                plt.legend()
                plt.title(f'Distribution NFE={np.prod(np.array(N_list))} {N_list}')
                plt.tight_layout()
                plt.savefig(os.path.join(img_dir, f"dist_hrf_progress.png"))
                plt.close()
                v_net.train()

                plt.figure()
                plt.plot(loss_curve)
                plt.title('Training Loss Curve')
                plt.savefig(os.path.join(img_dir, "total_loss.png"))
                plt.close()
                
    return v_net


def main(argv):
    seed = FLAGS.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    iterations = FLAGS.iter
    base_dir = os.path.join(FLAGS.base_dir, FLAGS.data_type)
    device = torch.device(f'cuda:{FLAGS.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    data = LowDimData(data_type=FLAGS.data_type, device=device)
    checkpoint = {
        'v_net_state_dict': None,
        'optimizer_state_dict': None,
        'save_every_steps': 5000,
        'step': 0,
        'batchsize': FLAGS.batchsize,
    }
    hrf_dir = os.path.join(base_dir, "hrfD")
    img_dir = os.path.join(hrf_dir, f"fig")
    dist_dir = os.path.join(img_dir, "dist")
    traj_dir = os.path.join(img_dir, "traj")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    if FLAGS.mode == "train":
        # N_list = [100]
        # N_list = [10,10]
        N_list = [2,5,10]
        # N_list = [1,2,5,10]
        # N_list = [1,2,5,5,10]
        # N_list = [1,1,1,1,1,1,2,5,5,10]
        v_net = train_hrf(data, len(N_list), N_list, checkpoint, iterations, hrf_dir, seed, device, progress=True)

    elif FLAGS.mode == "eval":
        with torch.inference_mode():
            N_list = [2,5,10]
            depth = len(N_list)
            v_net = VNetD(data_dim=data.dim, depth=depth).to(device)
            step = 50000
            ckpt_name = f'hrf_{step}_D{depth}_seed{seed}'
            v_net = load_ckpt(hrf_dir, data.dim, v_net, ckpt=ckpt_name+'.pt')

            model_size = 0
            for param in v_net.parameters():
                model_size += param.data.nelement()
            print(f"Model params number: {model_size}")
            print("Model params: %.2f M" % (model_size / 1000 / 1000))

            x0_eval = data.initial_model.sample((data.batchsize,)).to(data.device).detach()
            x0 = torch.cat([x0_eval[:,None,:], torch.randn((x0_eval.shape[0],depth-1)+x0_eval.shape[1:],device=device)], dim=1)
            t0 = torch.zeros(depth, device=device)
            x0, xt, traj = sample_hierarchical(v_net, x0, t0, 0, depth, N_list, return_traj=True)
            if data.dim == 1:
                distance = wasserstein_distance(data.x1[:, 0].cpu().numpy(), xt[:,0].cpu().numpy())
            else:
                distance = ot.sliced_wasserstein_distance(data.x1, xt, seed=1)
            plot_traj(traj, distance, traj_dir, file_name=f"traj_{ckpt_name}_{N_list}.png", title=f'Trajectory with {N_list} Sampling Steps')

            plt.figure()
            if data.dim == 1:
                bins = np.linspace(-2, 2, 201)
                plt.hist(data.x1[:, 0].cpu().numpy(), bins=bins, color="#ff7f0e", density=True, alpha=0.8, histtype='step', linewidth=1, label=f'Target')
                plt.hist(xt.cpu().numpy(), bins=bins, color="#2ca02c", density=True, alpha=0.8, histtype='step', linewidth=1, label=f'Gen WD={distance:.3f}')
            else:
                plt.scatter(data.x0[:5000, 0].cpu().numpy(), data.x0[:5000, 1].cpu().numpy(), c="#1f77b4", label="Source", alpha=0.25, s=3)
                plt.scatter(data.x1[:5000, 0].cpu().numpy(), data.x1[:5000, 1].cpu().numpy(), c="#ff7f0e", label="Target", alpha=0.25, s=3)
                plt.scatter(xt[:5000,0].cpu().numpy(), xt[:5000,1].cpu().numpy(), c="#2ca02c", label=f'Gen SWD={distance:.3f}', alpha=0.25, s=3)
            
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            if data.dim == 1:
                # plt.gca().get_xaxis().set_visible(False)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlabel('Space', fontsize=16)
                plt.ylabel('Density', fontsize=16)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                # plt.gca().spines['bottom'].set_visible(False)
            else:
                plt.axis('off')
            
            plt.legend(fontsize=16, framealpha=0.5, loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, f"xdist_{ckpt_name}_{N_list}.png"), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_enum("data_type", None, ["1to2", "1to5", "2D1to6", "moon", "3to3", "scurve", "2to2", "tree"], "data type")
    flags.DEFINE_integer("batchsize", 5000, "batch size")
    flags.DEFINE_integer("iter", 50000, "training iterations")
    flags.DEFINE_integer("gpu", 0, "GPU number")
    flags.DEFINE_integer("seed", 0, "random seed")
    flags.DEFINE_string("base_dir", "lowdim", "work dir")
    flags.DEFINE_enum("mode", None, ["train", "eval"], "running mode")
    

    app.run(main)




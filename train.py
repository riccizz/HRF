import copy
import os

import torch
from absl import app, flags
from torch.utils import tensorboard
from tqdm import trange

from dataset import get_datalooper
from model import get_model
from utils import ema, generate_samples, load_model
from cfm import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)


FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "./", help="output directory")
flags.DEFINE_string("imagenet_root", "./", help="root directory for imagenet")
flags.DEFINE_string("exp_name", "base", help="experiment name")
flags.DEFINE_enum("dataset", "cifar10", ["cifar10", "mnist", "imagenet32"], help="dataset name")
flags.DEFINE_bool("hrf", False, help="train hrf or baseline")
flags.DEFINE_integer("gpu", 0, help="GPU number")

# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")
flags.DEFINE_list("channel_mult", [1, 2, 2, 2], help="channel_mult of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("ot_bs", 128, help="optimal transport batch size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("continue_train", False, help="continue training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)
flags.DEFINE_integer(
    "tb_step",
    50,
    help="frequency of saving loss to tensorboard",
)


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step, mix method, before middle block:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
        FLAGS.tb_step,
    )

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{FLAGS.gpu}" if use_cuda else "cpu")

    savedir = os.path.join(FLAGS.output_dir, f"results_{FLAGS.dataset}", f"{FLAGS.exp_name}")
    os.makedirs(savedir, exist_ok=True)
    ckptdir = os.path.join(savedir, "ckpt")
    os.makedirs(ckptdir, exist_ok=True)
    imgdir = os.path.join(savedir, "img_train")
    os.makedirs(imgdir, exist_ok=True)
    writer = tensorboard.SummaryWriter(savedir)

    datalooper, data_shape = get_datalooper(
        FLAGS.dataset, 
        FLAGS.batch_size, 
        FLAGS.num_workers, 
        train=True, 
        imagenet_root=FLAGS.imagenet_root,
    )

    unet = get_model(
        FLAGS.dataset,
        data_shape,
        FLAGS.channel_mult,
        FLAGS.num_channel,
        device,
        hrf=FLAGS.hrf,
    )
    
    model_size = 0
    for param in unet.parameters():
        model_size += param.data.nelement()
    print(f"Model params number: {model_size}")
    print("Model params: %.2f M" % (model_size / 1000 / 1000))

    ema_model = copy.deepcopy(unet)
    optim = torch.optim.Adam(unet.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # continue training
    cur_step = 0
    ckpt_list = os.listdir(ckptdir)
    if FLAGS.continue_train and len(ckpt_list) > 0:
        ckpt = sorted(ckpt_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        print(f"loading {ckpt}")
        ckpt = torch.load(os.path.join(ckptdir, ckpt), weights_only=True)
        load_model(unet, ckpt['model'])
        load_model(ema_model, ckpt['ema_model'])
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])
        cur_step = ckpt['step']

    FM = ConditionalFlowMatcher(sigma=0.0)
    # FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0, ot_bs=FLAGS.ot_bs)

    with trange(cur_step, cur_step + FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, target = FM.sample_location_and_conditional_flow(x0, x1)
            if FLAGS.hrf:
                v0 = torch.randn_like(target)
                tau, vtau, target = FM.sample_location_and_conditional_flow(v0, target)
                pred = unet(tau, vtau, t, xt)
            else:
                pred = unet(t, xt)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(unet, ema_model, FLAGS.ema_decay)  # new
            pbar.set_description(f'loss: {loss.item():.4f}')
            pbar.update(1)
            
            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples(unet, imgdir, step, (16, *data_shape), device, net_="normal", hrf=FLAGS.hrf)
                generate_samples(ema_model, imgdir, step, (16, *data_shape), device, net_="ema", hrf=FLAGS.hrf)
                torch.save(
                    {
                        "model": unet.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    os.path.join(ckptdir, f"{FLAGS.exp_name}_{FLAGS.dataset}_weights_step_{step}.pt"),
                )
            if FLAGS.tb_step > 0 and step % FLAGS.tb_step == 0:
                writer.add_scalar("training_loss", loss, step)


if __name__ == "__main__":
    app.run(train)

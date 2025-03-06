import os

import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags
from torchvision.utils import save_image

from dataset import get_datalooper
from model import get_model
from utils import sample_rf, sample_hrf, load_model

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "./", help="output_directory")
flags.DEFINE_string("imagenet_root", "./", help="root directory for imagenet")
flags.DEFINE_string("exp_name", "exp", help="experiment name")
flags.DEFINE_enum("dataset", "cifar10", ["cifar10", "mnist", "imagenet32"], help="dataset name")
flags.DEFINE_bool("hrf", False, help="train hrf or baseline")
flags.DEFINE_integer("gpu", 0, help="GPU number")

# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")
flags.DEFINE_list("channel_mult", [1, 2, 2, 2], help="channel_mult of UNet")

# Sample
flags.DEFINE_enum("integration_method", "euler", ["euler", "dopri5"], help="integration method to use")


def preprocess_images(images):
    images_resized = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    images_resized = (images_resized.clamp(-1, 1) + 1) / 2
    return images_resized


def eval(argv):
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{FLAGS.gpu}" if use_cuda else "cpu")
    
    savedir = os.path.join(FLAGS.output_dir, f"results_{FLAGS.dataset}", f"{FLAGS.exp_name}")
    ckptdir = os.path.join(savedir, "ckpt")
    imgdir = os.path.join(savedir, "img_eval")
    os.makedirs(imgdir, exist_ok=True)

    _, data_shape = get_datalooper(
        FLAGS.dataset, 
        batch_size=1, 
        num_workers=0, 
        train=False, 
        imagenet_root=FLAGS.imagenet_root,
    )

    with torch.no_grad():
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

        # Load the model
        ckpt = sorted(os.listdir(ckptdir), key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        print(f"loading {ckpt}")
        ckpt = torch.load(os.path.join(ckptdir, ckpt), weights_only=True)
        load_model(unet, ckpt['ema_model'])
        unet.eval()

        sample_shape = (16, *data_shape)
        if FLAGS.hrf:
            generated_img, nfe = sample_hrf(unet, sample_shape, 2, 100, device, FLAGS.integration_method)
            file = f"hrf_{FLAGS.integration_method}_{nfe}.png"
        else:
            generated_img, nfe = sample_rf(unet, sample_shape, 100, device, FLAGS.integration_method)
            file = f"rf_{FLAGS.integration_method}_{nfe}.png"
        save_image(generated_img.clip(-1, 1) / 2 + 0.5, os.path.join(imgdir, file), nrow=4)


if __name__ == "__main__":
    app.run(eval)
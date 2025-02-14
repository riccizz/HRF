from torchcfm.models.unet.unet import UNetModelWrapper as UNetModelWrapperBaseline
from models.unet_cat_xt_v import UNetModelWrapper as UNetModelWrapperCat
from models.unet_2_unet import UNetModelWrapper as UNetModelWrapper2UNet

def get_model(dataset, data_shape, channel_mult, num_channel, device, hrf=True):
    if hrf:
        if dataset == "mnist":
            UNetModelWrapper = UNetModelWrapperCat
        else:
            UNetModelWrapper = UNetModelWrapper2UNet
    else:
        UNetModelWrapper = UNetModelWrapperBaseline
    
    if dataset == "mnist":
        unet = UNetModelWrapper(
            dim=data_shape,
            num_res_blocks=1,
            num_channels=num_channel,
        ).to(
            device
        )
    elif dataset == "cifar10":
        channel_mult = [int(i) for i in channel_mult]
        unet = UNetModelWrapper(
            dim=data_shape,
            num_res_blocks=2,
            num_channels=num_channel,
            channel_mult=channel_mult,
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        ).to(
            device
        )
    elif dataset == "imagenet32":
        channel_mult = [int(i) for i in channel_mult]
        unet = UNetModelWrapper(
            dim=data_shape,
            num_res_blocks=2,
            num_channels=num_channel,
            channel_mult=channel_mult,
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16,8",
            dropout=0.1,
        ).to(
            device
        )
    else:
        raise NotImplementedError
    
    return unet
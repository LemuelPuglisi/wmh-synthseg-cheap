import torch

from wmh_synthseg.unet_3d.model import UNet3D


def init_model(model_checkpoints: str, device: str):
    """
    """    
    n_labels      = 33
    in_channels   = 1
    out_channels  = n_labels + 4 + 1 + 1
    f_maps        = 64
    layer_order   = 'gcl'
    num_groups    = 8
    num_levels    = 5
    
    model = UNet3D(
        in_channels=in_channels, 
        out_channels=out_channels, 
        final_sigmoid=False, 
        f_maps=f_maps, 
        layer_order=layer_order,
        num_groups=num_groups, 
        num_levels=num_levels, 
        is_segmentation=False, 
        is3d=True
    ).to(device)
    
    model.load_state_dict(
        torch.load(
            model_checkpoints, 
            map_location=device,
            weights_only=False
        )['model_state_dict']
    )
    
    return model
    
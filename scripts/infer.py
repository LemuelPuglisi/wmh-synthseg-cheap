import os
import argparse

import torch
import numpy as np
import nibabel as nib
from torch.nn import Softmax
from monai import transforms
from tqdm import tqdm

from wmh_synthseg.initialize import init_model
from wmh_synthseg.consts import label_list_segmentation
from wmh_synthseg.utils import MRIwrite, align_volume_to_ref


def find_ext(path):
    if path.endswith('.nii.gz'):
        return '.nii.gz'
    elif path.endswith('.nii'):
        return '.nii'
    else:
        raise Exception('format not supported.')


@torch.inference_mode()
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',     type=str, required=True,  help='text file where each line is the path of an image to process')
    parser.add_argument('--outputs',    type=str, required=True,  help='text file where each line is the path to an output')
    parser.add_argument("--model",      type=str, required=True,  help="model checkpoints")
    parser.add_argument("--device",     type=str, default='cpu',  help="device (cpu or cuda; optional)")
    parser.add_argument("--threads",    type=int, default=1,      help="(optional) Number of CPU cores to be used. Default is 1. You can use -1 to use all available cores")
    args = parser.parse_args()


    inp_file    = args.inputs
    out_file    = args.outputs
    device      = torch.device(args.device)
    threads     = os.cpu_count() if args.threads < 0 else args.threads
    model_path  = args.model
    device      = torch.device(device)
    
    print('Using ' + args.device)
    print('Using %s thread(s)' % threads)
    torch.set_num_threads(threads)
    
    #==================================
    # run all the sanity checks
    #==================================
    
    assert os.path.exists(inp_file), 'input file doesn\'t exist'
    assert os.path.exists(out_file), 'output file doesn\'t exist'

    print('🚀 reading input files')

    with open(inp_file, 'r') as f:
        inp_list = [ l.strip() for l in f.readlines() ]

    with open(out_file, 'r') as f:
        out_list = [ l.strip() for l in f.readlines() ]
    
    for filepath, output_dir in zip(inp_list, out_list):        
        
        # check if the input file exists
        if not os.path.exists(filepath):
            raise Exception(f'{filepath} does not exist.')
        
        # Try to create the output directory
        os.makedirs(output_dir, exist_ok=True)
    
    #==================================
    # Load the model
    #==================================
    softmax = Softmax(dim=0)
    model   = init_model(model_path, device)
    model.eval()
    
    #========================================
    # params required in WMH-SynthSeg code.
    #========================================
    n_labels            = len(label_list_segmentation)
    n_neutral_labels    = 7
    label_list_segm     = torch.tensor(label_list_segmentation)
    nlat                = int((n_labels - n_neutral_labels) / 2.0)
    vflip               = np.concatenate([
                            np.array(range(n_neutral_labels)),
                            np.array(range(n_neutral_labels + nlat, n_labels)),
                            np.array(range(n_neutral_labels, n_neutral_labels + nlat))
                        ])

    #========================================
    # Loading and running.
    #========================================

    # Since all of my MRIs are registered to the MNI152 space,
    # let's just crop them to (182, 218, 182), and then use the
    # same preprocessing that has been used by the authors.

    MNI152_SHAPE = (182, 218, 182)

    transform_fn = transforms.Compose([
        transforms.LoadImage(),
        transforms.EnsureChannelFirst(channel_dim='no_channel'),
        transforms.ResizeWithPadOrCrop(spatial_size=MNI152_SHAPE),
        transforms.ScaleIntensity()
    ])

    n_inputs = len(inp_list)
    zipped   = zip(inp_list, out_list)

    for nim, (input_file, output_dir) in tqdm(enumerate(zipped), total=n_inputs):
        
        extension = find_ext(input_file)
        filename  = os.path.basename(input_file)
        output_segm_file = os.path.join(output_dir, filename.replace(extension, f'_wmh_segm{extension}'))
        output_prob_file = os.path.join(output_dir, filename.replace(extension, f'_wmh_prob{extension}'))
            
        # assume the image is in MNI152 space
        affine_orig = nib.load(input_file).affine
        image_orig = transform_fn(input_file).squeeze(0)
        image_orig, affine_aligned = align_volume_to_ref(
            image_orig, 
            affine_orig, 
            aff_ref=np.eye(4), 
            return_aff=True, 
            n_dims=3
        )

        with torch.no_grad():
            # Here they upscale to make dimensions divisible by 32 (they have 5 layers)
            upscaled_padded = torch.zeros(tuple((np.ceil(np.array(image_orig.shape) / 32.0) * 32).astype(int)))
            upscaled_padded[:image_orig.shape[0], :image_orig.shape[1], :image_orig.shape[2]] = image_orig                
        
        x_orig = upscaled_padded.unsqueeze(0)
        x_flip = torch.flip(x_orig.squeeze(0).detach(), [0]).unsqueeze(0)
                
        with torch.no_grad():            
            outp_orig = model(x_orig.unsqueeze(0).to(device))
            outp_orig = outp_orig.cpu()
            outp_orig = outp_orig[:, :, :image_orig.shape[0], :image_orig.shape[1], :image_orig.shape[2]]
            del x_orig
                        
        with torch.no_grad():
            outp_flip = model(x_flip.unsqueeze(0).to(device))
            outp_flip = torch.flip(outp_flip, [2]).cpu()
            outp_flip = outp_flip[:, :, :image_orig.shape[0], :image_orig.shape[1], :image_orig.shape[2]]
            del x_flip
            
        with torch.no_grad():
            outp_orig_smax = softmax(outp_orig[0, 0:n_labels, ...])
            outp_flip_smax = softmax(outp_flip[0, vflip, ...])
            pred_seg_p = (.5 * outp_orig_smax + .5 * outp_flip_smax)
            pred_seg = label_list_segm[torch.argmax(pred_seg_p, 0)]
            pred_seg = pred_seg.detach().cpu().numpy()
        
        # save wmh segmentation
        MRIwrite(pred_seg, affine_aligned, output_segm_file)    

        # save wmh probs                
        idx      = label_list_segmentation.index(77)
        lesion_p = pred_seg_p[idx, ...]
        lesion_p = np.squeeze(lesion_p.detach().cpu().numpy())
        MRIwrite(lesion_p, affine_aligned, output_prob_file)


if __name__ == '__main__': main()
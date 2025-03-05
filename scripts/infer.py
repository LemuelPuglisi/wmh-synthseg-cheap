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
from wmh_synthseg.utils import (
    MRIwrite, 
    get_vram_usage, 
    align_volume_to_ref
)

@torch.inference_mode()
def main():

    parser = argparse.ArgumentParser(description="WMH-SynthSeg: joint segmentation of anatomy and white matter hyperintensities ", epilog='\n')
    parser.add_argument("--i", help="Input image or directory.", required=True)
    parser.add_argument("--o", help="Output segmentation (or directory, if the input is a directory)", required=True)
    parser.add_argument("--model", help="Checkpoints path", required=True)
    parser.add_argument("--device", default='cpu', help="device (cpu or cuda; optional)")
    parser.add_argument("--threads", type=int, default=1, help="(optional) Number of CPU cores to be used. Default is 1. You can use -1 to use all available cores")
    parser.add_argument("--save_lesion_probabilities", action="store_true", help="(optional) Saves lesion probability maps")
    parser.add_argument("--crop", action="store_true", help="(optional) Does two passes, to limit size to 192x224x192 cuboid (needed for GPU processing)")
    args = parser.parse_args()

    input_path          = args.i
    output_path         = args.o
    device              = torch.device(args.device)
    threads             = os.cpu_count() if args.threads < 0 else args.threads
    save_lesion_probs   = args.save_lesion_probabilities
    crop                = args.crop
    model_path          = args.model
    device              = torch.device(device)
    
    print('Using ' + args.device)
    print('Using %s thread(s)' % threads)
    torch.set_num_threads(threads)
    
    #==================================
    # run all the sanity checks
    #==================================
    
    if not os.path.exists(input_path): raise Exception('Input does not exist')
    if not os.path.exists(input_path): raise Exception('Model does not exist')

    if os.path.isfile(input_path):    
        images_to_segment = [input_path]
        segmentations_to_write = [output_path]

    if os.path.isdir(input_path):  # directory
        images_to_segment = []
        segmentations_to_write = []
        for im in os.listdir(input_path):
            if im.endswith('.nii') or im.endswith('.nii.gz') or im.endswith('.mgz'):
                images_to_segment.append(os.path.join(input_path, im))
                segmentations_to_write.append(
                    os.path.join(output_path, im.replace('.nii', '_seg.nii').replace('.mgz', '_seg.mgz')))
        if len(images_to_segment) == 0:
            raise Exception('Input directory does not contain images with supported type (.nii, .nii.gz, or .mgz)')
        if output_path.endswith('.nii') or output_path.endswith('.nii.gz') or output_path.endswith('.mgz'):
            raise Exception('If input is a directory, output should be a directory too')
        if os.path.isdir(output_path) is False:
            os.mkdir(output_path)
    
    #==================================
    # run all the sanity checks
    #==================================

    model = init_model(model_path, device)
    model.eval()
    
    softmax = Softmax(dim=0)
    
    #========================================
    # params required in WMH-SynthSeg code.
    #========================================
    label_list_segmentation_torch = torch.tensor(label_list_segmentation)
    
    n_neutral_labels    = 7
    ref_res             = np.array([1.0, 1.0, 1.0])
    voxelsize           = np.prod(ref_res)
    n_labels            = len(label_list_segmentation)

    nlat  = int((n_labels - n_neutral_labels) / 2.0)

    vflip = np.concatenate([np.array(range(n_neutral_labels)),
                            np.array(range(n_neutral_labels + nlat, n_labels)),
                            np.array(range(n_neutral_labels, n_neutral_labels + nlat))])

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

    n_inputs = len(images_to_segment)
    zipped   = zip(images_to_segment, segmentations_to_write)

    for nim, (input_file, output_file) in tqdm(enumerate(zipped), total=n_inputs):
        
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
            pred_seg = label_list_segmentation_torch[torch.argmax(pred_seg_p, 0)]
            pred_seg = pred_seg.detach().cpu().numpy()
        
        MRIwrite(pred_seg, affine_aligned, output_file)    
                
        if save_lesion_probs:
            idx      = label_list_segmentation.index(77)
            lesion_p = pred_seg_p[idx, ...]
            lesion_p = np.squeeze(lesion_p.detach().cpu().numpy())
            if output_file.endswith('.nii'):      name = output_file[:-4] + '.lesion_probs.nii'
            elif output_file.endswith('.nii.gz'): name = output_file[:-7] + '.lesion_probs.nii.gz'
            elif output_file.endswith('.mgz'):    name = output_file[:-4] + '.lesion_probs.mgz'
            MRIwrite(lesion_p, affine_aligned, name)
            
        print('memory usage')
        print(get_vram_usage())
        print(f'{nim} done.')



if __name__ == '__main__': main()
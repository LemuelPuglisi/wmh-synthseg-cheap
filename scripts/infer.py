import os
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="WMH-SynthSeg: joint segmentation of anatomy and white matter hyperintensities ", epilog='\n')
    parser.add_argument("--i", help="Input image or directory.", required=True)
    parser.add_argument("--o", help="Output segmentation (or directory, if the input is a directory)", required=True)
    parser.add_argument("--model", help="Checkpoints path", required=True)
    parser.add_argument("--csv_vols", help="(optional) CSV file with volumes of ROIs")
    parser.add_argument("--device", default='cpu', help="device (cpu or cuda; optional)")
    parser.add_argument("--threads", type=int, default=1, help="(optional) Number of CPU cores to be used. Default is 1. You can use -1 to use all available cores")
    parser.add_argument("--save_lesion_probabilities", action="store_true", help="(optional) Saves lesion probability maps")
    parser.add_argument("--crop", action="store_true", help="(optional) Does two passes, to limit size to 192x224x192 cuboid (needed for GPU processing)")
    args = parser.parse_args()

    input_path          = args.i
    output_path         = args.o
    output_csv_path     = args.csv_vols
    device              = args.device
    threads             = args.threads
    save_lesion_probs   = args.save_lesion_probabilities
    crop                = args.crop
    model_path          = args.model
    
    #==================================
    # run all the sanity checks
    #==================================
    
    if not os.path.exists(input_path): raise Exception('Input does not exist')
    if not os.path.exists(input_path): raise Exception('Model does not exist')

    if  output_csv_path is not None and ( \
        not output_csv_path.endswith('.csv') or \
        not os.path.exists(os.path.basedir(output_csv_path))):
        raise Exception('Something\'s wrong with your CSV file path.')

    if os.path.isfile(input_path):        
        if (input_path.endswith('.nii') or input_path.endswith('.nii.gz') or input_path.endswith('.mgz')) is False:
            raise Exception('Input image is not of a supported type (.nii, .nii.gz, or .mgz)')
        head, tail = os.path.split(output_path)
        if len(tail) == 0:
            raise Exception('If input is a file, output must be a file')
        if (tail.endswith('.nii') or tail.endswith('.nii.gz') or tail.endswith('.mgz')) is False:
            raise Exception('Output image is not of a supported type (.nii, .nii.gz, or .mgz)')
        if ((len(head) > 0) and (os.path.isdir(head) is False)):
            raise Exception('Parent directory of output image does not exist')
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
    # Load libraries and the model
    #==================================
    import torch
    import numpy as np
    from torch.nn import Softmax
    
    from wmh_synthseg.initialize import init_model
    from wmh_synthseg.consts import (
        label_names,
        label_list_segmentation
    )
    from wmh_synthseg.utils import (
        MRIread, 
        MRIwrite, 
        myzoom_torch, 
        align_volume_to_ref
    )
    
    device  = torch.device(device)    
    threads = os.cpu_count() if threads < 0 else threads

    print('Using ' + args.device)
    print('Using %s thread(s)' % threads)
    torch.set_num_threads(threads)

    model_file = "/home/lpuglisi/projects/preprocessing/wmh-synthseg-multigpu/checkpoints/WMH-SynthSeg_v10_231110.pth"   
    label_list_segmentation_torch = torch.tensor(label_list_segmentation, device=device)
        
    model = init_model(model_file, device)
    model.eval()
    
    #==================================
    # Load libraries and the model
    #==================================
    
    n_ims = len(images_to_segment)
    if output_csv_path is not None:
        csv = open(output_csv_path, 'w')
        csv.write('Input-file,Intracranial-volume')
        for l in range(len(label_names)):
            lab = label_list_segmentation[l]
            if lab > 0:
                name = label_names[l]
                csv.write(',' + name + '(' + str(lab) + ')')
        csv.write('\n')


    #========================================
    # params required in WMH-SynthSeg code.
    #========================================
    n_neutral_labels    = 7
    ref_res             = np.array([1.0, 1.0, 1.0])
    voxelsize           = np.prod(ref_res)
    n_labels            = len(label_list_segmentation)

    for nim in range(n_ims):
        
        input_file  = images_to_segment[nim]
        output_file = segmentations_to_write[nim]
        print('Working on image ' + str(1 + nim) + ' of ' + str(+n_ims) + ': ' + input_file)

        try:
            
            #========================================
            # Loading the image.
            #========================================
            print('\tLoading input volume and normalizing to [0,1]')
            image, aff = MRIread(input_file)
            image_torch = torch.tensor(np.squeeze(image).astype(float), device='cpu')
            while len(image_torch.shape) > 3:
                image_torch = image_torch.mean(image, dim=-1)
            image_torch, aff2 = align_volume_to_ref(image_torch, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
            image_torch = image_torch / torch.max(image_torch)

            #=============================================
            # Upscaling the image to the target resolution
            #=============================================
            print('\tUpscaling to target resolution')
            voxsize       = np.sqrt(np.sum(aff2 ** 2, axis=0))[:-1]
            factors       = voxsize / ref_res
            upscaled      = myzoom_torch(image_torch, factors, device='cpu')
            aff_upscaled  = aff2.copy()
            
            for j in range(3):
                aff_upscaled[:-1, j] = aff_upscaled[:-1, j] / factors[j]
            
            aff_upscaled[:-1, -1] = aff_upscaled[:-1, -1] - np.matmul(aff_upscaled[:-1, :-1], 0.5 * (factors - 1))
            
            if crop:
                siz_c = (np.ceil(np.array(upscaled.shape) / 32.0) * 32).astype(int)
                siz_c[0] = min(192, siz_c[0])
                siz_c[1] = min(224, siz_c[1])
                siz_c[2] = min(192, siz_c[2])
                
                cuboid = torch.zeros(tuple(siz_c), device='cpu')
                
                if upscaled.shape[0] <= 192:
                    i1i = 0;
                    i2i = upscaled.shape[0];
                    i1o = np.floor((192 - upscaled.shape[0]) / 2).astype(int);
                    i2o = i1o + upscaled.shape[0];
                else:
                    i1i = np.floor((upscaled.shape[0] - 192) / 2).astype(int);
                    i2i = i1i + 192
                    i1o = 0;
                    i2o = 192
                if upscaled.shape[1] <= 224:
                    j1i = 0;
                    j2i = upscaled.shape[1];
                    j1o = np.floor((224 - upscaled.shape[1]) / 2).astype(int);
                    j2o = j1o + upscaled.shape[1];
                else:
                    j1i = np.floor((upscaled.shape[1] - 224) / 2).astype(int);
                    j2i = j1i + 224
                    j1o = 0;
                    j2o = 224
                if upscaled.shape[2] <= 192:
                    k1i = 0;
                    k2i = upscaled.shape[2];
                    k1o = np.floor((192 - upscaled.shape[2]) / 2).astype(int);
                    k2o = k1o + upscaled.shape[2];
                else:
                    k1i = np.floor((upscaled.shape[2] - 192) / 2).astype(int);
                    k2i = k1i + 192
                    k1o = 0;
                    k2o = 192
                    
                cuboid[i1o:i2o, j1o:j2o, k1o:k2o] = upscaled[i1i:i2i, j1i:j2i, k1i:k2i]

                #=============================================
                # Determining the center of the brain... 
                #=============================================

                with torch.no_grad():

                    print('\tPreliminary pass to determine center of brain')
                    
                    pred    = model(cuboid[None, None, ...].to(device))
                    seg_p   = Softmax(dim=0)(pred[0, 0:n_labels, ...].detach().cpu())
                    
                    p_th_lv = seg_p[label_list_segmentation.index(10)] \
                            + seg_p[label_list_segmentation.index(49)] \
                            + seg_p[label_list_segmentation.index(4)]  \
                            + seg_p[label_list_segmentation.index(43)]
                    
                    vi = torch.tensor(range(192), device='cpu')
                    vj = torch.tensor(range(224), device='cpu')
                    vk = torch.tensor(range(192), device='cpu')
                    
                    gi, gj, gk = torch.meshgrid(vi,vj,vk, indexing='ij')
                    
                    den = torch.sum(p_th_lv)
                    
                    ic = max(0, int(torch.sum(p_th_lv * gi) / den + i1i - 192/2))
                    jc = max(0, int(torch.sum(p_th_lv * gj) / den + j1i - 224/2))
                    kc = max(0, int(torch.sum(p_th_lv * gk) / den + k1i - 192/2))
                    
                    upscaled = upscaled[ic:, jc:, kc:]
                    aff_upscaled[:-1, -1] = aff_upscaled[:-1, -1] + np.matmul(aff_upscaled[:-1, :-1], np.array([ic, jc, kc]))
                    upscaled = upscaled[:min(192,upscaled.shape[0]), :min(224,upscaled.shape[1]), :min(192,upscaled.shape[2])]

            #=============================================
            # Finally predicting the segmentation
            #=============================================

            print('\tPushing data through the CNN')
            
            with torch.no_grad():
                
                upscaled_padded = torch.zeros(tuple((np.ceil(np.array(upscaled.shape) / 32.0) * 32).astype(int)),  device=device)
                upscaled_padded[:upscaled.shape[0], :upscaled.shape[1], :upscaled.shape[2]] = upscaled                
                pred1 = model(upscaled_padded[None, None, ...])[:, :, :upscaled.shape[0], :upscaled.shape[1], :upscaled.shape[2]]
                pred2 = torch.flip(model(torch.flip(upscaled_padded, [0])[None, None, ...]), [2])[:, :, :upscaled.shape[0], :upscaled.shape[1], :upscaled.shape[2]]

                softmax = Softmax(dim=0)
                nlat = int((n_labels - n_neutral_labels) / 2.0)
                vflip = np.concatenate([np.array(range(n_neutral_labels)),
                                        np.array(range(n_neutral_labels + nlat, n_labels)),
                                        np.array(range(n_neutral_labels, n_neutral_labels + nlat))])

                pred_seg_p = 0.5 * softmax(pred1[0, 0:n_labels, ...]) + 0.5 * softmax(pred2[0, vflip, ...])
                pred_seg = label_list_segmentation_torch[torch.argmax(pred_seg_p, 0)]
                pred_seg = np.squeeze(pred_seg.detach().cpu().numpy())
            
                #================================================
                # Write volumes from soft segmentation, if needed
                #================================================
                
                vols = voxelsize * torch.sum(pred_seg_p, dim=[1, 2, 3]).detach().cpu().numpy()
                
                if output_csv_path is not None:
                    # Subject name and ICV
                    csv.write(output_file + ',' + str(np.sum(vols[1:])))
                    # volumes of structures
                    for l in range(len(label_list_segmentation)):
                        if label_list_segmentation[l] > 0:
                            csv.write(',' + str(vols[l]))
                    csv.write('\n')

            #================================================
            # Save the segmentation on disk
            #================================================

            print('\tWriting segmentation to disk: ' + output_file)
            MRIwrite(pred_seg, aff_upscaled, output_file)
            
            if save_lesion_probs:
                idx      = label_list_segmentation.index(77)
                lesion_p = pred_seg_p[idx, ...]
                lesion_p = np.squeeze(lesion_p.detach().cpu().numpy())
                if output_file.endswith('.nii'):      name = output_file[:-4] + '.lesion_probs.nii'
                elif output_file.endswith('.nii.gz'): name = output_file[:-7] + '.lesion_probs.nii.gz'
                elif output_file.endswith('.mgz'):    name = output_file[:-4] + '.lesion_probs.mgz'
                MRIwrite(lesion_p, aff_upscaled, name)
            print('\n')


        except Exception as e:
            print("\n\tAn error occurred in this volume:", str(e)); print(' ')
            
    #================================================
    # Write CSV file to disk
    #================================================
    if output_csv_path is not None: csv.close()

    #================================================
    # Print credicts
    #================================================  

    print('All done!')
    print('If you use this method in a publication, please cite the following article:')
    print(
        'Quantifying white matter hyperintensity and brain volumes in heterogeneous clinical and low-field portable MRI')
    print(
        'Laso P, Cerri S, Sorby-Adams A, Guo J, Matteen F, Goebl P, Wu J, Li H, Young SI, Billot B, Puonti O, Rosen MS,')
    print('Kirsch J, Strisciuglio N, Wolterink JM, Eshaghi A, Barkhof F, Kimberly WT, and Iglesias JE')
    print('Under review.')
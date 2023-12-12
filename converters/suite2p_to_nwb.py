import os

import numpy as np
import yaml
from pynwb.ophys import Fluorescence, ImageSegmentation

from utils.server_paths import get_suite2p_folder
import utils.gf_utils as utils_gf
import utils.ci_processing as utils_ci


def convert_suite2p_data(nwb_file, config_file, ci_frame_timestamps):
    """
    :param nwb_file: nwb file
    :param config_file : general config file allowing to reconstruct path to suite2p folder
    :param ci_frame_timestamps: timestamps of each imaging frame obtained from analyse of continuous log
    :return:
    """

    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    experimenter = config['session_metadata']['experimenter']

    # First check whether there is data to add in the ophys module
    if experimenter in ['GF', 'MI']:
        suite2p_folder = utils_gf.check_gf_suite2p_folder(config_file)
    else:
        suite2p_folder = get_suite2p_folder(config_file)
    if suite2p_folder is None:
        print("No suite2p folder to add")
        return

    if 'ophys' in nwb_file.processing:
        module = nwb_file.processing['ophys']
    else:
        module = nwb_file.create_processing_module('ophys', 'contains optical physiology processed data')

    image_series = nwb_file.acquisition.get("motion_corrected_ci_movie")
    if image_series is None:
        print("No calcium imaging movie named 'motion_corrected_ci_movie' found in nwb_file")

    img_seg = ImageSegmentation(name="all_cells")
    module.add_data_interface(img_seg)
    imaging_plane = nwb_file.get_imaging_plane("my_imaging_plane")

    ps = img_seg.create_plane_segmentation(description='output from segmenting',
                                        imaging_plane=imaging_plane, name='my_plane_segmentation',
                                        reference_images=image_series)

    # Load Suite2p data
    F = None
    dcnv = None

    if experimenter not in ['GF', 'MI']:
        suite2p_folder = os.path.join(suite2p_folder, "plane0")
        if not os.path.isfile(os.path.join(suite2p_folder, "stat.npy")):
            print(f"Stat file missing in {suite2p_folder}")
            return
        else:
            stat = np.load(os.path.join(suite2p_folder, "stat.npy"), allow_pickle=True)
            is_cell = np.load(os.path.join(suite2p_folder, "iscell.npy"), allow_pickle=True)
            F = np.load(os.path.join(suite2p_folder, "F.npy"), allow_pickle=True)
            Fneu = np.load(os.path.join(suite2p_folder, "Fneu.npy"), allow_pickle=True)
            dcnv = np.load(os.path.join(suite2p_folder, "spks.npy"), allow_pickle=True)
    else:
        stat, is_cell, F, Fneu, dcnv, F_fissa = utils_gf.get_gf_processed_ci(config_file)

    # Compute F0 and dff.
    print('Computing F0 dff.')
    fs = config['log_continuous_metadata']['scanimage_dict']['theoretical_ci_sampling_rate']
    F0, dff = utils_ci.compute_dff(F, Fneu, fs=fs, window=60)
    # Fissa is substracted but not normalized.
    dff_fissa = F_fissa / F0
        
    # Extract image dimensions
    if image_series is not None:
        dim_y, dim_x = image_series.dimension[1:]
    else:
        dim_y, dim_x = 512, 512  # GF case.

    # Add suite2p pixel mask
    pixel_masks = []
    n_cells = 0
    for cell in np.arange(len(stat)):
        if is_cell[cell][0] == 0:
            continue
        n_cells += 1
        pix_mask = [(y, x, 1) for x, y in zip(stat[cell]["xpix"], stat[cell]["ypix"])]
        image_mask = np.zeros((dim_y, dim_x))  # recently inverted (12th of may 2020)
        for pix in pix_mask:
            image_mask[int(pix[0]), int(pix[1])] = pix[2]  # pix[0] pix[1] recently inverted (12th of may 2020)
        # we can id to identify the cell (int) otherwise it will be incremented at each step
        pixel_masks.append(pix_mask)
        ps.add_roi(pixel_mask=pix_mask, image_mask=image_mask)

    fl = Fluorescence(name="fluorescence_all_cells")
    module.add_data_interface(fl)
    rt_region = ps.create_roi_table_region('all cells', region=list(np.arange(n_cells)))

    if experimenter in ['GF', 'MI']:
        data = [F, Fneu, dcnv, F0, F_fissa, dff, dff_fissa]
        data_labels = ['F', 'Fneu', 'dcnv', 'F0', 'F_fissa', 'dff', 'dff_fissa']
        descriptions = ['F: Suite 2P raw fluoresence.',
                        'Fneu: Suite 2P neuropil.',
                        'spks: Suite 2P deconvolved fluorescence.',
                        'F0: 5% percentile baseline computed over a 2 min rolling window.',
                        'Fissa output.',
                        'dF/F0: Normalized neuropil corrected suite2p fluorescence.',
                        'dF_fissa/F0: Normalized fissa output, with F0 of original data.',]
    else:
        data = [F, Fneu, dcnv]
        data_labels = ['F', 'Fneu', 'dcnv', 'F0', 'dff']
        descriptions = ['F: Suite 2P raw fluoresence.',
                        'Fneu: Suite 2P neuropil.',
                        'spks: Suite 2P deconvolved fluorescence.',
                        'F0: 5% percentile baseline computed over a 2 min rolling window.',
                        'dF/F0: Normalized neuropil corrected suite2p fluorescence.',]

    for d, l, desc in zip(data, data_labels, descriptions):
        if d is not None:
            # Filtering is already done for GF data.
            if experimenter in ['GF', 'MI']:
                d_filt = d
            else:
                d_filt = d[is_cell[:,0].astype(bool)]

            fl.create_roi_response_series(name=l,
                                        data=np.transpose(d_filt),
                                        unit='lumens',
                                        rois=rt_region, timestamps=ci_frame_timestamps,
                                        description=desc)

            print(f"- Creating Roi Response Series with: {desc}"
                  f"shape: {(np.transpose(d_filt)).shape}")
        

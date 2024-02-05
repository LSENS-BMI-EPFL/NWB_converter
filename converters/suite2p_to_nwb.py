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

    # Create the 'ophys module'
    if 'ophys' in nwb_file.processing:
        ophys_module = nwb_file.processing['ophys']
    else:
        ophys_module = nwb_file.create_processing_module('ophys', 'contains optical physiology processed data')

    image_series = nwb_file.acquisition.get("motion_corrected_ci_movie")
    if image_series is None:
        print("No calcium imaging movie named 'motion_corrected_ci_movie' found in nwb_file")

    img_seg = ImageSegmentation(name="all_cells")
    ophys_module.add_data_interface(img_seg)
    imaging_plane = nwb_file.get_imaging_plane("my_imaging_plane")

    ps = img_seg.create_plane_segmentation(description='output from segmenting',
                                           imaging_plane=imaging_plane, name='my_plane_segmentation',
                                           reference_images=image_series)

    # Load Suite2p data
    print('Load suite2p data.')
    if experimenter not in ['GF', 'MI']:
        stat, is_cell, F, Fneu, dcnv = utils_ci.get_processed_ci(suite2p_folder)
    else:
        stat, is_cell, F, Fneu, dcnv, F_fissa = utils_gf.get_gf_processed_ci(config_file)

    # Correct is_cell for merges.
    is_cell = utils_ci.set_merged_roi_to_non_cell(stat, is_cell)

    # Compute F0 and dff.
    print('Compute F0 and dff.')
    fs = config['log_continuous_metadata']['scanimage_dict']['theoretical_ci_sampling_rate']
    F0, dff = utils_ci.compute_dff(F, Fneu, fs=fs, window=60)
    # Fissa is substracted but not normalized.
    if F_fissa is not None:
        dff_fissa = F_fissa / F0

    # Extract image dimensions
    if image_series is not None:
        dim_y, dim_x = image_series.dimension[1:]
    else:
        dim_y, dim_x = 512, 512  # GF case.

    # Add suite2p pixel mask of 'real cells' (is_cell == 1)
    print('Add cell masks.')
    utils_ci.add_suite2p_roi(ps, stat, is_cell, dim_x, dim_y)

    # Create Fluorescence object to store fluorescence data
    fl = Fluorescence(name="fluorescence_all_cells")
    ophys_module.add_data_interface(fl)
    n_cells = F[is_cell[:, 0].astype(bool)].shape[0]
    rt_region = ps.create_roi_table_region('all cells', region=list(np.arange(n_cells)))

    # List fluorescence data to save
    if experimenter in ['GF', 'MI']:
        data = [F, Fneu, dcnv, F0, F_fissa, dff, dff_fissa]
        data_labels = ['F', 'Fneu', 'dcnv', 'F0', 'F_fissa', 'dff', 'dff_fissa']
        descriptions = ['F: Suite 2P raw fluoresence.',
                        'Fneu: Suite 2P neuropil.',
                        'spks: Suite 2P deconvolved fluorescence.',
                        'F0: 5% percentile baseline computed over a 2 min rolling window.',
                        'Fissa output.',
                        'dF/F0: Normalized neuropil corrected suite2p fluorescence.',
                        'dF_fissa/F0: Normalized fissa output, with F0 of original data.']
    else:
        data = [F, Fneu, dcnv, F0, dff]
        data_labels = ['F', 'Fneu', 'dcnv', 'F0', 'dff']
        descriptions = ['F: Suite 2P raw fluoresence.',
                        'Fneu: Suite 2P neuropil.',
                        'spks: Suite 2P deconvolved fluorescence.',
                        'F0: 5% percentile baseline computed over a 2 min rolling window.',
                        'dF/F0: Normalized neuropil corrected suite2p fluorescence.']

    # Add information about cell type (projections, ... ).
    # ####################################################

    projection_folder = get_rois_label_folder(config_file)
    
    if not projection_folder:
        cell_type_names = None
        cell_type_codes = None
    else:
        far_red_rois, red_rois, un_rois, info = utils_ci.get_roi_labels(config_file)

        # Code: 1: wM1, 2: wS2 and 0: unassigned.
        # Cell code list [1, 1, 2, 0, 0, 1, 2, 0, 0] same length as d_filt.
        # Cell type list: ["M1", "M1", "S2", "UN", "UN", "M1", "S2", "UN", "UN"].
        projection_code = {'na': 0, 'wM1': 1, 'wS2': 2}
        cell_type_codes = [0 for i in range(n_cells)] 
        cell_type_names = ['na' for i in range(n_cells)]
        for iroi in range(n_cells):
            # Far red.
            if iroi in far_red_rois:
                cell_type_codes[iroi] = projection_code[info['CTB-647']]
                cell_type_names[iroi] = info['CTB-647']
            # Red.
            if iroi in red_rois:
                cell_type_codes[iroi] = projection_code[info['CTB-594']]
                cell_type_names[iroi] = info['CTB-594']
            
    # Add fluorescence data to roi response series.
    print('Add Roi Response Series.')
    # todo : add control (list of int code for cell type) and control_description (list of str for name of cell type)
    for d, l, desc in zip(data, data_labels, descriptions):
        if d is not None:
            # Filtering is already done for GF data.
            if experimenter in ['GF', 'MI']:
                d_filt = d
            else:
                d_filt = d[is_cell[:, 0].astype(bool)]
            if cell_type_codes is not None and cell_type_names is not None:
                fl.create_roi_response_series(name=l,
                                            data=np.transpose(d_filt),
                                            unit='lumens',
                                            rois=rt_region, timestamps=ci_frame_timestamps,
                                            description=desc,
                                            control=cell_type_codes,
                                            control_description=cell_type_names)
            else:
                fl.create_roi_response_series(name=l,
                                            data=np.transpose(d_filt),
                                            unit='lumens',
                                            rois=rt_region, timestamps=ci_frame_timestamps,
                                            description=desc)
            print(f"- Creating Roi Response Series with: {desc}"
                f"shape: {(np.transpose(d_filt)).shape}")
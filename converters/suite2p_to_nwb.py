import os

import numpy as np
import yaml
from pynwb.ophys import Fluorescence, ImageSegmentation

import utils.server_paths as server_paths
import utils.utils_gf as utils_gf
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
    # if experimenter in ['GF', 'MI']:
    #     suite2p_folder = utils_gf.check_gf_suite2p_folder(config_file)
    # else:
    suite2p_folder = server_paths.get_suite2p_folder(config_file)
    if suite2p_folder is None:
        print("No suite2p folder to add")
        return

    # Create the 'ophys module'
    if 'ophys' in nwb_file.processing:
        ophys_module = nwb_file.processing['ophys']
    else:
        ophys_module = nwb_file.create_processing_module('ophys', 'contains optical physiology processed data')

    # image_series = nwb_file.acquisition.get("motion_corrected_ci_movie")
    # TODO: add registered movie
    image_series = None
    if image_series is None:
        print("No calcium imaging movie named 'motion_corrected_ci_movie' found in nwb_file")

    img_seg = ImageSegmentation(name="all_cells")
    ophys_module.add_data_interface(img_seg)
    imaging_plane = nwb_file.get_imaging_plane("my_imaging_plane")

    ps = img_seg.create_plane_segmentation(description='output from segmenting',
                                           imaging_plane=imaging_plane, name='my_plane_segmentation',
                                           reference_images=image_series)

    # Load Suite2p data
    print('Loading suite2p data.')
    # if experimenter not in ['GF', 'MI']:
    #     stat, is_cell, F_raw, F_neu, F0, spks  = utils_ci.get_processed_ci(suite2p_folder)
    # else:
        # stat, is_cell, F_raw, F_neu, F0, spks = utils_gf.get_gf_processed_ci(config_file)

    # Assumes that non-cells are already filtered out.
    stat, is_cell, F_raw, F_neu, F0_cor, F0_raw, dff = utils_ci.get_processed_ci(suite2p_folder)

    # # If Fissa did not converge, set cell to non-cell.
    # if fissa_output:
    #     ncells, ntifs = fissa_output.result.shape
    #     converged = []
    #     for icell in range(ncells):
    #         converged.append(np.all([exp.info[icell][itif]['converged'] for itif in range(ntifs)]))
    #     is_cell[is_cell[:,0]==1,0] = converged
    #     print(f"A total of {np.sum(~converged)} cells did not converge in Fissa. Set as non-cells.")

    # if experimenter in ['GF', 'MI']:
    #     # Fissa substracts baseline but do not normalized.
    #     # Normalizing with baseline of the raw signal.
    #     dff = F_cor / F0

    # Extract image dimensions
    if image_series is not None:
        dim_y, dim_x = image_series.dimension[1:]
    else:
        dim_y, dim_x = 512, 512

    # Add suite2p pixel mask of 'real cells' (is_cell == 1)
    print('Add cell masks.')
    utils_ci.add_suite2p_roi(ps, stat, is_cell, dim_x, dim_y)

    # Create Fluorescence object to store fluorescence data
    fl = Fluorescence(name="fluorescence_all_cells")
    ophys_module.add_data_interface(fl)
    n_cells = (is_cell[:, 0] == 1).sum()
    rt_region = ps.create_roi_table_region('all cells', region=list(np.arange(n_cells)))

    # List fluorescence data to save
    data = [F_raw, F_neu, F0_cor, F0_raw, dff]
    data_labels = ["F_raw", "F_neu", "F0_cor", "F0_raw", "dff",]
    descriptions = ["F_raw: raw fluorescence traces extracted by Suite2p",
                    "F_neu: neuropil fluorescence traces extracted by Suite2p",
                    'F0_cor: 5% percentile baseline computed over a 2 min rolling window of F_raw - 0.7 * F_neu.',
                    'F0_raw: 5% percentile baseline computed over a 2 min rolling window of F_raw.',
                    'dff: Normalized fissa output, dff = (F_raw - 0.7 * F_neu) - F0_cor / F0_raw.']

    # Add information about cell type (projections, ... ).
    # ####################################################

    if experimenter in ['GF', 'MI']:
        projection_folder = utils_gf.get_rois_label_folder_GF(config_file)
    else:
        projection_folder = server_paths.get_rois_label_folder(config_file)

    if not projection_folder:
        cell_type_names = None
        cell_type_codes = None
    else:
        # It is assumed that that cell type indices are not the suite2p indices, but the indices reindexed
        # after filtering out non-cells.
        if experimenter in ['GF', 'MI']:
            far_red_rois, red_rois, na_rois, info = utils_gf.get_roi_labels_GF(config_file, projection_folder)
        else:
            far_red_rois, red_rois, na_rois, info = utils_ci.get_roi_labels(projection_folder)

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
    # #############################################

    print('Add Roi Response Series.')
    # todo : add control (list of int code for cell type) and control_description (list of str for name of cell type)
    for d, l, desc in zip(data, data_labels, descriptions):
        if d is not None:

            if cell_type_codes is not None and cell_type_names is not None:
                fl.create_roi_response_series(name=l,
                                            data=np.transpose(d),
                                            unit='lumens',
                                            rois=rt_region, timestamps=ci_frame_timestamps,
                                            description=desc,
                                            control=cell_type_codes,
                                            control_description=cell_type_names)
            else:
                fl.create_roi_response_series(name=l,
                                            data=np.transpose(d),
                                            unit='lumens',
                                            rois=rt_region, timestamps=ci_frame_timestamps,
                                            description=desc)
            print(f"- Creating Roi Response Series with: {desc}"
                f"shape: {(np.transpose(d)).shape}")




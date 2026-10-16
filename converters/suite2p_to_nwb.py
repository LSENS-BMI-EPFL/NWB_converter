import os
import yaml
import numpy as np
from pynwb.epoch import TimeIntervals
from pynwb.ophys import Fluorescence, ImageSegmentation, RoiResponseSeries


import utils.utils_gf as utils_gf
import utils.ci_processing as utils_ci
import utils.server_paths as server_paths


def add_motion_artifacts(nwb_file, artifact_frames, ci_frame_timestamps):
    """
    Add motion artifacts (artifact_frames.npy from compute_dff_updated.py, one bool per frame) as a
    'motion_artifacts' time intervals table. Nothing is added if artifact_frames is None or has no artifact.
    """
    if artifact_frames is None or not np.any(artifact_frames):
        return
    timestamps = np.asarray(ci_frame_timestamps)
    mask = np.asarray(artifact_frames, dtype=bool)[:len(timestamps)]
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1  # first / last artifact frame
    motion_artifacts = TimeIntervals(name='motion_artifacts',
                                     description='Motion artifacts: suite2p bad frames merged and padded '
                                                 '(compute_dff_updated.py). Excluded from F0 and set to NaN in '
                                                 'F_cor_masked / dff_masked.')
    for start, stop in zip(starts, stops):
        motion_artifacts.add_row(start_time=float(timestamps[start]), stop_time=float(timestamps[stop]))
    nwb_file.add_time_intervals(motion_artifacts)
    print(f'Add {len(starts)} motion artifact intervals.')


def convert_suite2p_data(nwb_file, config_file, ci_frame_timestamps,
                         cascade_model_name='GC8s_EXC_30Hz_smoothing50ms_causalkernel'):
    """
    :param nwb_file: nwb file
    :param config_file : general config file allowing to reconstruct path to suite2p folder
    :param ci_frame_timestamps: timestamps of each imaging frame obtained from analyse of continuous log
    :param cascade_model_name: Cascade model whose outputs are read from plane0/<cascade_model_name>/
                               (falls back to plane0/ for older sessions)
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

    # Load Suite2p data
    print('Loading suite2p data.')
    # Assumes that non-cells are already filtered out.
    ci = utils_ci.get_processed_ci(suite2p_folder, cascade_model_name=cascade_model_name)
    if ci is None:
        print('Suite 2p data is not processed, no dff yet. Return')
        return
    stat, is_cell, dff = ci['stat'], ci['iscell'], ci['dff']
    spike_rate, discrete_spikes = ci['spike_rate'], ci['discrete_spikes']

    # Infer imaging rate from timestamps and convert spike rate from number of spikes within bin to Hz
    imaging_rate = np.round(1 / (np.median(np.diff(ci_frame_timestamps[0:200]))), 2)
    if spike_rate is not None:
        spike_rate = spike_rate * imaging_rate

    # From discrete spikes to spike raster (Cascade discrete spikes are lists of frame index with spikes for each neuron)
    n_cells = (is_cell[:, 0] == 1).sum()
    if discrete_spikes is not None:
        spike_counts = np.zeros((n_cells, dff.shape[1]), dtype=np.int16)
        for cell, frames in enumerate(discrete_spikes):
            frames = np.asarray(frames, dtype=int) + 1                    # correct the 1-frame offset
            frames = frames[(frames >= 0) & (frames < dff.shape[1])]      # safety check
            np.add.at(spike_counts[cell], frames, 1)                      # counts repeated frames correctly
    else:
        spike_counts = None

    add_motion_artifacts(nwb_file, ci['artifact_frames'], ci_frame_timestamps)

    # Only if we are going to add data
    img_seg = ImageSegmentation(name="all_cells")
    ophys_module.add_data_interface(img_seg)
    imaging_plane = nwb_file.get_imaging_plane("my_imaging_plane")

    ps = img_seg.create_plane_segmentation(description='output from segmenting',
                                           imaging_plane=imaging_plane, name='my_plane_segmentation',
                                           reference_images=image_series)
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

    # Create rt region (dynamical table to store roi information)
    rt_region = ps.create_roi_table_region('all cells', region=list(np.arange(n_cells)))

    # List fluorescence data to save. Series whose file is missing (None) are skipped below.
    # compute_dff_updated.py writes F_cor and F0, compute_dff.py (older sessions) writes F0_cor and F0_raw.
    if ci['F0'] is not None:
        dff_series = [
            (ci['F_cor'], "F_cor", "F_cor: neuropil-corrected fluorescence, F_raw - 0.7 * F_neu (not clipped)."),
            (ci['F0'], "F0", "F0: baseline of F_cor, rolling low percentile after gaussian smoothing, "
                             "motion artifact frames excluded (compute_dff_updated.py)."),
            (dff, "dff", "dff: (F_cor - F0) / F0, motion artifact frames kept (no NaN)."),
            (ci['F_cor_masked'], "F_cor_masked", "F_cor_masked: F_cor with motion artifact frames set to NaN."),
            (ci['dff_masked'], "dff_masked", "dff_masked: dff with motion artifact frames set to NaN."),
        ]
    else:
        dff_series = [
            (ci['F0_cor'], "F0_cor", 'F0_cor: 5% percentile baseline computed over a 2 min rolling window of F_raw - 0.7 * F_neu.'),
            (ci['F0_raw'], "F0_raw", 'F0_raw: 5% percentile baseline computed over a 2 min rolling window of F_raw.'),
            (dff, "dff", 'dff: Normalized fissa output, dff = (F_raw - 0.7 * F_neu) - F0_cor / F0_raw.'),
        ]
    series = [(ci['F_raw'], "F_raw", "F_raw: raw fluorescence traces extracted by Suite2p"),
              (ci['F_neu'], "F_neu", "F_neu: neuropil fluorescence traces extracted by Suite2p")] + dff_series + [
              (spike_rate, "infered_spike_rate",
               'infered_spike_rate: the deconvolved instantaneous firing rate using Cascade (converted to Hz)'),
              (spike_counts, "infered_spike_count",
               'infered_spike_count: the number of discrete spike at each imaging frame inferred using Cascade')]
    data, data_labels, descriptions = zip(*series)

    # Add information about cell type (projections, ... ).
    # ####################################################

    if experimenter in ['GF', 'MI']:
        projection_folder = utils_gf.get_rois_label_folder_GF(config_file)
    else:
        projection_folder = server_paths.get_rois_label_folder(config_file)

    if not projection_folder:
        cell_type_names = None
        cell_type_codes = None
        info = None
    else:
        # It is assumed that that cell type indices are not the suite2p indices, but the indices reindexed
        # after filtering out non-cells.
        if experimenter in ['GF', 'MI']:
            far_red_rois, red_rois, na_rois, info = utils_gf.get_roi_labels_GF(config_file, projection_folder)
        else:
            far_red_rois, red_rois, na_rois, info = utils_ci.get_roi_labels(projection_folder,
                                                                            default_info_dict = {'CTB-594': 'wM1'})

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

    # Add marker and cell type to the dynamic roi table as custom column.
    # ###################################################################
    if (info is not None) and (cell_type_names is not None):
        print('Add anatomical markers and cell type custom columns to roi table')
        info_inverted = {v: k for k, v in info.items()}
        cell_markers = [info_inverted.get(cell, 'na') for cell in cell_type_names]
        rt_region.table.add_column(name='markers', data=cell_markers, description='anatomical marker of the cells')
        rt_region.table.add_column(name='cell_type', data=cell_type_names, description='type of the cells')
    
    # Add per-cell QC metrics (<name>.npy from compute_dff_updated.py, one value per cell) to the dynamic roi table.
    # #############################################################################################################

    qc_columns = [
        ('snr', 'SNR: 99.9th percentile of dff / frame noise of dff.'),
        ('neuropil_ratio', 'Share of the baseline removed by neuropil subtraction: 0.7 * median(F_neu) / '
                           'median(F_raw). > ~0.8: F_cor baseline collapsed, dff unreliable.'),
        ('neuropil_f_rho', 'Correlation of F_raw and F_neu fluctuations, band-passed 0.5-30 s.'),
        ('contamination', 'Neuropil contamination: neuropil_ratio * neuropil_f_rho. ~0: clean, > ~0.8: mostly '
                          'neuropil. Negative when neuropil_f_rho < 0, check neuropil_ratio too.'),
        ('motion_r', 'Correlation of dff with the registration correlation corrXY, drift removed. Large |r| '
                     '(either sign): dff follows motion frame by frame (motion artifact).'),
    ]
    for name, description in qc_columns:
        if ci[name] is not None and len(ci[name]) == n_cells:
            rt_region.table.add_column(name=name, data=list(ci[name]),
                                       description=f'{description} (compute_dff_updated.py)')
        else:
            print(f'No {name}.npy matching the number of cells, {name} column not added.')

    
    # Add fluorescence data to roi response series.
    # #############################################

    print('Add Roi Response Series.')
    # todo : add control (list of int code for cell type) and control_description (list of str for name of cell type)
    for d, l, desc in zip(data, data_labels, descriptions):
        if d is not None:
            if  d.shape[1] == len(ci_frame_timestamps):
                ci_frame_timestamps_corr = ci_frame_timestamps
            elif d.shape[1] < len(ci_frame_timestamps):
                print(f'Found {len(ci_frame_timestamps) - d.shape[1]} more timestamps than imaging frames')
                ci_frame_timestamps_corr = ci_frame_timestamps[:-int(len(ci_frame_timestamps) - d.shape[1])]
            else:
                print(f'Found {d.shape[1] - len(ci_frame_timestamps)} more imaging frames than timestamps')
                d = d[:, :len(ci_frame_timestamps)]
                ci_frame_timestamps_corr = ci_frame_timestamps
            if cell_type_codes is not None and cell_type_names is not None:
                roi_resp_series = RoiResponseSeries(
                    name=l,
                    description=desc,
                    data=np.transpose(d),
                    rois=rt_region,
                    unit="lumens",
                    timestamps=ci_frame_timestamps_corr,
                    control=cell_type_codes,
                    control_description=cell_type_names
                )
                fl.add_roi_response_series(roi_resp_series)
            else:
                roi_resp_series = RoiResponseSeries(
                    name=l,
                    description=desc,
                    data=np.transpose(d),
                    rois=rt_region,
                    unit="lumens",
                    timestamps=ci_frame_timestamps_corr
                )
                fl.add_roi_response_series(roi_resp_series)
            print(f"- Adding Roi Response Series with: {desc} "
                f"shape: {(np.transpose(d)).shape}")



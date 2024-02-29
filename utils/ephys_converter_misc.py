#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: NWB_converter
@file: ephys_converter_misc.py
@time: 8/24/2023 9:25 AM
"""

import itertools
import os
import pathlib
import json
import numpy as np
import pandas as pd
import yaml
import re

from utils import server_paths
from utils.continuous_processing import detect_piezo_lick_times
from utils.read_sglx import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ChannelCountsNI

# MAP of (AP,ML) coordinates relative to bregma
AREA_COORDINATES_MAP = {
    'wS1': 'IOS',
    'wS2': 'IOS',
    'A1': 'IOS',
    'wM1': (1, 1),
    'wM2': (2, 1),
    'mPFC': (2, 0.5),
    'Vis': (-3.8, 2.5),
    'PPC': (-2, 1.75),
    'dCA1': (-2.7, 2),
    'tjM1': (2, 2),
    'DLS': (0, 3.5)
}


def get_probe_insertion_info(config_file):
    """
    Read probe insertion information from a metadata external file.
    Args:
        config_file:

    Returns:

    """
    # Read config file
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # This is experimenter-specific tracking of that information
    if config.get('session_metadata').get('experimenter') == 'AB':
        # Load probe insertion table
        path_to_info_file = r'M:\analysis\Axel_Bisi\mice_info\probe_insertion_info.xlsx'
        probe_info_df = pd.read_excel(path_to_info_file)

    else:
        print('No probe insertion information found for this experimenter.')
        raise NotImplementedError

    return probe_info_df


def get_target_location(config_file, device_name):
    """
    Read location target: hemisphere, stereotaxic coordinate, angles from a metadata external file.
    Args:
        config_file: Path to config file
        device_name: Name of the device (e.g. imec0)

    Returns:
    """

    # Read config file
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    location_df = get_probe_insertion_info(config_file=config_file)

    # Keep subset for mouse and probe_id
    mouse_name = config.get('subject_metadata').get('subject_id')
    location_df = location_df[(location_df['mouse_name'] == mouse_name)
                              &
                              (location_df['probe_id'] == int(device_name[-1]))
                              ]

    # Get coordinates of target area
    target_area = location_df['target_area'].values[0]
    if target_area in AREA_COORDINATES_MAP.keys():

        if type(AREA_COORDINATES_MAP[target_area]) is tuple:

            ap = AREA_COORDINATES_MAP[target_area][0]
            ml = AREA_COORDINATES_MAP[target_area][1]

        elif type(AREA_COORDINATES_MAP[target_area]) is str:

            ap = AREA_COORDINATES_MAP[target_area]
            ml = AREA_COORDINATES_MAP[target_area]
        else:
            print('Unknown type for AP, ML coordinates. Setting to NaN')
            ap, ml = (np.nan, np.nan)

    else:
        print('No standard coordinates found for this target area. Setting to NaN')
        ap, ml = (np.nan, np.nan)

    # Create ephys target location dictionary
    location_dict = {
        'hemisphere': 'left',
        'area': location_df['target_area'].values[0],
        'ap': ap,
        'ml': ml,
        'azimuth': location_df['azimuth'].values[0],
        'elevation': location_df['elevation'].values[0],
        'depth': location_df['depth'].values[0],
    }

    return location_dict

def read_ephys_binary_data(bin_file, meta_file):
    """
    Read ephys binary data and return a dictionary with the data.
    This only reads the analog data of these binary files.
    Args:
        bin_file:
        meta_file:
    Returns:
    """
    # Parameters about what data to read
    t_start = 0
    t_end = -1
    channel_dict = {0: 'sync',
                    1: 'trial_TTL',
                    2: 'whisker_stim',
                    3: 'auditory_stim',
                    4: 'valve',
                    5: 'cam1',
                    6: 'cam2',
                    7: 'lick_trace'}
    channel_list = list(channel_dict.keys())

    # Read metafile
    meta_dict = readMeta(pathlib.Path(meta_file))

    # Parameters common to NI and IMEC data
    s_rate = SampRate(meta_dict)
    first_sample = int(s_rate * t_start)
    last_sample = int(s_rate * t_end)

    # Read binary file
    raw_data = makeMemMapRaw(pathlib.Path(bin_file), meta_dict)

    # Note: this deals with analog data only
    select_data = raw_data[channel_list, first_sample:last_sample]

    # Read IMEC data
    if meta_dict['typeThis'] == 'imec':
        # Apply gain correction and convert to uV
        conv_data = 1e6 * GainCorrectIM(select_data, channel_list, meta_dict)

        conv_data_dict = {}
        conv_data_dict['imec'] = conv_data

    # Read NI data
    else:
        MN, MA, XA, DW = ChannelCountsNI(meta_dict)
        #print("NI channel counts: %d, %d, %d, %d" % (MN, MA, XA, DW))
        # Apply gain correction and convert to V
        conv_data = GainCorrectNI(select_data, channel_list, meta_dict)

        conv_data_dict = {}
        for chan_idx, chan_key in channel_dict.items():
            channel_data = conv_data[chan_idx, :]

            if chan_key == 'lick_trace':
                channel_data = np.abs(channel_data)
            conv_data_dict[chan_key] = channel_data

    return conv_data_dict

def load_ephys_sync_timestamps(config_file, log_timestamps_dict):
    """
    Load sync timestamps derived from CatGT/TPrime from config file.
    Add and compare timestamps with log_continuous.bin timestamps.

    :param config_file: path to config file
    :param log_timestamps_dict: dictionary of timestamps from log_continuous.bin
    :return: sync timestamps
    """

    event_map = {
        'trial_start_times': 'trial_TTL',
        'cam0_frame_times': 'cam1',
        'cam1_frame_times': 'cam2',
        'whisker_stim_times': 'whisker_stim_times',
        'auditory_stim_times': 'auditory_stim_times',
        'valve_times': 'reward_times',
    }

    # List event times existing in folder
    sync_event_times_folder = server_paths.get_sync_event_times_folder(config_file)
    event_files = [f for f in os.listdir(sync_event_times_folder) if f.endswith('.txt')]
    event_keys = [f.split('.')[0] for f in event_files]
    print('Existing sync event times:', event_keys)

    timestamps_dict = {}
    events_to_do = ['trial_start_times', 'cam0_frame_times', 'cam1_frame_times', 'valve_times']
    events_available = [event for event in events_to_do if event in event_keys]
    for event in events_available:
        print('Ephys session with {} event'.format(event))

        # Load sync timestamps
        timestamps = np.loadtxt(os.path.join(sync_event_times_folder, event + '.txt'))

        # Make sure same number as from log_continuous.bin
        if event == 'trial_start_times':
            if len(timestamps) != len(log_timestamps_dict[event_map[event]]):
                print(
                    'Warning: {} has {} timestamps from nidq.bin (CatGT), while {} has {} timestamps from log_continuous.bin'.format(
                        event, len(timestamps), event_map[event], len(log_timestamps_dict[event_map[event]]))
                )

        # Add to dictionary
        timestamps_dict[event_map[event]] = timestamps

    # Add piezo lick timestamps separately
    sync_delay = get_sglx_behaviour_log_delay(log_timestamps_dict, timestamps_dict)
    timestamps_dict['lick_trace'] = log_timestamps_dict['lick_trace'] + sync_delay

    return timestamps_dict


def format_ephys_timestamps(config_file, ephys_timestamps_dict):
    """
    Format ephys timestamps into (on,off) tuples.
    Args:
        config_file:
        ephys_timestamps_dict:
    Returns:
    """

    # Init. new timestamps dict
    timestamps_dict = {}
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config.get('behaviour_metadata').get('camera_flag'):
        movie_files = server_paths.get_session_movie_files(config_file)
        if movie_files is not None:
            movie_file_names = [os.path.basename(f) for f in movie_files]
            movie_file_suffix = [f.split('-')[0] for f in movie_file_names]
            movie_file_suffix = [f.split('_')[1] for f in movie_file_suffix]
            movie_file_suffix = [f.split(' ')[0] for f in movie_file_suffix]
        else:
            movie_files = None
            movie_file_suffix = None
    else:
        movie_files = None
        movie_file_suffix = None

    # Format each timestamps type separately
    for event in ephys_timestamps_dict.keys():

        timestamps = ephys_timestamps_dict[event]

        if event == 'trial_TTL':

            # Remove last timestamp that signals session end
            ts_on = timestamps[:-1]

            # Get trial stop times
            behavior_results_file = server_paths.get_behavior_results_file(config_file)
            trial_table = pd.read_csv(behavior_results_file)
            trial_response_windows = trial_table.response_window.values / 1000
            trial_artifact_windows = trial_table.artifact_window.values / 1000
            trial_durations_sec = trial_response_windows + trial_artifact_windows
            trial_durations_sec = trial_durations_sec.astype(float)

            # Keep only as many trials as in recorded behaviour output
            n_trials_behaviour = len(trial_durations_sec)
            n_trials_ts = len(ts_on)
            if n_trials_behaviour > n_trials_ts:
                trial_durations_sec = trial_durations_sec[:n_trials_ts]
            elif n_trials_behaviour < n_trials_ts:
                ts_on = ts_on[:n_trials_behaviour]
            else:
                ts_on = ts_on
            ts_off = ts_on + trial_durations_sec
            timestamps = list(zip(ts_on, ts_off))

            timestamps_dict[event] = timestamps

        elif event in ['cam1', 'cam2']:

            view_key_mapper = {
                'cam1': 'top',
                'cam2': 'side'
            }
            # If not movies or specific movie absent, set timestamps to empty list
            if movie_files is None:
                timestamps_dict[event] = []
            elif view_key_mapper[event] not in movie_file_suffix:
                timestamps_dict[event] = []
            else:
                ts_on = timestamps
                # Get exposure time
                exposure_time = float(config['behaviour_metadata']['camera_exposure_time']) / 1000
                ts_off = ts_on + exposure_time
                timestamps = list(zip(ts_on, ts_off))
                timestamps_dict[event] = timestamps

        elif event == 'reward_times':
            timestamps = list(zip(timestamps, itertools.repeat(np.nan)))
            timestamps_dict[event] = timestamps

        elif event == 'lick_trace':
            print('Info: lick_trace timestamps are already formatted as (on,off) tuples.')

        else:
            print('Warning: {} is not a recognized timestamp type'.format(event))

    print('Done formatting ephys timestamps as tuples.')
    return timestamps_dict


def get_sglx_behaviour_log_delay(log_timestamps_dict, ephys_timestamps_dict):
    """
    Get delay between SpikeGLX and behaviour logging timestamps.
    Args:
        log_timestamps_dict: dictionary of timestamps from log_continuous.bin
        ephys_timestamps_dict: dictionary of timestamps from CatGT/TPrime NIDQ acquisition

    Returns:

    """

    # Get trial timestamps
    log_trial_ts = log_timestamps_dict['trial_TTL']
    ephys_trial_ts = ephys_timestamps_dict['trial_TTL']

    # Get first trials timestamps onset
    log_sess_start = log_trial_ts[0][0]  # (on,off)-formatted
    ephys_sess_start = ephys_trial_ts[0]  # just onset

    time_delay = ephys_sess_start - log_sess_start

    return time_delay


def extract_ephys_timestamps(config_file, continuous_data_dict, threshold_dict, log_timestamps_dict):
    """
    Load and format ephys timestamps for continuous_log_analysis.
    Args:
        config_file: path to config file
        continuous_data_dict: dictionary of continuous data from SpikeGLX
        threshold_dict: dictionary of thresholds for continuous data processing
        log_timestamps_dict: dictionary of timestamps from log_continuous.bin

    Returns:

    """
    print("Extract ephys timestamps")

    # Load and format existing timestamps extracted by CatGT and TPrime
    timestamps_dict = load_ephys_sync_timestamps(config_file, log_timestamps_dict)
    timestamps_dict = format_ephys_timestamps(config_file, timestamps_dict)

    # Extract timestamps from binary files
    ephys_nidq_meta, _ = server_paths.get_raw_ephys_nidq_files(config_file)
    meta_dict = readMeta(pathlib.Path(ephys_nidq_meta))
    lick_threshold = threshold_dict.get('lick_trace')
    lick_timestamps = detect_piezo_lick_times(continuous_data_dict,
                                              ni_session_sr=meta_dict['niSampRate'],
                                              lick_threshold=lick_threshold,
                                              sigma=500)
    # Format as tuples of on/off times for NWB
    lick_timestamps_on_off = list(zip(lick_timestamps, itertools.repeat(np.nan)))
    timestamps_dict['lick_trace'] = lick_timestamps_on_off

    # The only mandatory timestamps
    assert 'trial_TTL' in timestamps_dict.keys()
    assert isinstance(timestamps_dict['trial_TTL'][0], tuple)

    n_frames_dict = {k: len(v) for k, v in timestamps_dict.items()}

    return timestamps_dict, n_frames_dict


def create_electrode_table(nwb_file):
    """
    Create electrode table in nwb file.
    Args:
        nwb_file: NWB file object

    Returns:

    """
    # Create ElectrodeTable object
    dict_columns_to_add = {'index_on_probe': 'index of saved channel per probe per shank',
                           'ccf_ml': 'ccf coordinate in ml axis',
                           'ccf_ap': 'ccf coordinate in ap axis',
                           'ccf_dv': 'ccf coordinate in dv axis',
                           'shank': 'shank number',
                           'shank_col': 'column number of electrode on shank',
                           'shank_row': 'row number of electrode on shank',
                           'ccf_id': 'ccf region ID',
                           'ccf_acronym': 'ccf region acronym',
                           'ccf_name': 'ccf region name',
                           'ccf_parent_id': 'ccf parent region ID',
                           'ccf_parent_acronym': 'ccf parent region acronym',
                           'ccf_parent_name': 'ccf parent region name',
                           }

    for col_key, col_desc in dict_columns_to_add.items():
        nwb_file.add_electrode_column(name=col_key, description=col_desc)

    return


def create_simplified_unit_table(nwb_file):
    """
    Create a simplified units table in nwb file.
    Args:
        nwb_file: NWB file object

    Returns:

    """
    default_cols = ['id', 'spike_times']
    # Create Units table (default columns are id and spike_times)
    dict_columns_to_add = {
        'cluster_id': 'cluster index, from KS(probe-wise)',
        'peak_channel': 'electrode with max waveform amplitude, from KS',
        'electrode_group': 'ElectrodeGroup object (i.e. probe) recording the unit',
        'depth': 'depth of peak electrode, in probe space, from KS',
        'ks_label': 'unit quality label, form Kilosort and curation (Phy): “good”, “mua”',
        'firing_rate': 'total firing rate in session, in Hz',
        'waveform_mean': 'mean spike waveform (a vector), in uV',
        'sampling_rate': 'sampling rate used for that probe, in Hz',
        'duration': 'spike duration, in ms, from trough to peak',
        'pt_ratio': 'peak-to-trough ratio',
        # 'unit_type': '“rsu” or “fsu” classification',
    }
    for col_key, col_desc in dict_columns_to_add.items():
        nwb_file.add_unit_column(name=col_key, description=col_desc)

    return


def create_unit_table(nwb_file):
    """
    Create units table in nwb file.
    Args:
        nwb_file: NWB file object

    Returns:

    """
    default_cols = ['id', 'spike_times']
    # Create Units table (default columns are id and spike_times)
    dict_columns_to_add = {
        'cluster_id': 'cluster index, from KS(probe-wise)',
        'peak_channel': 'electrode with max waveform amplitude, from KS',
        'electrode_group': 'ElectrodeGroup object (i.e. probe) recording the unit',
        'depth': 'depth of peak electrode, in probe space, from KS',
        'ks_label': 'unit quality label, form Kilosort and curation (Phy): “good”, “mua”',
        'firing_rate': 'total firing rate in session, in Hz',
        'waveform_mean': 'mean spike waveform (a vector), in uV',
        'sampling_rate': 'sampling rate used for that probe, in Hz',
        'duration': 'spike duration, in ms, from trough to peak',
        'pt_ratio': 'peak-to-trough ratio',
        # 'unit_type': '“rsu” or “fsu” classification',
        'ccf_ml': 'ccf peak channel coordinate in ml axis',
        'ccf_ap': 'ccf peak channel coordinate in ap axis',
        'ccf_dv': 'ccf peak channel coordinate in dv axis',
        'ccf_id': 'ccf region ID',
        'ccf_acronym': 'ccf region acronym',
        'ccf_name': 'ccf region name',
        'ccf_parent_id': 'ccf parent region ID',
        'ccf_parent_acronym': 'ccf parent region acronym',
        'ccf_parent_name': 'ccf parent region name',
    }
    for col_key, col_desc in dict_columns_to_add.items():
        nwb_file.add_unit_column(name=col_key, description=col_desc)


def build_unit_table(imec_folder, sync_spike_times_path):
    """
    Build unit table from spike sorting/curation output.
    Args:
        imec_folder: path to imec folder
        sync_spike_times_path:  path to sync spike times

    Returns:

    """
    # Init. table
    unit_table = pd.DataFrame()

    # Load cluster table
    cluster_info_path = pathlib.Path(imec_folder, 'cluster_info.tsv')
    try:
        cluster_info_df = pd.read_csv(cluster_info_path, sep='\t')
    except FileNotFoundError:
        print('No spike sorting at: {}'.format(cluster_info_path))
        return

    cluster_info_df.rename(columns={'KSLabel': 'ks_label',
                                    'Amplitude': 'amplitude',
                                    'ContamPct': 'contam_pct'}, inplace=True)

    # Find if cluster had a curated label
    cluster_info_df['curated'] = cluster_info_df.apply(lambda x: 0 if pd.isnull(x.group) else 1, axis=1)

    # Phy-based new clusters/ new splits have no ks_label: convert NaN to None
    cluster_info_df.fillna(value='', inplace=True)  # returns None

    # Get valid cluster indices
    valid_cluster_ids = cluster_info_df[cluster_info_df.group.isin(['good', 'mua'])].index  # dataframe indices
    cluster_info_df_sub = cluster_info_df.loc[valid_cluster_ids, :]

    # Add cluster information
    unit_table['cluster_id'] = cluster_info_df_sub['cluster_id']
    unit_table['peak_channel'] = cluster_info_df_sub['ch']
    unit_table['depth'] = cluster_info_df_sub['depth']
    unit_table['ks_label'] = cluster_info_df_sub['group']  # "group" is the curated label, "KSLabel" is the KS label
    unit_table['firing_rate'] = cluster_info_df_sub['fr']

    # Load spikes times
    spike_times_sync = np.load(sync_spike_times_path)
    spike_times_sync_df = pd.DataFrame(data=spike_times_sync, columns=['spike_times'])
    spike_times_sync_df.index.name = 'spike_id'
    spike_times_per_cluster = []

    # Load spike cluster assignments
    spike_clusters = np.load(os.path.join(imec_folder, 'spike_clusters.npy'))
    spike_clusters_df = pd.DataFrame(data=spike_clusters, columns=['cluster_id'])
    spike_clusters_df.index.name = 'spike_id'

    # Note: Iterate over selected good cluster only !
    for c_id in cluster_info_df.cluster_id.values:
        spike_ids = spike_clusters_df[spike_clusters_df.cluster_id == c_id].index
        spike_times_per_cluster.append(np.array(spike_times_sync_df.iloc[spike_ids].spike_times))
    cluster_info_df['spike_times'] = spike_times_per_cluster

    unit_table['spike_times'] = cluster_info_df.loc[valid_cluster_ids].spike_times

    # Load mean waveform data
    mean_wfs = np.load(os.path.join(imec_folder, 'cwaves', 'mean_waveforms.npy'))
    peak_channels = cluster_info_df_sub.loc[valid_cluster_ids, 'ch'].values
    mean_wfs = mean_wfs[valid_cluster_ids, peak_channels, :]  # note: keep only valid clusters and peak channels
    unit_table['waveform_mean'] = pd.DataFrame(mean_wfs).to_numpy().tolist()

    # Load mean waveform metrics
    mean_wf_metrics = pd.read_csv(os.path.join(imec_folder, 'cwaves', 'waveform_metrics.csv'))
    unit_table['duration'] = mean_wf_metrics.loc[valid_cluster_ids].duration.values
    unit_table['pt_ratio'] = mean_wf_metrics.loc[valid_cluster_ids].pt_ratio.values

    return unit_table


def build_area_table(imec_folder):
    """
    Build area table from brainreg output.
    Args:
        imec_folder: path to imec folder anatomical data

    Returns:

    """

    # -----------------------------------------
    # Load ccf probe track areas (sample space)
    # -----------------------------------------

    imec_id = imec_folder[-1]
    mouse_name = imec_folder.split('\\')[7]
    #path_to_proc_anat = imec_folder.replace('Ephys', 'Anatomy')  # TODO: confirm location and update
    #path_to_proc_anat = path_to_proc_anat.replace(imec_folder.partition('Ephys')[-1], '\\brainreg\\manual_segmentation\\')
    path_to_proc_anat = r'M:\analysis\Axel_Bisi\ImagedBrains\{}\brainreg\manual_segmentation'.format(mouse_name)
    area_table = pd.read_csv(os.path.join(path_to_proc_anat, 'sample_space\\tracks', 'imec{}.csv'.format(imec_id)))

    # Format table for future shank row matching
    area_table.rename(columns={'Position': 'shank_row',
                               'Region ID': 'ccf_id',
                               'Region acronym': 'ccf_acronym',
                               'Region name': 'ccf_name'}, inplace=True)

    # Set outside brain points to atlas root
    area_table.loc[area_table['ccf_id'] == 'Not found in brain', 'ccf_id'] = 997
    area_table.loc[area_table['ccf_acronym'] == 'Not found in brain', 'ccf_acronym'] = 'root'
    area_table.loc[area_table['ccf_name'] == 'Not found in brain', 'ccf_name'] = 'root'

    area_table = area_table.iloc[::-1]  # reverse order (from probe tip upwards)
    area_table = area_table.iloc[9:, :]  # remove first 9 rows (probe tip)
    max_position = np.max(area_table['shank_row'].values)
    area_table['shank_row'] = max_position - area_table['shank_row'].values  # make values start at 0

    # Add atlas metadata
    path_to_atlas = r'C:\Users\bisi\.brainglobe\allen_mouse_25um_v1.2' #TODO: hard-coded path
    with open(os.path.join(path_to_atlas, 'metadata.json')) as f:
        atlas_metadata = json.load(f)
    area_table['atlas_metadata'] = str(atlas_metadata)

    # Simplify CCF hierarchical nomenclature with parent structure
    with open(os.path.join(path_to_atlas, 'structures.json')) as f:
        structures_dict_list = json.load(f)

    # For each region_id, get parent structures
    ccf_ids = np.array(area_table['ccf_id'].values, dtype=int)
    present_structures = {i['id']: i for i in structures_dict_list if i['id'] in ccf_ids}  # all present structures

    # Get corresponding parent structure IDs
    ccf_parent_ids = {ccf_id: (struct['structure_id_path'][-2] if struct['name']!='root' else 997)
                      for ccf_id, struct in present_structures.items()}

    # Map region to parent structure
    ccf_parent_dict = {i['id']: i for i in structures_dict_list if i['id'] in ccf_parent_ids.values()}  # parent structures

    # Make hierarchical mappers: cff area <-> ccf parent area information
    ccf_parent_id_mapper = {ccf_id: ccf_parent_dict[ccf_parent_ids[ccf_id]]['id'] for ccf_id in ccf_parent_ids.keys()}
    ccf_parent_acronym_mapper = {ccf_id: ccf_parent_dict[ccf_parent_ids[ccf_id]]['acronym'] for ccf_id in
                                 ccf_parent_ids.keys()}
    ccf_parent_name_mapper = {ccf_id: ccf_parent_dict[ccf_parent_ids[ccf_id]]['name'] for ccf_id in
                              ccf_parent_ids.keys()}

    # Add parent structure information
    area_table['ccf_parent_id'] = [ccf_parent_id_mapper[ccf_id] for ccf_id in ccf_ids]
    area_table['ccf_parent_acronym'] = [ccf_parent_acronym_mapper[ccf_id] for ccf_id in ccf_ids]
    area_table['ccf_parent_name'] = [ccf_parent_name_mapper[ccf_id] for ccf_id in ccf_ids]

    # -----------------------------------------
    # Load ccf coordinates (ccf standard space)
    # -----------------------------------------

    coords = np.load(os.path.join(path_to_proc_anat, 'standard_space\\tracks', 'imec{}.npy'.format(imec_id)))
    coords = coords[::-1]
    coords = coords[9:, :]
    area_table['ccf_ap'] = coords[:, 0]
    area_table['ccf_ml'] = coords[:, 1]
    area_table['ccf_dv'] = coords[:, 2]

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
    # plt.show()


    return area_table

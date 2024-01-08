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

import numpy as np
import pandas as pd
import yaml

from utils import server_paths
from utils.continuous_processing import detect_piezo_lick_times
from utils.read_sglx import readMeta

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

    # This is experimenter-specific tracking of that information
    if config.get('session_metadata').get('experimenter') == 'AB':

        # Load probe insertion table
        path_to_info_file = r'M:\analysis\Axel_Bisi\mice_info\probe_insertion_info.xlsx'
        location_df = pd.read_excel(path_to_info_file, sheet_name='Sheet1')

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

    else:
        print('No location information found for this experimenter.')
        raise NotImplementedError

    return location_dict


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
    for event in events_to_do:
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

    movie_files = server_paths.get_session_movie_files(config_file)
    movie_file_names = [os.path.basename(f) for f in movie_files]
    movie_file_suffix = [f.split('-')[0] for f in movie_file_names]
    movie_file_suffix = [f.split('_')[1] for f in movie_file_suffix]

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
    Get delay between SGLX and behaviour logging timestamps.
    Args:
        log_timestamps_dict:
        ephys_timestamps_dict:

    Returns:

    """

    # Get trial timestamps
    log_trial_ts = log_timestamps_dict['trial_TTL']
    ephys_trial_ts = ephys_timestamps_dict['trial_TTL']

    # Get first trials timestamps onset
    log_sess_start = log_trial_ts[0][0] # (on,off)-formatted
    ephys_sess_start = ephys_trial_ts[0] # just onset

    time_delay = ephys_sess_start - log_sess_start

    return time_delay


def extract_ephys_timestamps(config_file, continuous_data_dict, threshold_dict, log_timestamps_dict):
    """
    Load and format ephys timestamps for continuous_log_analysis.
    Args:
        config_file:
        continuous_data_dict:
        threshold_dict:
        log_timestamps_dict:

    Returns:

    """
    print("Extracting ephys timestamps")

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

    assert 'trial_TTL' in timestamps_dict.keys()
    assert 'cam1' in timestamps_dict.keys()
    assert 'cam2' in timestamps_dict.keys()

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
                           'ccf_location': 'ccf area of electrode'
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
    # Create Units table
    dict_columns_to_add = {
        'cluster_id': 'cluster index, from KS(probe-wise)',
        'peak_channel': 'electrode with max waveform amplitude, from KS',
        'electrode_group': 'ElectrodeGroup object (i.e. probe) recording the unit',
        'depth': 'depth of peak electrode, in probe space, from KS',
        'ks_label': 'unit quality label, form Kilosort and curation (Phy): “good”, “mua”',
        'firing_rate': 'total firing rate in session, in Hz',
        'waveform_mean': 'mean spike waveform (a vector), in uV',
        'sampling_rate': 'sampling rate used for that probe, in Hz',
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
    # Create Units table
    dict_columns_to_add = {
        'cluster_id': 'cluster index, from KS(probe-wise)',
        'peak_channel': 'electrode with max waveform amplitude, from KS',
        'rel_x': 'peak channel x-coordinate in probe space (as above)',
        'rel_y': 'peak channel y-coordinate in probe space (as above)',
        'electrode_group': 'ElectrodeGroup object (i.e. probe) recording the unit',
        'depth': 'depth of peak electrode, in probe space, from KS',
        'ks_label': 'unit quality label, form Kilosort and curation (Phy): “good”, “mua”',
        'firing_rate': 'total firing rate in session, in Hz',
        'ccf_area': 'atlas label from anatomical estimation',
        'ccf_area_layer': 'same as ccf_area, with layer-specficity',
        'waveform_mean': 'mean spike waveform (a vector), in uV',
        'spike_width': 'spike duration, in ms, form trough to baseline',
        'peak_to_trough': 'spike duration, in ms, from peak to trough',
        'unit_type': '“rsu” or “fsu” classification',
        'sampling_rate': 'sampling rate used for that probe, in Hz',
        'isi_violation': '',
        'isolation_distance': '',
    }
    for col_key, col_desc in dict_columns_to_add.items():
        nwb_file.add_unit_column(name=col_key, description=col_desc)


def build_simplified_unit_table(imec_folder,
                                sync_spike_times_path):
    # Init. table
    unit_table = pd.DataFrame()

    # Load unit data
    # 2. Cwaves waveform data + metrics todo
    # 3. track localization output (electrode) -> function matching electrode peak channel and neuron area

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
    valid_cluster_ids = cluster_info_df[cluster_info_df.group.isin(['good', 'mua'])].index # dataframe indices

    cluster_info_df_sub = cluster_info_df.loc[valid_cluster_ids, :]

    unit_table['cluster_id'] = cluster_info_df_sub['cluster_id']
    unit_table['peak_channel'] = cluster_info_df_sub['ch']
    unit_table['depth'] = cluster_info_df_sub['depth']
    unit_table['ks_label'] = cluster_info_df_sub['group'] # "group" is the curated label, "KSLabel" is the KS label
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

    #valid_cluster_ids = cluster_info_df[cluster_info_df.group.isin(['good', 'mua'])].index
    peak_channels = cluster_info_df_sub.loc[valid_cluster_ids, 'ch'].values

    # Filter mean waveform array and add to table
    mean_wfs = mean_wfs[valid_cluster_ids, peak_channels, :]
    unit_table['waveform_mean'] = pd.DataFrame(mean_wfs).to_numpy().tolist()

    # Load mean waveform metrics TODO
    mean_wf_metrics = pd.read_csv(os.path.join(imec_folder, 'cwaves', 'waveform_metrics.csv'))
    #unit_table['spike_width'] = mean_wf_metrics[valid_cluster_ids, 'spike_width'].values
    #unit_table['peak_to_trough'] = mean_wf_metrics[valid_cluster_ids, 'peak_to_trough'].values



    return unit_table

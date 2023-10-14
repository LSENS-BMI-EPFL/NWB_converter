#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: NWB_converter
@file: ephys_utils.py
@time: 8/24/2023 9:25 AM
"""

import os
import yaml
import numpy as np
import pandas as pd
from utils import server_paths


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
    Read location target: hemisphere, stereotaxic coordinate, angles.
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
    Compare timestamps with log_continuous.bin timestamps.
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
        'valve_times:': 'reward_times',
    }

    # List event times existing in folder
    sync_event_times_folder = server_paths.get_sync_event_times_folder(config_file)
    event_files = [f for f in os.listdir(sync_event_times_folder) if f.endswith('.txt')]
    event_keys = [f.split('.')[0] for f in event_files]
    print('Existing sync event times:', event_keys)

    timestamps_dict = {}
    events_to_do = ['trial_start_times', 'cam0_frame_times', 'cam1_frame_times']
    for event in events_to_do:
        print('Ephys session with {} event'.format(event))

        # Load sync timestamps
        timestamps = np.loadtxt(os.path.join(sync_event_times_folder, event + '.txt'))

        # Make sure same number as from log_continuous.bin
        if len(timestamps) != len(log_timestamps_dict[event_map[event]]):
            print(f'Warning: {event} has {len(timestamps)} timestamps from nidq.bin (CatGT), while {event_map[event]} has {len(log_timestamps_dict[event_map[event]])} timestamps from log_continuous.bin')

        # Add to dictionary
        timestamps_dict[event_map[event]] = timestamps

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

            # Get trial stops
            behavior_results_file = server_paths.get_behavior_results_file(config_file)
            trial_table = pd.read_csv(behavior_results_file)
            trial_durations_sec = trial_table.trial_duration.values / 1000
            trial_durations_sec = 1.0 # max. response window
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

        else:
            print('Warning: {} is not a recognized timestamp type'.format(event))


    print('Done formattting ephys timestamps as tuples')
    return timestamps_dict

def get_ephys_timestamps(config_file, log_timestamps_dict):
    """
    Load and format ephys timestamps for continuous_log_analysis.
    Args:
        config_file:
        log_timestamps_dict:

    Returns:

    """
    timestamps_dict = load_ephys_sync_timestamps(config_file, log_timestamps_dict)
    timestamps_dict = format_ephys_timestamps(config_file, timestamps_dict)

    assert 'trial_TTL' in timestamps_dict.keys()
    assert 'cam1' in timestamps_dict.keys()
    assert 'cam2' in timestamps_dict.keys()

    assert type(timestamps_dict['trial_TTL'][0]) == tuple

    n_frames_dict = {k:len(v) for k,v in timestamps_dict.items()}

    return timestamps_dict, n_frames_dict

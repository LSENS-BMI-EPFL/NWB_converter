#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: NWB_converter
@file: ecephys_utils.py
@time: 8/24/2023 9:25 AM
"""

import os
import numpy as np
from utils import server_paths


def get_ephys_sync_timestamps(config_file):
    """
    Get sync timestamps derived from CatGT/TPrime from config file.
    :param config_file: path to config file
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

    timestamps_dict, n_frames_dict = {}, {}
    events_to_do = ['trial_start_times', 'cam0_frame_times', 'cam1_frame_times']
    for event in events_to_do:

        timestamps = np.loadtxt(os.path.join(sync_event_times_folder, event + '.txt'))
        if event == 'trial_start_times':
            timestamps = timestamps[1:-1] # remove first and last trials (session start and stops) #TODO: check this
        timestamps_dict[event_map[event]] = timestamps
        n_frames_dict[event_map[event]] = len(timestamps_dict[event_map[event]])

    assert 'trial_TTL' in timestamps_dict.keys()
    assert 'cam1' in timestamps_dict.keys()
    assert 'cam2' in timestamps_dict.keys()

    return timestamps_dict, n_frames_dict




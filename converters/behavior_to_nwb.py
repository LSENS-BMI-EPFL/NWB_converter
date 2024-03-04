import os
import re

import numpy as np
import yaml
from pynwb.base import TimeSeries
from pynwb.behavior import BehavioralEpochs, BehavioralEvents
from pynwb.image import ImageSeries

from utils import continuous_processing, server_paths
from utils.behavior_converter_misc import (add_trials_standard_to_nwb,
                                           add_trials_to_nwb,
                                           build_simplified_trial_table,
                                           build_standard_trial_table,
                                           get_context_timestamps_dict,
                                           get_piezo_licks_timestamps_dict,
                                           get_trial_timestamps_dict,
                                           get_motivated_epoch_ts)


def convert_behavior_data(nwb_file, timestamps_dict, config_file):
    """
    Convert behavior data to NWB format and add to NWB file.
    Args:
        nwb_file:
        timestamps_dict:
        config_file:

    Returns:

    """

    # Get session behaviour results file
    behavior_results_file = server_paths.get_behavior_results_file(config_file)

    # Make trial table
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    if 'behaviour_metadata' in config_dict:
        if config_dict.get('behaviour_metadata').get('trial_table') == 'standard':
            trial_table = build_standard_trial_table(
                config_file=config_file,
                behavior_results_file=behavior_results_file,
                timestamps_dict=timestamps_dict
            )
        elif config_dict.get('behaviour_metadata').get('trial_table') == 'simple':  # TODO: remove?
            trial_table = build_simplified_trial_table(behavior_results_file=behavior_results_file,
                                                       timestamps_dict=timestamps_dict)
    else:
        print('NWB config file lacks a behaviour_metadata section. Fix config file first.')

    print("Adding trials to NWB file")
    if config_dict.get('behaviour_metadata').get('trial_table') == 'standard':
        add_trials_standard_to_nwb(nwb_file=nwb_file, trial_table=trial_table)
    else:
        add_trials_to_nwb(nwb_file=nwb_file, trial_table=trial_table)

    # Create NWB behaviour module (and module interfaces)

    if timestamps_dict is None:
        return

    print("Creating behaviour processing module")
    if 'behavior' in nwb_file.processing:
        bhv_module = nwb_file.processing['behavior']
    else:
        bhv_module = nwb_file.create_processing_module('behavior', 'contains behavioral processed data')

    try:
        behavior_events = bhv_module.get(name='BehavioralEvents')
    except KeyError:
        behavior_events = BehavioralEvents(name='BehavioralEvents')
        bhv_module.add_data_interface(behavior_events)

    # Get trial timestamps and indexes
    trial_timestamps_dict, trial_indexes_dict = get_trial_timestamps_dict(timestamps_dict,
                                                                          behavior_results_file, config_file)

    # Add a time series of trial timestamps, for each trial type
    trial_types = list(trial_timestamps_dict.keys())
    for trial_type in trial_types:
        data_to_store = np.transpose(np.array(trial_indexes_dict.get(trial_type)))
        timestamps_on_off = trial_timestamps_dict.get(trial_type)
        timestamps_to_store = timestamps_on_off[0]

        trial_timeseries = TimeSeries(name=f'{trial_type}_trial',
                                      data=data_to_store,
                                      unit='seconds',
                                      resolution=-1.0,
                                      conversion=1.0,
                                      offset=0.0,
                                      timestamps=timestamps_to_store,
                                      starting_time=None,
                                      rate=None,
                                      comments='no comments',
                                      description=f'index (data) and timestamps of {trial_type} trials',
                                      control=None,
                                      control_description=None,
                                      continuity='instantaneous')

        behavior_events.add_timeseries(trial_timeseries)
        print(f"Adding {len(data_to_store)} {trial_type} to BehavioralEvents")

    # Get piezo lick timestamps
    piezo_licks_timestamps_dict = get_piezo_licks_timestamps_dict(timestamps_dict)

    if piezo_licks_timestamps_dict is not None:
        timestamps_to_store = np.array(piezo_licks_timestamps_dict)
        if timestamps_to_store.any():
            timestamps_to_store = timestamps_to_store[:, 0]

        data_to_store = np.transpose(np.array(timestamps_to_store))
        lick_timeseries = TimeSeries(name='piezo_lick_times',
                                     data=data_to_store,
                                     unit='seconds',
                                     resolution=-1.0,
                                     conversion=1.0,
                                     offset=0.0,
                                     timestamps=timestamps_to_store,
                                     starting_time=None,
                                     rate=None,
                                     comments='no comments',
                                     description='piezo lick timestamps',
                                     control=None,
                                     control_description=None,
                                     continuity='instantaneous')
        behavior_events.add_timeseries(lick_timeseries)
        print(f"Adding {len(data_to_store)} piezo lick times to BehavioralEvents")

    # Get context timestamps if they exist
    context_timestamps_dict, context_sound_dict = get_context_timestamps_dict(timestamps_dict=timestamps_dict,
                                                                              nwb_trial_table=trial_table)
    # If context, add context timestamps to NWB file
    if context_timestamps_dict is not None:
        print("Adding context epochs to NWB file")
        try:
            behavior_epochs = bhv_module.get(name='BehavioralEpochs')
        except KeyError:
            behavior_epochs = BehavioralEpochs(name='BehavioralEpochs')
            bhv_module.add_data_interface(behavior_epochs)

        for epoch, intervals_list in context_timestamps_dict.items():
            print(f"Add {len(intervals_list)} {epoch} epochs to NWB ")
            time_stamps_to_store = []
            data_to_store = []
            description = context_sound_dict.get(epoch)
            for interval in intervals_list:
                start_time = interval[0]
                stop_time = interval[1]
                time_stamps_to_store.extend([start_time, stop_time])
                data_to_store.extend([1, -1])
            behavior_epochs.create_interval_series(name=epoch, data=data_to_store, timestamps=time_stamps_to_store,
                                                   comments='no comments',
                                                   description=description,
                                                   control=None, control_description=None)

    if config_dict.get("two_photon_metadata") is not None:
        # Get motivated/unmotivated timestamps
        motivated_timestamps_dict = get_motivated_epoch_ts(timestamps_dict=timestamps_dict, nwb_trial_table=trial_table)

        # Add motivated epochs
        if motivated_timestamps_dict is not None:
            print("Adding motivated epochs to NWB file")
            try:
                behavior_epochs = bhv_module.get(name='BehavioralEpochs')
            except KeyError:
                behavior_epochs = BehavioralEpochs(name='BehavioralEpochs')
                bhv_module.add_data_interface(behavior_epochs)

            for epoch, intervals_list in motivated_timestamps_dict.items():
                print(f"Add {len(intervals_list)} {epoch} epochs to NWB ")
                time_stamps_to_store = []
                data_to_store = []
                description = epoch + '_epoch'
                for interval in intervals_list:
                    start_time = interval[0]
                    stop_time = interval[1]
                    time_stamps_to_store.extend([start_time, stop_time])
                    data_to_store.extend([1, -1])
                behavior_epochs.create_interval_series(name=epoch, data=data_to_store, timestamps=time_stamps_to_store,
                                                       comments='no comments',
                                                       description=description,
                                                       control=None, control_description=None)

    # Check if behaviour video filming
    if config_dict.get('session_metadata').get('experimenter') == 'AB':
        movie_files = server_paths.get_session_movie_files(config_file)
    else:
        movie_files = server_paths.get_movie_files(config_file)

    # If there is a behaviour video, add camera frame timestamps to NWB file
    if config_dict.get('behaviour_metadata').get('camera_flag') == 0:
        movie_files = None

    if movie_files is not None:
        print("Adding behavior movies as external file to NWB file")
        for movie_index, movie in enumerate(movie_files):

            # If movie file does not exist, skip
            if not os.path.exists(movie):
                print(f"File not found, do next video")
                continue

            # Get information about video
            print("Check length and frame rate")
            video_length, video_frame_rate = continuous_processing.read_behavior_avi_movie(movie_file=movie)
            print(f"Video length: {video_length}, frame rate: {video_frame_rate}")

            #  Check number of frames in video vs. number of timestamps
            if config_dict.get('session_metadata').get('experimenter') == 'AB':
                key_view_mapper = {
                    'top': 'cam1',
                    'side': 'cam2',
                    'lateral': 'cam2'
                }
                movie_file_name = os.path.basename(movie)
                movie_file_name = movie_file_name.replace('-', '_')
                movie_file_name = movie_file_name.replace(' ', '_')
                move_file_parts = movie_file_name.split('_')
                movie_file_suffix = [part for part in move_file_parts if part in key_view_mapper.keys()][0]
                cam_key = key_view_mapper[movie_file_suffix]

                movie_nwb_file_name = movie

            else:
                movie_nwb_file_name = f"{ os.path.split(movie)[1].split('.')[0]}_camera_{movie_index + 1}"
                if 'side' in movie_nwb_file_name:
                    cam_key = 'cam1'
                if 'top' in movie_nwb_file_name:
                    cam_key = 'cam2'
                if ('top' not in movie_nwb_file_name) and ('side' not in movie_nwb_file_name):
                    cam_key = 'cam1'

            # Get frame timestamps
            on_off_timestamps = timestamps_dict[cam_key]
            if np.abs(len(on_off_timestamps) - video_length) > 2:
                print(f"Difference in number of frames ({video_length}) vs detected frames ({len(on_off_timestamps)}) "
                      f"is {len(on_off_timestamps) - video_length} (larger than 2), do next video")
                continue
            else:
                movie_timestamps = [on_off_timestamps[i][0] for i in range(video_length)]

            behavior_external_file = ImageSeries(
                name=movie_nwb_file_name,
                description="Behavior video of animal in the task",
                unit="n.a.",
                external_file=[movie],
                format="external",
                starting_frame=[0],
                timestamps=movie_timestamps
            )

            nwb_file.add_acquisition(behavior_external_file)

import os
import yaml
import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from utils.server_paths import get_dlc_file_path
from utils.dlc_utils import *

from pynwb.base import TimeSeries
from pynwb.behavior import BehavioralEvents, BehavioralTimeSeries, BehavioralEpochs


def convert_dlc_data(nwb_file, config_file, video_timestamps):

    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    dlc_file_path = get_dlc_file_path(config_file)
    print(dlc_file_path)

    print("Creating behaviour processing module")
    if 'behavior' in nwb_file.processing:
        bhv_module = nwb_file.processing['behavior']
    else:
        bhv_module = nwb_file.create_processing_module('behavior', 'contains behavioral processed data')

    # Get the behavioral events module (that we will use for discrete data, eg lick times)
    try:
        behavior_events = bhv_module.get(name='BehavioralEvents')
    except KeyError:
        behavior_events = BehavioralEvents(name='BehavioralEvents')
        bhv_module.add_data_interface(behavior_events)

    # Get the behavioral timeseries module (that we will use for continuous data)
    try:
        behavior_t_series = bhv_module.get(name='BehavioralTimeSeries')
    except KeyError:
        behavior_t_series = BehavioralTimeSeries(name='BehavioralTimeSeries')
        bhv_module.add_data_interface(behavior_t_series)

    # Retrieve the multiindex dataframes with the DLC results for top and side views
    side_dlc, top_dlc = get_dlc_dataframe(dlc_file_path)

    # If one of the two is missing, stop processing
    if len(side_dlc) == 0 or len(top_dlc) == 0:
        ValueError("DLC data can't be found or needs to be analyzed")

    # Compute kinematics for each of the views.
    if config['session_metadata']['experimenter'] == 'AB':
        side_dlc = compute_kinematics_alt(side_dlc, 'sideview')
        top_dlc = compute_kinematics_alt(top_dlc, 'topview')
        pcutoff_tongue = 0.5

    else:
        side_dlc = compute_kinematics(side_dlc, 'sideview')
        top_dlc = compute_kinematics(top_dlc, 'topview')
        pcutoff_tongue = 0.8 #cutoff for tongue/lick events detection

    px_ref = get_reference_from_grid(config)

    print('Adding DLC time series and events to NWB file...')
    for name, data in side_dlc.items():
        ts = [timestamp[0] for timestamp in video_timestamps['cam1']]
        rate = np.round(1 / np.median(np.diff(ts[0:200])), 2)
        if 'velocity' in name: # scale frame difference wrt. to sampling rate
            data = data * rate

        # Add times series for bodybarts
        timeseries = TimeSeries(name=f'{name}',
                                         data=data.to_numpy(),
                                         unit='seconds',
                                         resolution=-1.0,
                                         conversion=[1/px_ref[key].values[0] for key in px_ref.keys() if "side" in key][0],
                                         offset=0.0,
                                         timestamps=[timestamp[0] for timestamp in video_timestamps['cam1']],
                                         starting_time=None,
                                         rate=None,
                                         comments='no comments',
                                         description=f'no description',
                                         control=None,
                                         control_description=None,
                                         continuity='continuous')

        behavior_t_series.add_timeseries(timeseries)

    for name, data in top_dlc.items():
        if 'nose' in name or 'particle' in name:
            name = 'top_' + name

        ts = [timestamp[0] for timestamp in video_timestamps['cam2']]
        rate = np.round(1 / np.median(np.diff(ts[0:200])), 2)
        if 'velocity' in name: # scale frame difference wrt. to sampling rate
            data = data * rate

        # Add times series for bodybarts
        timeseries = TimeSeries(name=f'{name}',
                                         data=data.to_numpy(),
                                         unit='seconds',
                                         resolution=-1.0,
                                         conversion=[1/px_ref[key].values[0] for key in px_ref.keys() if "top" in key][0],
                                         offset=0.0,
                                         timestamps=[timestamp[0] for timestamp in video_timestamps['cam2']],
                                         starting_time=None,
                                         rate=None,
                                         comments='no comments',
                                         description=f'no description',
                                         control=None,
                                         control_description=None,
                                         continuity='continuous')

        behavior_t_series.add_timeseries(timeseries)

    # Add lick times counted as the peaks of tongue distance
    tongue_licks, _ = find_peaks(np.where(side_dlc['tongue_likelihood'] > pcutoff_tongue, side_dlc['tongue_distance'], np.nan), distance= 20)

    data_to_store = np.arange(len(tongue_licks))  # data would be lick index
    timestamps_to_store = tongue_licks  # same length as n_licks absolute times of licks
    lick_timeseries = TimeSeries(name=f'tongue_dlc_licks',
                                  data=data_to_store,
                                  unit='seconds',
                                  resolution=-1.0,
                                  conversion=1.0,
                                  offset=0.0,
                                  timestamps=timestamps_to_store,
                                  starting_time=None,
                                  rate=None,
                                  comments='no comments',
                                  description=f'index (data) and timestamps of DLC detected licks',
                                  control=None,
                                  control_description=None,
                                  continuity='instantaneous')

    behavior_events.add_timeseries(lick_timeseries)

    # Add movement times as peaks of jaw angle
    jaw_licks, _ = find_peaks(np.where(side_dlc['jaw_likelihood'] > 0.8, side_dlc['jaw_angle'], np.nan), prominence=side_dlc['jaw_angle'].std() * 1.8)

    data_to_store = np.arange(len(jaw_licks))  # data would be lick index
    timestamps_to_store = jaw_licks  # same length as n_licks absolute times of licks
    lick_timeseries = TimeSeries(name=f'jaw_dlc_licks',
                                  data=data_to_store,
                                  unit='seconds',
                                  resolution=-1.0,
                                  conversion=1.0,
                                  offset=0.0,
                                  timestamps=timestamps_to_store,
                                  starting_time=None,
                                  rate=None,
                                  comments='no comments',
                                  description=f'index (data) and timestamps of DLC detected licks',
                                  control=None,
                                  control_description=None,
                                  continuity='instantaneous')

    behavior_events.add_timeseries(lick_timeseries)

    print('Done DLC conversion to NWB.')
    return
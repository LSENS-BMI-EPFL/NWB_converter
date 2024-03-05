from pynwb.base import TimeSeries
from pynwb.behavior import BehavioralEvents, BehavioralTimeSeries

import numpy as np


def convert_dlc_data(nwb_file, timestamps_dict, config_file):

    # todo
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

    # Add lick times
    data_to_store = np.arange(n_licks)  # data would be lick index
    timestamps_to_store = lick_times  # same length as n_licks absolute times of licks
    trial_timeseries = TimeSeries(name=f'dlc_licks',
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

    behavior_events.add_timeseries(trial_timeseries)
    print(f"Adding {len(data_to_store)} DLC lick times to BehavioralEvents")

    # Add times series for bodybarts
    wh_angle_timeseries = TimeSeries(name=f'whisker_angle',
                                     data=whisker_angle,
                                     unit='seconds',
                                     resolution=-1.0,
                                     conversion=1.0,
                                     offset=0.0,
                                     timestamps=video_timestamps,
                                     starting_time=None,
                                     rate=None,
                                     comments='no comments',
                                     description=f'Angle of whisker',
                                     control=None,
                                     control_description=None,
                                     continuity='continuous')

    behavior_t_series.add_timeseries(wh_angle_timeseries)
    print(f"Adding ... to BehavioraltiemSeries")

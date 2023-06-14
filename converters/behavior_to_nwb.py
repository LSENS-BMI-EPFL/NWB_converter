import os
import numpy as np
import yaml
from utils.behavior_converter_misc import get_trial_timestamps_dict, build_simplified_trial_table, add_trials_to_nwb
from pynwb.behavior import BehavioralEvents
from pynwb.base import TimeSeries
from utils import server_paths


def convert_behavior_data(nwb_file, timestamps_dict, config_file):

    behavior_results_file = server_paths.get_behavior_results_file(config_file)
    trial_timestamps_dict, trial_indexes_dict = get_trial_timestamps_dict(timestamps_dict,
                                                                          behavior_results_file, config_file)
    trial_types = list(trial_timestamps_dict.keys())

    if 'behavior' in nwb_file.processing:
        bhv_module = nwb_file.processing['behavior']
    else:
        bhv_module = nwb_file.create_processing_module('behavior', 'contains behavioral processed data')

    try:
        behavior_events = bhv_module.get(name='BehavioralEvents')
    except KeyError:
        behavior_events = BehavioralEvents(name='BehavioralEvents')
        bhv_module.add_data_interface(behavior_events)

    for trial_type in trial_types:
        data_to_store = np.transpose(np.array(trial_indexes_dict.get(trial_type)))
        timestamps_on_off = trial_timestamps_dict.get(trial_type)
        timestamps_to_store = timestamps_on_off[0]

        trial_timeseries = TimeSeries(name=f'{trial_type}_trial', data=data_to_store, unit='seconds',
                                      resolution=-1.0, conversion=1.0, offset=0.0, timestamps=timestamps_to_store,
                                      starting_time=None, rate=None, comments='no comments',
                                      description=f'index (data) and timestamps of {trial_type} trials',
                                      control=None, control_description=None, continuity='instantaneous')

        behavior_events.add_timeseries(trial_timeseries)
        print(f"Adding {len(data_to_store)} {trial_type} to BehavioralEvents")

    simplified_trial_table = build_simplified_trial_table(behavior_results_file=behavior_results_file,
                                                          timestamps_dict=timestamps_dict)

    print("Adding trials to NWB file")
    add_trials_to_nwb(nwb_file, simplified_trial_table)



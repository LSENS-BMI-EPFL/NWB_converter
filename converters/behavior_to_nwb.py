import numpy as np
from utils.behavior_converter_misc import get_trial_timestamps_dict, build_simplified_trial_table, add_trials_to_nwb, \
    get_context_timestamps_dict
from pynwb.behavior import BehavioralEvents, BehavioralEpochs
from pynwb.base import TimeSeries
from utils import server_paths


def convert_behavior_data(nwb_file, timestamps_dict, config_file):

    behavior_results_file = server_paths.get_behavior_results_file(config_file)
    trial_timestamps_dict, trial_indexes_dict = get_trial_timestamps_dict(timestamps_dict,
                                                                          behavior_results_file, config_file)

    simplified_trial_table = build_simplified_trial_table(behavior_results_file=behavior_results_file,
                                                          timestamps_dict=timestamps_dict)

    context_timestamps_dict, context_sound_dict = get_context_timestamps_dict(timestamps_dict=timestamps_dict,
                                                                              nwb_trial_table=simplified_trial_table)

    if 'behavior' in nwb_file.processing:
        bhv_module = nwb_file.processing['behavior']
    else:
        bhv_module = nwb_file.create_processing_module('behavior', 'contains behavioral processed data')

    try:
        behavior_events = bhv_module.get(name='BehavioralEvents')
    except KeyError:
        behavior_events = BehavioralEvents(name='BehavioralEvents')
        bhv_module.add_data_interface(behavior_events)

    trial_types = list(trial_timestamps_dict.keys())
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

    print("Adding trials to NWB file")
    add_trials_to_nwb(nwb_file, simplified_trial_table)

    if context_timestamps_dict is not None:
        print("Adding epochs to NWB file")
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







import numpy as np
import pandas as pd
import yaml


def load_timestamps_data(trial_file, timestamps_dict):
    trial_list = np.load(trial_file, allow_pickle=True)
    trial_timestamps = timestamps_dict['trial_TTL']
    ci_timestamps = timestamps_dict['galvo_position']

    return trial_list, trial_timestamps, ci_timestamps


def get_trial_timestamps_dict(timestamps_dict, behavior_results_file, bhv_mapping_file):
    behavior_results = pd.read_csv(behavior_results_file)
    trial_outcomes = behavior_results['perf'].values
    trial_types = np.unique(trial_outcomes)
    trial_timestamps = np.array(timestamps_dict['trial_TTL'])
    trial_timestamps_dict = dict()
    trial_indexes_dict = dict()

    for trial_type in trial_types:
        trial_idx = np.where(trial_outcomes == trial_type)[0]
        trial_timestamps_dict[trial_type] = np.transpose(np.array(trial_timestamps[trial_idx]))
        trial_indexes_dict[trial_type] = trial_idx

    # Mapping between performance number and trial type
    with open(bhv_mapping_file, 'r') as stream:
        bhv_mapping_file_data = yaml.safe_load(stream)
    old_keys = list(trial_timestamps_dict.keys())
    for old_key in old_keys:
        trial_timestamps_dict[bhv_mapping_file_data[old_key]] = trial_timestamps_dict.pop(old_key)
        trial_indexes_dict[bhv_mapping_file_data[old_key]] = trial_indexes_dict.pop(old_key)

    return trial_timestamps_dict, trial_indexes_dict




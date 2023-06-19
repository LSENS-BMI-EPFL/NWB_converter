import os
import numpy as np
import pandas as pd
import yaml
import json


def find_training_days(subject_id, input_folder):
    sessions_list = os.listdir(os.path.join(input_folder, 'Training'))
    sessions_list = [s for s in sessions_list if os.path.isdir(os.path.join(input_folder, 'Training', s))]
    # Ordering in time with lexicographic ordering assumes %Y%m%d data format in session id.
    sessions_list = sorted(sessions_list)

    # Find session type (auditory or whisker day) and label days with integer from whisker day.
    behavior_type = []
    for isession in sessions_list:
        json_path = os.path.join(input_folder, 'Training', isession, 'session_config.json')
        with open(json_path, 'r') as f:
            json_config = json.load(f)
        behavior_type.append(json_config['behaviour_type'])
    n_aud = len([s for s in behavior_type if s in ['free_licking', 'auditory']])
    n_wh = len([s for s in behavior_type if s in ['whisker', 'context']])
    label = list(range(-n_aud, 0)) + list(range(0, n_wh))
    label = [f"+{d}" if d > 0 else str(d) for d in label]
    behavior_type = [f"{t}_{l}" for t, l in zip(behavior_type, label)]

    training_days = list(zip(sessions_list, behavior_type))

    return training_days


def load_timestamps_data(trial_file, timestamps_dict):
    trial_list = np.load(trial_file, allow_pickle=True)
    trial_timestamps = timestamps_dict['trial_TTL']
    ci_timestamps = timestamps_dict['galvo_position']

    return trial_list, trial_timestamps, ci_timestamps


def get_trial_timestamps_dict(timestamps_dict, behavior_results_file, config_file):
    if os.path.splitext(behavior_results_file)[1] == '.txt':
        sep = r'\s+'
    else:
        sep = ','
    behavior_results = pd.read_csv(behavior_results_file, sep=sep, engine='python')
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
    with open(config_file, 'r') as stream:
        bhv_mapping_file_data = yaml.safe_load(stream)['trial_map']
    old_keys = list(trial_timestamps_dict.keys())
    for old_key in old_keys:
        trial_timestamps_dict[bhv_mapping_file_data[old_key]] = trial_timestamps_dict.pop(old_key)
        trial_indexes_dict[bhv_mapping_file_data[old_key]] = trial_indexes_dict.pop(old_key)

    return trial_timestamps_dict, trial_indexes_dict


def list_trial_type(results_table):
    auditory_trials = np.where(results_table['is_auditory'])[0].astype(int)
    whisker_trials = np.where(results_table['is_whisker'])[0].astype(int)
    catch_trials = np.where(np.logical_not(results_table['is_stim']).astype(int))[0]

    n_trials = results_table['perf'].size

    trial_type_list = ["NA" for trial in range(n_trials)]
    for auditory_trial in auditory_trials:
        trial_type_list[auditory_trial] = "auditory"
    for whisker_trial in whisker_trials:
        trial_type_list[whisker_trial] = "whisker"
    for catch_trial in catch_trials:
        trial_type_list[catch_trial] = "catch"

    return trial_type_list


def build_simplified_trial_table(behavior_results_file, timestamps_dict):
    simplified_trial_table = pd.DataFrame()
    trial_table = pd.read_csv(behavior_results_file)
    trial_timestamps = np.array(timestamps_dict['trial_TTL'])
    trial_type_list = list_trial_type(results_table=trial_table)

    n_trials = trial_table['perf'].size
    print(f"Read '.csv' file to build trial NWB trial table ({n_trials} trials)")
    if len(trial_timestamps[:, 0]) > n_trials:
        print(f"csv table has one less trial than TTL up/down signal session must have been stop "
              f"before saving very last trial")
        trial_timestamps = trial_timestamps[0:-1, :]
    lick = trial_table['lick_flag'].values
    trial_outcome = ["Hit" if lick[trial] == 1 else "Miss" for trial in range(n_trials)]
    simplified_trial_table['trial_index'] = trial_table['trial_number']
    simplified_trial_table['trial_start'] = trial_timestamps[:, 0]
    simplified_trial_table['trial_stop'] = trial_timestamps[:, 1]
    simplified_trial_table['trial_type'] = trial_type_list
    simplified_trial_table['wh_reward'] = trial_table['wh_reward']
    simplified_trial_table['aud_reward'] = trial_table['aud_reward']
    simplified_trial_table['trial_outcome'] = trial_outcome
    simplified_trial_table['early_lick'] = trial_table['early_lick']
    simplified_trial_table['context_block'] = trial_table['context_block']

    return simplified_trial_table


def add_trials_to_nwb(nwb_file, simplified_trial_table):
    column_names = simplified_trial_table.columns
    columns_to_add = column_names[3:]

    for column in columns_to_add:
        nwb_file.add_trial_column(name=column, description="None")

    n_trials = simplified_trial_table['trial_type'].size
    for trial in range(n_trials):
        nwb_file.add_trial(start_time=simplified_trial_table['trial_start'].values[trial],
                           stop_time=simplified_trial_table['trial_stop'].values[trial],
                           trial_type=simplified_trial_table['trial_type'].values[trial],
                           wh_reward=simplified_trial_table['wh_reward'].values[trial],
                           aud_reward=simplified_trial_table['aud_reward'].values[trial],
                           trial_outcome=simplified_trial_table['trial_outcome'].values[trial],
                           early_lick=simplified_trial_table['early_lick'].values[trial],
                           context_block=simplified_trial_table['context_block'].values[trial])






import ast
import json
import os

import numpy as np
import pandas as pd
import scipy.io
import yaml


def read_excel_database(folder, file_name):
    excel_path = os.path.join(folder, file_name)
    database = pd.read_excel(excel_path, converters={'session_day': str})

    # Remove empty lines.
    database = database.loc[~database.isna().all(axis=1)]

    # Change yes/no columns to booleans.
    database = database.replace('yes', True)
    database = database.replace('no', False)
    database = database.astype({'2P_calcium_imaging': bool, 'optogenetic': bool,
                     'pharmacology': bool})

    return database


def format_session_day_GF(mouse_id, session_days):
    formated_days = []
    for iday in session_days:
        if iday[0] == '-':
            formated_days.append(f'auditory_{iday}')
        elif iday[0] in ['0', '+']:
            formated_days.append(f'whisker_{iday}')
        elif iday in ['whisker_on_1', 'whisker_off', 'whisker_on_2']:
            formated_days.append(iday)
        else:
            raise ValueError(f'Unrecognized session day {iday} for mouse {mouse_id}.')

    return formated_days


def infer_timestamps_dict(config_file):

    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    imouse = config['subject_metadata']['subject_id']
    isession = config['session_metadata']['identifier']

    exp_desc = ast.literal_eval(config['session_metadata']['experiment_description'])

    # If only behavior then return an empty dictionnary.
    # TODO: opto sessions will need synchronization.
    if exp_desc['session_type'] != 'twophoton_session':

        timestamps_dict = {
            'trial_TTL': [],
            'galvo_position': [],
            'cam1': [],
            'cam2': [],
        }
        n_frames_dict = {
            'trial_TTL': 0,
            'galvo_position': 0,
            'cam1': 0,
            'cam2': 0,
        }

        return timestamps_dict, n_frames_dict


    # Check that trial and calcium imaging time point numbers match across processed files.
    # #####################################################################################

    performance_json = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{imouse}\\Recordings\\BehaviourData\\{isession}\\performanceResults.json'
    with open(performance_json, 'r') as f:
        performance = json.load(f)
    performance = pd.DataFrame(performance['results'], columns=performance['headers'])

    nframes_file = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{imouse}\\Recordings\\FrameNumbers\\{isession}\\trialFrames.json'
    with open(nframes_file, 'r') as f:
        nframes_per_trial = json.load(f)
    nframes_per_trial = nframes_per_trial['trialFrames']

    # # Read number of calcium imaging tif files.
    # imaging_folder = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\data\\{imouse}\\Recordings\\Imaging\\{isession}'
    # tif_files = os.listdir(imaging_folder)
    # tif_files = [ifile for ifile in tif_files if os.path.splitext(ifile)[1] == '.tif']

    # suite2p_folder = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Georgios_Foustoukos\\Suite2PSessionData\\{imouse}\\{isession[:-7]}'
    # F = np.load(os.path.join(suite2p_folder, 'F.npy'), allow_pickle=True)

    # ntif_files = len(tif_files)
    # nperf_trials = performance.trial_number.iloc[-1]
    # nframes_suite2p = F.shape[1]
    # nframes_json = sum(nframes_per_trial)

    # if len(nframes_per_trial) != nperf_trials:
    #     raise ValueError('Number of trials in performance json and trialFrames json '
    #                     f'do not match for session {isession}.')
    # if ntif_files != nperf_trials:
    #     raise ValueError('Number of trials in performance json and tif files '
    #                     f'do not match for session {isession}.')
    # if nframes_suite2p != nframes_json:
    #     raise ValueError('Number of frames in json frame count and suite2p inputs'
    #                     f'are different for session {isession}')


    # Generate time stamps dict.
    # ##########################

    # Trial start and stop.
    # ---------------------

    # Trial start is defined as acquisition start of the first frame of each trial.
    trial_TTL_start = np.cumsum([0] + nframes_per_trial[:-1]) * 1/30
    # Stop is the acquisition end of last frame of each trial (so start of next one).
    trial_TTL_stop = np.cumsum(nframes_per_trial) * 1/30
    trial_TTL = list(zip(trial_TTL_start, trial_TTL_stop))


    # Galvo position.
    # ---------------

    # Cam 1.
    tot_nframes = np.sum(nframes_per_trial)
    galvo_position = np.arange(1, tot_nframes+1) * 1/30

    if os.path.exists(f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{imouse}\\Recordings\\FilmingData'):
        # Read frame counts.
        bad_frames = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{imouse}\\Recordings\\FilmingData\\{isession[:-7]}\\badFrames.npy'
        frame_count = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{imouse}\\Recordings\\FilmingData\\{isession[:-7]}\\framesNumber.npy'
        trial_index = f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{imouse}\\Recordings\\FilmingData\\{isession[:-7]}\\trialsNumber.npy'
        bad_frames = np.load(bad_frames, allow_pickle=True)
        frame_count = np.load(frame_count, allow_pickle=True)
        trial_index = np.load(trial_index, allow_pickle=True)

        cam1 = []

        # Use original trial numbers to map trials with imaging with trials with filming.
        for i, (istart, _) in enumerate(trial_TTL):
            itrial_or = performance.trial_number_or.iloc[i]
            idx = np.where(trial_index==itrial_or)[0][0]
            cam1.append(np.arange(1, frame_count[idx]+1) * 1/100 + istart)
    else:
        cam1 = []


    # Return timestamps and nframe dicts.
    # ###################################

    timestamps_dict = {
        'trial_TTL': trial_TTL,
        'galvo_position': galvo_position,
        'cam1': cam1,
        'cam2': [],
    }

    n_frames_dict = {
        'trial_TTL': len(trial_TTL),
        'galvo_position': len(galvo_position),
        'cam1': len(cam1),
        'cam2': len([]),
    }

    return timestamps_dict, n_frames_dict


def map_result_columns(behavior_results):

    column_map = {
        # Results.txt columns
        'trialnumber': 'trial_number',
        'WhStimDuration': 'wh_stim_duration',
        'Quietwindow': 'quiet_window',
        'ITI': 'iti',
        'Association': 'association_flag',
        'Stim/NoStim': 'is_stim',
        'Whisker/NoWhisker': 'is_whisker' ,
        'Auditory/NoAuditory': 'is_auditory',
        'Lick': 'lick_flag',
        'Perf': 'perf',
        'Light/NoLight': 'is_light',
        'ReactionTime': 'reaction_time',
        'WhStimAmp': 'wh_stim_amp',
        'TrialTime': 'trial_time',
        'Rew/NoRew': 'is_reward',
        'AudRew': 'aud_reward',
        'WhRew': 'wh_reward',
        'AudDur': 'aud_stim_duration',
        'AudDAmp': 'aud_stim_amp',
        'AudFreq': 'aud_stim_freq',
        'EarlyLick': 'early_lick',
        'LightAmp': 'light_amp',
        'LightDur': 'light_duration',
        'LightFreq': 'light_freq',
        'LightPreStim': 'light_prestim',

        # perf.json columns
        'trial_number': 'trial_number',
        'trial_number_or': 'trial_number_or',
        'whiskerstim_dur': 'wh_stim_duration',
        'quiet_window': 'quiet_window',
        'iti': 'iti',
        'stim_nostim': 'is_stim',
        'wh_nowh': 'is_whisker',
        'aud_noaud': 'is_auditory',
        'lick': 'lick_flag',
        'performance': 'perf',
        'light_nolight': 'is_light',
        'reaction_time': 'reaction_time',
        'whamp': 'wh_stim_amp',
        'trial_time': 'trial_time',
        'rew_norew': 'is_reward',
        'audrew': 'aud_reward',
        'whrew': 'wh_reward',
        'audstimdur': 'aud_stim_duration',
        'audstimamp': 'aud_stim_amp',
        'audstimfreq': 'aud_stim_freq',
        'early_lick': 'early_lick',
        'lightamp': 'light_amp',
        'lightdur': 'light_duration',
        'lightfreq': 'light_freq',
        'lightprestim': 'light_prestim',
    }

    f = lambda x:  column_map[x] if x in column_map.keys() else x
    behavior_results.columns = list(map(f, behavior_results.columns))
    behavior_results = behavior_results.astype({'trial_number': int, 'perf': int})

    # Deal with perf = -1. Label them as early licks.
    behavior_results.loc[behavior_results.perf==-1, 'perf'] = 6

    # Add what is missing.
    if 'association_flag' not in behavior_results.columns:
        behavior_results['association_flag'] = 0
    if 'baseline_window' not in behavior_results.columns:
        behavior_results['baseline_window'] = 2000
    if 'artifact_window' not in behavior_results.columns:
        behavior_results['artifact_window'] = 50
    if 'response_window' not in behavior_results.columns:
        behavior_results['response_window'] = 1000

    return behavior_results


# def load_mat_results(behavior_file):
#     mat = scipy.io.loadmat(behavior_file)
#     mat = mat['BehavResults']
#     mat = mat[0][0][0]

#     columns = [
#         'trial_number',
#         'trial_number_original',
#         'wh_stim_duration',
#         'quiet_window',
#         'iti',
#         'is_stim',
#         'is_whisker' ,
#         'is_auditory',
#         'lick_flag',
#         'perf',
#         'is_light',
#         'reaction_time',
#         'wh_stim_amp',
#         'trial_time',
#         'is_reward',
#         'aud_reward',
#         'wh_reward',
#         'aud_stim_duration',
#         'aud_stim_amp',
#         'aud_stim_freq',
#         'early_lick',
#         'light_amp',
#         'light_duration',
#         'light_freq',
#         'light_prestim',
#     ]

#     behavior_results = pd.DataFrame(mat, columns=columns)
#     behavior_results.drop('trial_number_original', axis=1)
#     behavior_results = behavior_results.astype({'trial_number': int, 'perf': int})

#     # Add what is missing.
#     if 'association_flag' not in behavior_results.columns:
#         behavior_results['association_flag'] = 0
#     if 'baseline_window' not in behavior_results.columns:
#         behavior_results['baseline_window'] = 2000
#     if 'artifact_window' not in behavior_results.columns:
#         behavior_results['artifact_window'] = 50
#     if 'response_window' not in behavior_results.columns:
#         behavior_results['response_window'] = 1000

#     return behavior_results

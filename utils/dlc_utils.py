import os
import numpy as np
import pandas as pd
import dlc2kinematics
import matplotlib.pyplot as plt

from scipy.signal import filtfilt, firwin

from PIL import Image


EXPERIMENTER_MAP = {
    'AR': '',
    'RD': '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Parameters/DLC_context/SV-07-051',
    'AB': '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Axel_Bisi/mice_info',
    'MH': '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Myriam_Hamon/mice_info',
    'MP': '',
    'PB': '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Parameters/DLC_context/SV-07-068',
    'MM': '',
    'MS': '',
    'GF': '',
    'MI': '',
}


def get_dlc_dataframe(dlc_file_path):

    side_path = [file for file in dlc_file_path if 'side' in file]
    top_path = [file for file in dlc_file_path if 'top' in file]

    if len(side_path) != 0:
        filetype = side_path[0].split('.')[-1]
        if filetype == 'csv':
            side_dlc = pd.read_csv(side_path[0], header=[1, 2])
            side_dlc = side_dlc.drop(('bodyparts', 'coords'), axis=1)
            side_dlc.columns = ["_".join(a) for a in side_dlc.columns.to_flat_index()]
        elif filetype == 'h5':
            side_dlc, bodyparts, scorer = dlc2kinematics.load_data(side_path[0], smooth=True, filter_window=3, order=1)
            side_dlc.columns = side_dlc.columns.droplevel(0) # remove multiindex
            side_dlc.columns = ["_".join(a) for a in side_dlc.columns.to_flat_index()] # change column names

    else:
        side_dlc = []

    if len(top_path) != 0:
        filetype = top_path[0].split('.')[-1]
        if filetype == 'csv':
            top_dlc = pd.read_csv(top_path[0], header=[1, 2])
            top_dlc = top_dlc.drop(('bodyparts', 'coords'), axis=1)
            top_dlc.columns = ["_".join(a) for a in top_dlc.columns.to_flat_index()]
        elif filetype == 'h5':
            top_dlc, bodyparts, scorer = dlc2kinematics.load_data(top_path[0], smooth=True, filter_window=3, order=1)
            top_dlc.columns = top_dlc.columns.droplevel(0)  # remove multiindex
            top_dlc.columns = ["_".join(a) for a in top_dlc.columns.to_flat_index()]  # change column names
    else:
        top_dlc = []

    return side_dlc, top_dlc


def get_reference_from_grid(config):
    """
    Get reference scale bar from grid image, for each view, per setup.
    :param config
    """
    experimenter = config['session_metadata']['experimenter']
    reference_folder = EXPERIMENTER_MAP[experimenter]

    if experimenter in ['AB', 'MH']:
        # Check if config has ephys metadata
        if 'ephys_metadata' in config.keys():
            setup = config['ephys_metadata']['setup']
            if setup == 'Neuropixels setup 1 AI3209':
                ref_path = os.path.join(reference_folder, 'reference_setup1.xlsx').replace('\\', '/')
            elif setup == 'Neuropixels setup 2 AI3209':
                ref_path = os.path.join(reference_folder, 'reference_setup2.xlsx').replace('\\', '/')
        else: # use a default reference
            ref_path = os.path.join(reference_folder, 'reference_setup1.xlsx').replace('\\', '/')

    else:
        ref_path = os.path.join(reference_folder, 'reference.xlsx').replace('\\', '/')

    if os.path.exists(ref_path):
        reference = pd.read_excel(ref_path)
    else:
        ValueError(f"No video scalebar reference.xlsx found in folder {ref_path}")

    return reference


def compute_kinematics(df, view):

    if view == 'sideview' and len(df) != 0:

        ## Jaw kinematics

        ref_x = np.percentile(np.where(df['jaw_likelihood'] > 0.8, df['jaw_x'], 0), 5)
        ref_y = np.percentile(np.where(df['jaw_likelihood'] > 0.8, df['jaw_y'], 0), 5)
        df.loc[:, 'jaw_angle'] = np.degrees(np.arcsin((df['jaw_y'] - ref_y) /
                                                           (np.sqrt((df['jaw_y'] - ref_y) ** 2 + (
                                                                   df['jaw_x'] - df['jaw_ref_x'].median()) ** 2))))
        df.loc[:, 'jaw_distance'] = np.sqrt((df['jaw_y'] - ref_y) ** 2 + (df['jaw_x'] - ref_x) ** 2)
        df.loc[:, 'jaw_velocity'] = np.zeros_like(df['jaw_distance'])
        df.loc[1:, 'jaw_velocity'] = np.diff(df['jaw_distance']) # Note: the correct scaling with the time period is done later, where camera rate is known

        ## Tongue kinematics

        df.loc[:, 'tongue_angle'] = np.degrees(np.arcsin((df['tongue_y'] - ref_y) /
                                                           (np.sqrt((df['tongue_y'] - ref_y) ** 2 + (
                                                                   df['tongue_x'] - df['jaw_ref_x'].median()) ** 2))))
        df.loc[:, 'tongue_distance'] = np.sqrt((df['tongue_y'] - ref_y) ** 2 + (df['tongue_x'] - ref_x) ** 2)
        df.loc[:, 'tongue_velocity'] = np.zeros_like(df['tongue_distance'])
        df.loc[1:, 'tongue_velocity'] = np.diff(df['tongue_distance'])

        ## Nose kinematics

        ref_x = np.percentile(np.where(df['nose_likelihood'] > 0.8, df['nose_x'], 0), 5)
        ref_y = np.percentile(np.where(df['nose_likelihood'] > 0.8, df['nose_y'], 0), 5)
        df.loc[:, 'nose_angle'] = np.degrees(np.arcsin((df['nose_y'] - ref_y) /
                                                           (np.sqrt((df['nose_y'] - ref_y) ** 2 + (
                                                                   df['nose_x'] - df['nose_base_x'].median()) ** 2))))
        df.loc[:, 'nose_distance'] = np.sqrt((df['nose_y'] - ref_y) ** 2 + (df['nose_x'] - ref_x) ** 2)
        df.loc[:, 'nose_velocity'] = np.zeros_like(df['nose_distance'])
        df.loc[1:, 'nose_velocity'] = np.diff(df['nose_distance'])

        ## Pupil

        df.loc[:, 'pupil_area'] = df.apply(lambda x: 0.5 * np.abs(
            (np.dot(x['pupil_top_x'], x['pupil_right_y']) - np.dot(x['pupil_top_y'], x['pupil_right_x'])) +
            (np.dot(x['pupil_right_x'], x['pupil_bottom_y']) - np.dot(x['pupil_right_y'], x['pupil_bottom_x'])) +
            (np.dot(x['pupil_bottom_x'], x['pupil_left_y']) - np.dot(x['pupil_bottom_y'], x['pupil_left_x'])) +
            (np.dot(x['pupil_left_x'], x['pupil_top_y']) - np.dot(x['pupil_left_y'], x['pupil_top_x']))), axis=1
                                                )
        df.loc[:, 'pupil_likelihood'] = (df.loc[:, 'pupil_top_likelihood'] > 0.9) & \
                                             (df.loc[:, 'pupil_right_likelihood'] > 0.9) & \
                                             (df.loc[:, 'pupil_bottom_likelihood'] > 0.9) & \
                                             (df.loc[:, 'pupil_left_likelihood'] > 0.9)

        df.loc[:, 'pupil_area_velocity'] = np.zeros_like(df['pupil_area'])
        df.loc[1:, 'pupil_area_velocity'] = np.diff(df['pupil_area'])

    elif view == 'topview' and len(df) != 0:
        ## TODO: to be tested
        ## Whisker kinematics

        dot_prod = (df['whisker_tip_x'] - df['whisker_base_x']) * (df['nose_tip_x'].median() - df['nose_base_x'].median()) + (
                    df['whisker_base_y'] - df['whisker_tip_y']) * (df['nose_tip_y'].median() - df['nose_base_y'].median())

        magnitude1 = np.sqrt((df['whisker_tip_x'] - df['whisker_base_x']) ** 2 + (
                    df['whisker_tip_y'] - df['whisker_base_y']) ** 2)
        magnitude2 = np.sqrt((df['nose_tip_x'].median() - df['nose_base_x'].median()) ** 2 + (
                    df['nose_tip_y'].median() - df['nose_base_y'].median()) ** 2)

        df.loc[:, 'whisker_angle'] = np.degrees(np.arccos(dot_prod/(magnitude1*magnitude2)))
        df.loc[:, 'whisker_velocity'] = np.zeros_like(df['whisker_angle'])
        df.loc[1:, 'whisker_velocity'] = np.diff(df['whisker_angle'])

        ## Nose_top kinematics

        ref_x = np.median(np.where(df['nose_tip_likelihood'] > 0.8, df['nose_tip_x'], 0))
        ref_y = np.median(np.where(df['nose_tip_likelihood'] > 0.8, df['nose_tip_y'], 0))

        df.loc[:, 'nose_angle'] = np.degrees(np.arcsin((ref_x - df['nose_tip_x']) /
                                                           (np.sqrt((ref_x - df['nose_tip_x']) ** 2 + (
                                                                   ref_y - df['nose_base_y'].median()) ** 2))))

        df.loc[:, 'nose_distance'] = np.sqrt((df['nose_tip_y'] - ref_y) ** 2 + (df['nose_tip_x'] - ref_x) ** 2)
        df.loc[:, 'nose_velocity'] = np.zeros_like(df['nose_distance'])
        df.loc[1:, 'nose_velocity'] = np.diff(df['nose_distance'])

    return df

def compute_kinematics_alt(df, view):

    pcutoff = 0.5

    if view == 'sideview' and len(df) != 0:
        keep_cols  = [
            'pupil_top_x', 'pupil_top_y', 'pupil_top_likelihood',
             'pupil_left_x','pupil_left_y', 'pupil_left_likelihood',
            'pupil_right_x', 'pupil_right_y', 'pupil_right_likelihood',
            'pupil_bottom_x', 'pupil_bottom_y', 'pupil_bottom_likelihood',
            'nose_tip_x', 'nose_tip_y', 'nose_tip_likelihood',
            'nose_base_x', 'nose_base_y', 'nose_base_likelihood',
            'jaw_x', 'jaw_y', 'jaw_likelihood',
            'jaw_ref_x', 'jaw_ref_y', 'jaw_ref_likelihood',
            'spout_x', 'spout_y', 'spout_likelihood',
            'particle_x', 'particle_y', 'particle_likelihood',
            'tongue_x', 'tongue_y', 'tongue_likelihood'
        ]
        df = df[keep_cols]

        ## Jaw kinematics
        ref_x = np.percentile(np.where(df['jaw_likelihood'] > pcutoff, df['jaw_x'], 0), 5)
        ref_y = np.percentile(np.where(df['jaw_likelihood'] > pcutoff, df['jaw_y'], 0), 5)
        df.loc[:, 'jaw_angle'] = np.degrees(np.arcsin((df['jaw_y'] - ref_y) /
                                                           (np.sqrt((df['jaw_y'] - ref_y) ** 2 + (
                                                                   df['jaw_x'] - df['jaw_ref_x'].median()) ** 2))))
        df.loc[:, 'jaw_distance'] = np.sqrt((df['jaw_y'] - ref_y) ** 2 + (df['jaw_x'] - ref_x) ** 2)
        df.loc[:, 'jaw_velocity'] = np.zeros_like(df['jaw_distance'])
        df.loc[1:, 'jaw_velocity'] = np.diff(df['jaw_distance'])  # Note: the correct scaling with the time period is done later, where camera rate is known

        ## Tongue kinematics
        df.loc[:, 'tongue_angle'] = np.degrees(np.arcsin((df['tongue_y'] - ref_y) /
                                                           (np.sqrt((df['tongue_y'] - ref_y) ** 2 + (
                                                                   df['tongue_x'] - df['jaw_ref_x'].median()) ** 2))))
        #df.loc[:, 'tongue_distance'] = np.sqrt((df['tongue_y'] - ref_y) ** 2 + (df['tongue_x'] - ref_x) ** 2)
        df.loc[:, 'tongue_distance'] = np.sqrt(df['tongue_y'] ** 2 + df['tongue_x'] ** 2)
        df.loc[:, 'tongue_velocity'] = np.zeros_like(df['tongue_distance'])
        df.loc[1:, 'tongue_velocity'] = np.diff(df['tongue_distance'])

        ## Nose kinematics
        ref_x = np.percentile(np.where(df['nose_tip_likelihood'] > pcutoff, df['nose_tip_x'], 0), 5)
        ref_y = np.percentile(np.where(df['nose_tip_likelihood'] > pcutoff, df['nose_tip_y'], 0), 5)
        df.loc[:, 'nose_angle'] = np.degrees(np.arcsin((df['nose_tip_y'] - ref_y) /
                                                           (np.sqrt((df['nose_tip_y'] - ref_y) ** 2 + (
                                                                   df['nose_tip_x'] - df['nose_base_x'].median()) ** 2))))
        df.loc[:, 'nose_distance'] = np.sqrt((df['nose_tip_y'] - ref_y) ** 2 + (df['nose_tip_x'] - ref_x) ** 2)
        df.loc[:, 'nose_velocity'] = np.zeros_like(df['nose_distance'])
        df.loc[1:, 'nose_velocity'] = np.diff(df['nose_distance'])

        ## Pupil
        df.loc[:, 'pupil_area'] = df.apply(lambda x: 0.5 * np.abs(
            (np.dot(x['pupil_top_x'], x['pupil_right_y']) - np.dot(x['pupil_top_y'], x['pupil_right_x'])) +
            (np.dot(x['pupil_right_x'], x['pupil_bottom_y']) - np.dot(x['pupil_right_y'], x['pupil_bottom_x'])) +
            (np.dot(x['pupil_bottom_x'], x['pupil_left_y']) - np.dot(x['pupil_bottom_y'], x['pupil_left_x'])) +
            (np.dot(x['pupil_left_x'], x['pupil_top_y']) - np.dot(x['pupil_left_y'], x['pupil_top_x']))), axis=1
                                                )
        df.loc[:, 'pupil_likelihood'] = (df.loc[:, 'pupil_top_likelihood'] > pcutoff) & \
                                             (df.loc[:, 'pupil_right_likelihood'] > pcutoff) & \
                                             (df.loc[:, 'pupil_bottom_likelihood'] > pcutoff) & \
                                             (df.loc[:, 'pupil_left_likelihood'] > pcutoff)

        df.loc[:, 'pupil_area_velocity'] = np.zeros_like(df['pupil_area'])
        df.loc[1:, 'pupil_area_velocity'] = np.diff(df['pupil_area'])

        debug = False
        bodyparts = ['jaw_distance', 'tongue_distance', 'nose_distance', 'pupil_area']
        bodyparts_diff = ['jaw_velocity', 'tongue_velocity', 'nose_velocity', 'pupil_area_velocity']
        if debug:
            start_sec = 2000
            end_sec = 2100
            start_idx = int(start_sec * 200)
            end_idx = int(end_sec * 200)
            n_bodyparts = len(bodyparts)
            # Plot all bodyparts
            fig, axs = plt.subplots(n_bodyparts, 1, figsize=(10, 10))
            for i, (bp, bp_diff) in enumerate(zip(bodyparts, bodyparts_diff)):
                axs[i].plot(df[bp], label=bp)
                axs[i].plot(df[bp_diff], label=bp_diff)
                axs[i].legend()

            fig.tight_layout()
            #plt.xlim(start_idx, end_idx)
            plt.show()

    elif view == 'topview' and len(df) != 0:
        keep_cols = [
            'nose_tip_x', 'nose_tip_y', 'nose_tip_likelihood',
            'nose_base_x', 'nose_base_y', 'nose_base_likelihood',
            'particle_x', 'particle_y', 'particle_likelihood',
            'whisker_base_x', 'whisker_base_y', 'whisker_base_likelihood',
            'whisker_tip_x', 'whisker_tip_y', 'whisker_tip_likelihood'
        ]
        df = df[keep_cols]

        ## Whisker kinematics
        dot_prod = (df['particle_x'] - df['whisker_base_x']) * (df['nose_tip_x'].median() - df['nose_base_x'].median()) + (
                    df['whisker_base_y'] - df['particle_y']) * (df['nose_tip_y'].median() - df['nose_base_y'].median())

        magnitude1 = np.sqrt((df['particle_x'] - df['whisker_base_x']) ** 2 + (
                    df['particle_y'] - df['whisker_base_y']) ** 2)
        magnitude2 = np.sqrt((df['nose_tip_x'].median() - df['nose_base_x'].median()) ** 2 + (
                    df['nose_tip_y'].median() - df['nose_base_y'].median()) ** 2)

        df.loc[:, 'whisker_angle'] = np.degrees(np.arccos(dot_prod/(magnitude1*magnitude2)))
        df.loc[:, 'whisker_velocity'] = np.zeros_like(df['whisker_angle'])
        df.loc[1:, 'whisker_velocity'] = np.diff(df['whisker_angle'])

        ## Nose_top kinematics
        ref_x = np.median(np.where(df['nose_tip_likelihood'] > pcutoff, df['nose_tip_x'], 0))
        ref_y = np.median(np.where(df['nose_tip_likelihood'] > pcutoff, df['nose_tip_y'], 0))

        df.loc[:, 'nose_angle'] = np.degrees(np.arcsin((ref_x - df['nose_tip_x']) /
                                                           (np.sqrt((ref_x - df['nose_tip_x']) ** 2 + (
                                                                   ref_y - df['nose_base_y'].median()) ** 2))))

        df.loc[:, 'nose_distance'] = np.sqrt((df['nose_tip_y'] - ref_y) ** 2 + (df['nose_tip_x'] - ref_x) ** 2)
        df.loc[:, 'nose_velocity'] = np.zeros_like(df['nose_distance'])
        df.loc[1:, 'nose_velocity'] = np.diff(df['nose_distance'])

        debug = False
        bodyparts = ['whisker_angle', 'nose_distance']
        bodyparts_diff = ['whisker_velocity', 'nose_velocity']
        if debug:
            start_sec = 2000
            end_sec = 2100
            start_idx = int(start_sec * 200)
            end_idx = int(end_sec * 200)
            n_bodyparts = len(bodyparts)
            # Plot all bodyparts
            fig, axs = plt.subplots(n_bodyparts, 1, figsize=(10, 10))
            for i, (bp, bp_diff) in enumerate(zip(bodyparts, bodyparts_diff)):
                axs[i].plot(df[bp], label=bp)
                axs[i].plot(df[bp_diff], label=bp_diff)
                axs[i].legend()

            fig.tight_layout()
            #plt.xlim(start_idx, end_idx)
            plt.show()

    return df

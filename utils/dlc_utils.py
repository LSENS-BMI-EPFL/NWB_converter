import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, firwin

from PIL import Image


EXPERIMENTER_MAP = {
    'AR': '',
    'RD': '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Parameters/DLC_context/SV-07-051',
    'AB': '',
    'MP': '',
    'PB': '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Parameters/DLC_context/SV-07-068',
    'MM': '',
    'MS': '',
    'GF': '',
    'MI': '',
}


def get_dlc_dataframe(dlc_file_path):

    side_path = [file for file in dlc_file_path if 'sideview' in file]
    top_path = [file for file in dlc_file_path if 'topview' in file]
    if len(side_path) != 0:
        side_dlc = pd.read_csv(side_path[0], header=[1, 2])
        side_dlc = side_dlc.drop(('bodyparts', 'coords'), axis=1)
        side_dlc.columns = ["_".join(a) for a in side_dlc.columns.to_flat_index()]

    else:
        side_dlc = []

    if len(top_path) != 0:
        top_dlc = pd.read_csv(top_path[0], header=[1, 2])
        top_dlc = top_dlc.drop(('bodyparts', 'coords'), axis=1)
        top_dlc.columns = ["_".join(a) for a in top_dlc.columns.to_flat_index()]

    else:
        top_dlc = []

    return side_dlc, top_dlc


def get_reference_from_grid(experimenter):
    reference_folder = EXPERIMENTER_MAP[experimenter]
    ref_path = os.path.join(reference_folder, 'reference.xlsx').replace('\\', '/')

    if os.path.exists(ref_path):
        reference = pd.read_excel(ref_path)
    else:
        ValueError(f"No reference.xlsx found in folder {ref_path}")

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
        df.loc[1:, 'jaw_velocity'] = np.diff(df['jaw_distance'])

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

    elif view == 'top' and len(df) != 0:
        ## TODO: to be tested
        ## Whisker kinematics

        dot_prod = (df['whisker_tip_x'] - df['whisker_base_x']) * (df['nose_tip_x'].median() - df['nose_base_x'].median()) + (
                    df['whisker_tip_y'] - df['whisker_base_y']) * (df['nose_tip_y'].median() - df['nose_base_y'].median())

        magnitude1 = np.sqrt((df['whisker_tip_x'] - df['whisker_base_x']) ** 2 + (
                    df['whisker_tip_y'] - df['whisker_base_y']) ** 2)
        magnitude2 = np.sqrt((df['nose_tip_x'].median() - df['nose_base_x'].median()) ** 2 + (
                    df['nose_tip_y'].median() - df['nose_base_y'].median()) ** 2)

        df.loc[:, 'whisker_angle'] = np.degrees(np.arccos(dot_prod/(magnitude1*magnitude2)))
        df.loc[:, 'whisker_velocity'] = np.zeros_like(df['whisker_angle'])
        df.loc[1:, 'whisker_velocity'] = np.diff(df['whisker_angle'])

        ## Nose_top kinematics

        ref_x = np.where(df['nose_tip_likelihood'] > 0.8, df['nose_tip_x'], 0).median()
        ref_y = np.where(df['nose_tip_likelihood'] > 0.8, df['nose_tip_y'], 0).median()

        df.loc[:, 'nose_top_angle'] = np.degrees(np.arcsin((df['nose_tip_x'] - ref_y) /
                                                           (np.sqrt((df['nose_tip_x'] - ref_x) ** 2 + (
                                                                   df['nose_tip_y'] - df['nose_base_y'].median()) ** 2))))

        df.loc[:, 'nose_top_distance'] = np.sqrt((df['nose_tip_y'] - ref_y) ** 2 + (df['nose_tip_x'] - ref_x) ** 2)
        df.loc[:, 'nose_top_velocity'] = np.zeros_like(df['nose_tip_distance'])
        df.loc[1:, 'nose_top_velocity'] = np.diff(df['nose_tip_distance'])

    return df


def compute_jaw_opening_epoch(df):
    nfilt = 100  # Number of taps to use in FIR filter
    fw_base = 5  # Cut-off frequency for lowpass filter, in Hz
    nyq_rate = 200 / 2.0
    cutoff = min(1.0, fw_base / nyq_rate)
    b = firwin(nfilt, cutoff=cutoff, window='hamming')
    padlen = 3 * nfilt
    filtered_jaw = filtfilt(b, [1.0], df['jaw_angle'], axis=0,
                            padlen=padlen)
    jaw_opening = np.where(filtered_jaw < 1.8*filtered_jaw.std(), np.zeros_like(filtered_jaw), 1)

    return jaw_opening


def compute_whisker_movement_epoch(df): ## TODO: revisit combination of parameters
    nfilt = 100  # Number of taps to use in FIR filter
    fw_base = 5  # Cut-off frequency for lowpass filter, in Hz
    nyq_rate = 200 / 2.0
    cutoff = min(1.0, fw_base / nyq_rate)
    b = firwin(nfilt, cutoff=cutoff, window='hamming')
    padlen = 3 * nfilt
    filtered_wh = filtfilt(b, [1.0], df['whisker_angle'], axis=0,
                            padlen=padlen)

    return np.where(filtered_wh < 1.8*filtered_wh.std(), np.zeros_like(filtered_wh), 1)
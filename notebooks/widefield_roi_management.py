import yaml
import math
import os
import numpy as np
import pandas as pd
import imageio as iio
from PIL import Image
from utils import server_paths
from continuous_log_analysis import analyze_continuous_log
from utils.behavior_converter_misc import (build_simplified_trial_table,
                                           build_standard_trial_table)


def find_nearest(array, value, is_sorted=True):
    """
    Return the index of the nearest content in array of value.
    from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    return -1 or len(array) if the value is out of range for sorted array
    Args:
        array:
        value:
        is_sorted:

    Returns:

    """
    if len(array) == 0:
        return -1

    if is_sorted:
        if value < array[0]:
            return -1
        elif value > array[-1]:
            return len(array)
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx
    else:
        array = np.asarray(array)
        idx = (np.abs(array - value)).idxmin()
        return idx


def align_image(img, start, dest):

    orig_x, orig_y = start[0], start[1]
    dest_x, dest_y = dest[0], dest[1]
    trans_x, trans_y = dest_x - orig_x, dest_y - orig_y
    margin_x = img.shape[1] - np.abs(trans_x)
    margin_y = img.shape[0] - np.abs(trans_y)
    if trans_y >= 0 and trans_x >= 0:
        img[trans_y:, trans_x:] = img[:margin_y, :margin_x]
        img[:trans_y, :] *= 0
        img[:, :trans_x] *= 0
    elif trans_y < 0 and trans_x >= 0:
        img[:trans_y, trans_x:] = img[-margin_y:, :margin_x]
        img[trans_y:, :] *= 0
        img[:, :trans_x] *= 0
    elif trans_y >= 0 and trans_x < 0:
        img[trans_y:, :trans_x] = img[:margin_y, -margin_x:]
        img[:trans_y, :] *= 0
        img[:, trans_x:] *= 0
    else:
        img[:trans_y, :trans_x] = img[-margin_y:, -margin_x:]
        img[trans_y:, :] *= 0
        img[:, trans_x:] *= 0

    return img


def get_alignment_reference(session_id, align_to='bregma'):
    analysis_folder = server_paths.get_subject_analysis_folder(session_id.split("_")[0])
    file = os.path.join(analysis_folder, 'alignment_ref.csv')
    reference_list = pd.read_csv(file)

    if session_id not in reference_list['session_id']:
        ValueError(f"{session_id} has no data in the bregma reference, run make_alignment_reference")

    x = reference_list.loc[reference_list['session_id'] == session_id, f"{align_to}_x"].values[0]
    y = reference_list.loc[reference_list['session_id'] == session_id, f"{align_to}_y"].values[0]

    return x, y


def get_wf_scalebar(scale=1):
    x = [62*scale, 167*scale]
    y = [162*scale, 152*scale]
    c = np.sqrt((x[1] - x[0]) ** 2 + (y[0] - y[1]) ** 2)

    return round(c / 6)


def extract_trial_average_images(config_file):
    # Get widefield timestamps
    timestamps_dict, _ = analyze_continuous_log(config_file=config_file,
                                                do_plot=False, plot_start=4500,
                                                plot_stop=4700, camera_filtering=False)
    wf_frames_ts = timestamps_dict['widefield']
    wf_ts = np.array([ts[0] for ts in wf_frames_ts])

    # Get trial table
    behavior_results_file = server_paths.get_behavior_results_file(config_file)
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    if config.get('behaviour_metadata').get('trial_table') == 'standard':
        trial_table = build_standard_trial_table(
            config_file=config_file,
            behavior_results_file=behavior_results_file,
            timestamps_dict=timestamps_dict
        )
    else:
        trial_table = build_simplified_trial_table(behavior_results_file=behavior_results_file,
                                                   timestamps_dict=timestamps_dict)

    # Build dict for whisker & auditory trial
    print("Get auditory and whisker trials")
    wh_trial_ts = trial_table.loc[trial_table.whisker_stim == 1].start_time.values[:].tolist()
    aud_trial_ts = trial_table.loc[trial_table.auditory_stim == 1].start_time.values[:].tolist()
    trial_dict = {'whisker': wh_trial_ts, 'auditory': aud_trial_ts}
    print(f"{len(wh_trial_ts)} whisker trials, {len(aud_trial_ts)} auditory trials")

    # Get path to WF file
    print("Extract individual images")
    wf_file = server_paths.get_widefield_file(config_file=config_file)
    if len(wf_file) > 1:
        wf_file = [file for file in wf_file if session in os.path.basename(file)]
    wf_file = wf_file[0]

    # Get trial average image:
    img_dict = {'whisker': [], 'auditory': []}
    for key, data in trial_dict.items():
        print(f"- {key} trials")
        for trial_ts in data:
            # Get baseline
            baseline_frames = np.arange(find_nearest(wf_ts, trial_ts) - 10, find_nearest(wf_ts, trial_ts))
            baseline_images = []
            for baseline_frame in baseline_frames:
                baseline_images.append(iio.v3.imread(wf_file, plugin='pyav', format='gray16be', index=baseline_frame))
            baseline_images = np.stack(baseline_images, axis=0)
            f_null = np.nanmean(baseline_images, axis=0)

            # Get single frames
            frames = np.arange(find_nearest(wf_ts, trial_ts), find_nearest(wf_ts, trial_ts) + 7)
            response_frames = []
            for frame in frames:
                response_frames.append(iio.v3.imread(wf_file, plugin='pyav', format='gray16be', index=frame))
            response_frames = np.stack(response_frames, axis=0)
            img = np.nanmean(response_frames, axis=0)

            # Subtract baseline
            df_img = img - f_null

            # Translate image
            start = get_alignment_reference(session, align_to='bregma')
            dest = (175, 240)
            df_img = align_image(df_img, start=start, dest=dest)

            # Reshape image
            df_img = df_img.reshape(-1, int(df_img.shape[0] / 2), 2, int(df_img.shape[1] / 2), 2).mean(axis=0).mean(axis=1).mean(axis=2)

            # Append to dict
            img_dict[key].append(df_img)

    # Produce average :
    print("Produce average images")
    subject_id = config_dict['subject_metadata']['subject_id']
    saving_path = os.path.join(server_paths.get_subject_analysis_folder(subject_id), session,
                               f'{session}_trial_average_images')
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    for key, data in img_dict.items():
        imgs = np.stack(data, axis=0)
        avg_img = np.nanmean(imgs, axis=0)
        iio.v3.imwrite(os.path.join(saving_path, f'{session}_{key}_trial_image.tiff'), avg_img)


# Select the sessions ID to do
group_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/group.yaml"
with open(group_file, 'r', encoding='utf8') as stream:
    group_dict = yaml.safe_load(stream)
# sessions = group_dict['NWB_CI_LSENS']['Context_expert_sessions']
# sessions = group_dict['NWB_CI_LSENS']['Context_good_params']
# sessions = group_dict['NWB_CI_LSENS']['context_expert_widefield']
# sessions = group_dict['NWB_CI_LSENS']['Context_contrast_expert']
sessions = group_dict['NWB_CI_LSENS']['context_contrast_widefield']
session_to_do = [session[0] for session in sessions]


for session in session_to_do:
    mouse_id = session[0:5]
    data_folder = server_paths.get_subject_data_folder(mouse_id)
    analysis_folder = server_paths.get_subject_analysis_folder(mouse_id)
    config_yaml = os.path.join(analysis_folder, session, f"config_{session}.yaml")
    if not os.path.exists(config_yaml):
        continue
    with open(config_yaml, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)
    if 'widefield_metadata' not in list(config_dict.keys()):
        continue

    print(" ")
    print(f"Session : {session}")
    extract_trial_average_images(config_yaml)





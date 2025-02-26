import os
import h5py
import time
import sys
import scipy
import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
import numpy as np
# import dask.array as da
import imageio as iio
import gc
gc.collect()
import matplotlib.pyplot as plt
from utils.server_paths import get_subject_analysis_folder, get_widefield_file



def read_motion_jpeg_2000_movie(mj2_file):
    """
    Open widefield movie file with imageio and return the number of frames and the frame rate.
    Args:
        mj2_file: path to movie file

    Returns:

    """
    import imageio as iio

    props = iio.v3.improps(mj2_file, plugin='pyav', format='gray16be')
    fps = iio.v3.immeta(mj2_file, plugin='pyav')['fps']

    print(" ")
    print(f"Widefield video frames: {props.n_images}, @ {fps} Hz")
    return props.n_images, fps



def compute_F0_CardinLAB(F):
    nfilt = 1000
    fw_base = 0.001
    fs = 100
    nyq_rate = fs / 2.0
    cutoff = min(0.001, fw_base / nyq_rate)
    padlen = min(3 * nfilt, F.shape[0] - 1)
    b = scipy.signal.firwin(nfilt, cutoff=cutoff, window='hamming')
    F0 = scipy.signal.filtfilt(b, [1.0], F, axis=0,
                               padlen=padlen)

    return F0


def compute_F0_percentile(F, winsize=500):
    F0 = []
    for i in tqdm.tqdm(range(0, F.shape[0], winsize)):
        percentile = np.nanpercentile(F[i:i + winsize], 5, axis=0)
        F0.append(np.tile(percentile, (winsize, 1, 1)))

    return np.vstack(F0)[:F.shape[0], :, :]


def compute_F0_early_percentile(F, winsize=500):

    return np.nanpercentile(F[:winsize], 5, axis=0)


def compute_dff0(data_folder, method='percentile'):

    start = time.time()
    print("Open F file to compute F0 on full recording ... ")
    F_file = h5py.File(os.path.join(data_folder, 'F_data.h5'), 'r')
    F = F_file['F'][:]

    print("Computing F0 ...")

    if method == 'low_pass_filter':
        F0 = compute_F0_CardinLAB(F)
    elif method == 'percentile':
        F0 = compute_F0_early_percentile(F, winsize=F.shape[0])
    F_dims = F.shape

    F_file.close()
    del F
    gc.collect()

    print("Saving F0 ... ")
    iio.imwrite(os.path.join(data_folder, 'F0.tiff'), F0)

    print("Computing dff0 ... ")
    F_file = h5py.File(os.path.join(data_folder, 'F_data.h5'), 'r')
    dff0 = np.zeros(F_dims)
    n_chunks = 20
    bloc_size = F_dims[0] // 20
    for chunk in range(n_chunks):
        if chunk == 0:
            dff0[0: bloc_size, :, :] = (F_file['F'][0: bloc_size] - F0) / F0
        else:
            start_slice = chunk * bloc_size
            stop_slice = (chunk+1) * bloc_size
            dff0[start_slice: stop_slice, :, :] = (F_file['F'][start_slice: stop_slice] - F0) / F0
    dff0[n_chunks * bloc_size:, :, :] = (F_file['F'][n_chunks * bloc_size:] - F0) / F0

    end = time.time()
    print("dFF0 calculation took %0.4f min" % ((end - start) / 60))

    F_file.close()

    return dff0, F0


def load_array(file, frame):
    return iio.v3.imread(file, plugin='pyav', format='gray16be', index=frame)


def transpose_reduce(data):
    return data.reshape(int(data.shape[0] / 2), 2, int(data.shape[1] / 2), 2).mean(axis=1).mean(axis=2)


def concat_and_save(file, wf_frame_timestamps, output_folder):
    results = []
    start = 0
    props = iio.v3.improps(file, plugin='pyav', format='gray16be')
    with h5py.File(output_folder + r'\F_data.h5', 'w') as f:
        wf_dataset = f.create_dataset('F', (props.shape[0], props.shape[1] / 2, props.shape[2] / 2), chunks=True, dtype='uint16')

        for idx, frame in tqdm.tqdm(enumerate(iio.v3.imiter(file, plugin='pyav', format='gray16be'))):
            if idx > len(wf_frame_timestamps):
                print(" ")
                print("More frames than timestamps")
                break

            results.append(frame)

            if idx > 1 and idx % 10000 == 0:
                data = np.stack(results)
                data = data.reshape(-1, int(data.shape[1] / 2), 2, int(data.shape[2] / 2), 2).mean(axis=2).mean(axis=3)
                wf_dataset[start:start + data.shape[0]] = data
                start += data.shape[0] + 1
                results = []

        data = np.stack(results)
        data = data.reshape(-1, int(data.shape[1] / 2), 2, int(data.shape[2] / 2), 2).mean(axis=2).mean(axis=3)
        wf_dataset[start:start + data.shape[0]] = data

    return output_folder + r'\F_data.h5'


def concat_inmemory_and_save(file, wf_frame_timestamps, output_folder, align_to='bregma'):
    start_time = time.time()
    props = iio.v3.improps(file, plugin='pyav', format='gray16be')

    print(" ")
    print("Loading widefield calcium imaging data")

    vid = iio.v3.imread(file, plugin='pyav', format='gray16be')
    print("Loaded widefield calcium imaging data")
    vid_size = sys.getsizeof(vid)
    print(f"Video size : {np.round(vid_size / 1000000000, 2)} Gb")

    session_id = file.split("\\")[-1].split(".")[0]
    start = get_alignment_reference(session_id, align_to=align_to)
    dest = (175, 240)
    vid = align_videos(vid, start=start, dest=dest)
    print("Widefield calcium imaging data is aligned to reference")

    n_frames = vid.shape[0]
    end_frames = n_frames % 20
    end_vid = vid[(n_frames-end_frames):, :, :]
    vid = np.split(vid[:(n_frames-end_frames), :, :], indices_or_sections=20, axis=0)
    new_vid = np.zeros((int(n_frames), int(vid[1].shape[1] / 2), int(vid[1].shape[2] / 2)))
    for i in range(len(vid)):
        tmp_vid = vid.pop(0)
        if i == 0:
            new_vid[0: tmp_vid.shape[0], :, :] = tmp_vid.reshape(-1, int(tmp_vid.shape[1] / 2), 2, int(tmp_vid.shape[2] / 2), 2).mean(axis=2).mean(axis=3)
        else:
            new_vid[i * tmp_vid.shape[0]: (i+1) * tmp_vid.shape[0], :, :] = tmp_vid.reshape(-1, int(tmp_vid.shape[1] / 2), 2, int(tmp_vid.shape[2] / 2), 2).mean(axis=2).mean(axis=3)
        del tmp_vid
        gc.collect()
    del vid
    gc.collect()
    new_vid[20*(n_frames//20):, :, :] = end_vid.reshape(-1, int(end_vid.shape[1] / 2), 2, int(end_vid.shape[2] / 2), 2).mean(axis=2).mean(axis=3)
    print("Widefield calcium imaging data is binned")

    if new_vid.shape[0]>len(wf_frame_timestamps):
        new_vid = new_vid[:len(wf_frame_timestamps), :, :]
    elif new_vid.shape[0] < len(wf_frame_timestamps):
        raise ValueError(f"Video has less frames than timestamps: Video frames = {new_vid.shape[0]}, timestamps = {len(wf_frame_timestamps)}")
        return
    else:
        print(" ")
        print("Number of video frames and timestamps match")

    print(" ")
    print("Saving widefield calcium imaging data")
    with h5py.File(output_folder + r'\F_data.h5', 'w') as f:
        wf_dataset = f.create_dataset('F', data=new_vid)

    end_time = time.time()
    print(f"F file created with shape {new_vid.shape}")
    print("Preprocess took %0.4f min" % ((end_time - start_time) / 60))

    return output_folder + r'\F_data.h5'


def concat_widefield_data(file, wf_frame_timestamps, output_folder):
    file = file[0]
    tstamps = iio.v3.improps(file, plugin='pyav', format='gray16be')

    if tstamps.shape[0] < len(wf_frame_timestamps):
        print(" ")
        print("There seems to be less WF frames than GUI pulses")
        wf_frame_timestamps = wf_frame_timestamps[:tstamps.shape[0]]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Get widefield size to know if we process in memory or of memory
    file_size = os.path.getsize(file)
    print(f"WF mj2 file is {np.round(file_size/1000000000, 2)} Gb")
    if file_size/1000000000 < 70:
        print(f"Process data in memory ... ")
        F_file = concat_inmemory_and_save(file, wf_frame_timestamps, output_folder)
    else:
        print(f"Process data off memory ... ")
        F_file = concat_and_save(file, wf_frame_timestamps, output_folder)

    return F_file


def prompt_overwrite(folder_path, file):
    if os.path.exists(folder_path) and os.path.isfile(folder_path + '\\' + file):
        overwrite = input(f"An {file} file already exists. Do you want to overwrite its contents? [y], n \n")
        if not overwrite or overwrite == 'y':
            return True
        elif overwrite == 'n':
            return False
    else:
        return True


def make_alignment_reference(mouse_id, align_to='bregma', overwrite_session=None):
    sessions_to_skip = ['PB175_20240308_140045', 'PB185_20240824_121743', 'PB187_20240823_131743', 'PB195_20241107_112405', 'PB197_20241128_161907']
    analysis_folder = get_subject_analysis_folder(mouse_id)
    ref_file = os.path.join(analysis_folder, 'alignment_ref.csv')
    sessions = os.listdir(analysis_folder)
    for session_id in sessions:
        if overwrite_session is not None and session_id != overwrite_session:
            continue

        if not os.path.isdir(os.path.join(analysis_folder, session_id)):
            continue

        if session_id in sessions_to_skip:
            continue
        if session_id == 'RD052_20240605_143534':
            wf_file = ['//sv-nas1.rcp.epfl.ch/Petersen-Lab/data/RD052/Recording/Imaging/RD052_20240605_152408/RD052_20240605_152408.mj2']
        else:
            wf_file = get_widefield_file(os.path.join(analysis_folder, session_id, f"config_{session_id}.yaml"))
        if wf_file is None:
            continue
        if os.path.exists(ref_file):
            reference_list = pd.read_csv(ref_file)
            if session_id in reference_list['session_id'].values:
                continue

            new_line = pd.Series(index=['session_id', 'bregma_x', 'bregma_y', 'c2_x', 'c2_y'])

            coord_x, coord_y = reference_list.loc[reference_list.index[-1], [f'{align_to}_x', f'{align_to}_y']]
            image = iio.v3.imread(wf_file[0], plugin='pyav', format='gray16be', index=0)
            fig, ax = plt.subplots()
            fig.suptitle(f"{session_id}")
            ax.imshow(image)
            ax.scatter(coord_x, coord_y, marker='+', color='red')
            plt.show()

            answer = input(f"Do you want to use the previous coordinates (x: {coord_x}, y: {coord_y})? [n], y")
            while answer != 'y':
                coords = input('Enter space separated ref coords: x y')
                coord_x, coord_y = tuple(int(item) for item in coords.split())
                fig, ax = plt.subplots()
                fig.suptitle(f"{session_id}")
                ax.imshow(image)
                ax.scatter(coord_x, coord_y, marker='+', color='red')
                plt.show()
                answer = input("Do you want to use these coordinates? [n], y")

            fig.savefig(os.path.join(analysis_folder, session_id, f'{align_to}_reference.png'))

        else:
            reference_list = pd.DataFrame(columns=['session_id', 'bregma_x', 'bregma_y', 'c2_x', 'c2_y'])
            new_line = pd.Series(index=['session_id', 'bregma_x', 'bregma_y', 'c2_x', 'c2_y'])

            image = iio.v3.imread(wf_file[0], plugin='pyav', format='gray16be', index=0)
            fig, ax = plt.subplots()
            fig.suptitle(f"{session_id}")
            ax.imshow(image)
            plt.show()

            answer='n'
            while answer != 'y':
                coords = input('Enter space separated ref coords: x y')
                coord_x, coord_y = tuple(int(item) for item in coords.split())
                fig, ax = plt.subplots()
                fig.suptitle(f"{session_id}")
                ax.imshow(image)
                ax.scatter(coord_x, coord_y, marker='+', color='red')
                plt.show()
                answer = input("Do you want to use these coordinates? [n], y")

            fig.savefig(os.path.join(analysis_folder, session_id, f'{align_to}_reference.png'))

        new_line["session_id"] = session_id
        new_line[f"{align_to}_x"] = coord_x
        new_line[f"{align_to}_y"] = coord_y
        reference_list =pd.concat([reference_list, new_line.to_frame().T], ignore_index=True)
        reference_list.to_csv(ref_file, index=False)


def get_alignment_reference(session_id, align_to='bregma'):
    analysis_folder = get_subject_analysis_folder(session_id.split("_")[0])
    file = os.path.join(analysis_folder, 'alignment_ref.csv')
    reference_list = pd.read_csv(file)

    if session_id not in reference_list['session_id']:
        ValueError(f"{session_id} has no data in the bregma reference, run make_alignment_reference")

    x = reference_list.loc[reference_list['session_id'] == session_id, f"{align_to}_x"].values[0]
    y = reference_list.loc[reference_list['session_id'] == session_id, f"{align_to}_y"].values[0]

    return x, y


def align_videos(vid, start, dest):

    orig_x, orig_y = start[0], start[1]
    dest_x, dest_y = dest[0], dest[1]
    trans_x, trans_y = dest_x - orig_x, dest_y - orig_y
    margin_x = vid.shape[2] - np.abs(trans_x)
    margin_y = vid.shape[1] - np.abs(trans_y)
    if trans_y >= 0 and trans_x >= 0:
        vid[:, trans_y:, trans_x:] = vid[:, :margin_y, :margin_x]
        vid[:, :trans_y, :] *= 0
        vid[:, :, :trans_x] *= 0
    elif trans_y < 0 and trans_x >= 0:
        vid[:, :trans_y, trans_x:] = vid[:, -margin_y:, :margin_x]
        vid[:, trans_y:, :] *= 0
        vid[:, :, :trans_x] *= 0
    elif trans_y >= 0 and trans_x < 0:
        vid[:, trans_y:, :trans_x] = vid[:, :margin_y, -margin_x:]
        vid[:, :trans_y, :] *= 0
        vid[:, :, trans_x:] *= 0
    else:
        vid[:, :trans_y, :trans_x] = vid[:, -margin_y:, -margin_x:]
        vid[:, trans_y:, :] *= 0
        vid[:, :, trans_x:] *= 0

    return vid

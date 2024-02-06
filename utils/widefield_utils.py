import os
import h5py
import time
import scipy
import tqdm
import numpy as np
import dask.array as da
import imageio as iio
import gc
gc.collect()



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
    F_file = h5py.File(os.path.join(data_folder, 'F_data.h5'), 'r')
    F = F_file['F'][:]

    print(" ")
    print("Computing F0...")

    if method == 'low_pass_filter':
        F0 = compute_F0_CardinLAB(F)
    elif method == 'percentile':
        F0 = compute_F0_early_percentile(F, winsize=F.shape[0])

    dff0 = (F - F0) / F0

    end = time.time()
    print("dFF0 calculation and saving took %0.4f min" % ((end - start) / 60))

    F_file.close()

    return dff0


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
        F = da.from_array(f['F'])
    f.close()

    return output_folder + r'\F_data.h5', F


def concat_inmemory_and_save(file, wf_frame_timestamps, output_folder):
    start = time.time()
    props = iio.v3.improps(file, plugin='pyav', format='gray16be')

    print(" ")
    print("Loading widefield calcium imaging data")

    vid = iio.v3.imread(file, plugin='pyav', format='gray16be')
    vid = vid.reshape(-1, int(vid.shape[1] / 2), 2, int(vid.shape[2] / 2), 2).mean(axis=2).mean(axis=3)

    if vid.shape[0]>len(wf_frame_timestamps):
        vid = vid[:len(wf_frame_timestamps), :, :]
    elif vid.shape[0] < len(wf_frame_timestamps):
        raise ValueError(f"Video has less frames than timestamps: Video frames = {vid.shape[0]}, timestamps = {len(wf_frame_timestamps)}")
        return
    else:
        print(" ")
        print("Number of video frames and timestamps match")

    print(" ")
    print("Saving widefield calcium imaging data")
    #
    with h5py.File(output_folder + r'\F_data.h5', 'w') as f:
        wf_dataset = f.create_dataset('F', data=vid)

        # F = da.from_array(vid).rechunk(10000, -1, -1)
        # da.store(F, wf_dataset)

    end = time.time()
    print(f"F file created with shape {vid.shape}")
    print("Preprocess took %0.4f min" % ((end - start) / 60))

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

    F_file = concat_inmemory_and_save(file, wf_frame_timestamps, output_folder)


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


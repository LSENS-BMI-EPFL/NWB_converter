import os
import yaml
from tqdm import tqdm
import numpy as np
from pynwb.ophys import OpticalChannel, Device, OnePhotonSeries
from mat73 import loadmat
from utils.server_paths import get_widefield_file, get_subject_analysis_folder, get_subject_data_folder
from npy_append_array import NpyAppendArray



def concat_widefield_data(files, timestamps, output_folder, overwrite=False):

    if not overwrite:
        return

    filenames = files[0].split('\\')[-1][:-9]
    save_path = output_folder + '\\' + filenames + '.npy'
    with NpyAppendArray(save_path, delete_if_exists=True) as npaa:
        for i, file in tqdm(enumerate(files)):
            data = loadmat(file)['data']
            data =data.reshape(
                int(data.shape[0] / 2), 2, int(data.shape[1] / 2), 2, -1).mean(axis=3).mean(axis=1)
            npaa.append(data)
    npaa.close()

    save_path = output_folder + '\\' + filenames + '_timestamps.npy'
    with NpyAppendArray(save_path, delete_if_exists=True) as npaa:
        for file in timestamps:
            data = loadmat(file)['timestamps']
            npaa.append(data)
    npaa.close()

def convert_widefield_recording(nwb_file, config_file, wf_frame_timestamps):
    """
    Function to process widefield data
    Args:
        nwb_file:
        config_file:

    Returns:

    """

    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    wf_metadata = config["widefield_metadata"]
    subject_id = config["subject_metadata"]["subject_id"]
    session_name = config["session_metadata"]["session_id"]
    data_folder = get_subject_data_folder(subject_id)
    mat_path = os.path.join(data_folder, "Recording", "Imaging", session_name)

    analysis_folder = get_subject_analysis_folder(subject_id=subject_id)
    file_names, timestamps = get_widefield_file(mat_path=mat_path)

    if f"{session_name}.npy" in os.listdir(os.path.join(analysis_folder, session_name)):
        overwrite = input("A .npy file with this name already exists. Do you want to overwrite it? [y], n \n")
        if not overwrite or overwrite == 'y':
            overwrite = True
        elif overwrite == 'n':
            overwrite = False
    else:
        overwrite = True

    concat_widefield_data(file_names, timestamps, output_folder=os.path.join(analysis_folder, session_name), overwrite=overwrite)

    data_sample = np.load(os.path.join(analysis_folder, session_name, f"{session_name}.npy"), mmap_mode="r")

    device = Device("HamamatsuOrcaFlash4.0v3")

    optical_channel = OpticalChannel("optical_channel", "Green_channel", 488.0)
    indicator = "GFP"
    image_plane_location = "hemisphere"

    imaging_plane = nwb_file.create_imaging_plane(name="widefield_imaging_plane",
                                                  optical_channel=optical_channel,
                                                  description="Recording of the whole dorsal cortex in the left hemisphere",
                                                  device=device,
                                                  excitation_lambda=488.0,
                                                  imaging_rate=float(wf_metadata["CameraFrameRate"]),
                                                  indicator=indicator,
                                                  location=image_plane_location)

    raw_wf_series = OnePhotonSeries(
        name="raw_widefield_imaging_488",
        binning=64,
        dimension=data_sample.shape,
        external_file=[os.path.join(analysis_folder, session_name, f"{session_name}.npy")],
        imaging_plane=imaging_plane,
        starting_frame=[0],
        format="external",
        exposure_time=1/wf_metadata["CameraFrameRate"] if "Synchronous" in wf_metadata["CameraTriggerMode"] else wf_metadata["CameraExposure"],
        timestamps=[timestamp[0] for timestamp in wf_frame_timestamps],
        unit="grayscale values"
    )

    dFF0_wf_series = OnePhotonSeries(
        name="dFF0_widefield_imaging_488",
        binning=64,
        dimension=data_sample.shape,
        external_file=[os.path.join(analysis_folder, session_name, f"{session_name}_dFF0.npy")],
        imaging_plane=imaging_plane,
        starting_frame=[0],
        format="external",
        exposure_time=1/wf_metadata["CameraFrameRate"] if "Synchronous" in wf_metadata["CameraTriggerMode"] else wf_metadata["CameraExposure"],
        timestamps=[timestamp[0] for timestamp in wf_frame_timestamps],
        unit="normalized amplitude"
    )

    nwb_file.add_acquisition(raw_wf_series)
    nwb_file.add_acquisition(dFF0_wf_series)

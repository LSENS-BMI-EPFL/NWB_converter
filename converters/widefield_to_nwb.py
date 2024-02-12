import gc
gc.collect()
import sys
import yaml
import importlib
import subprocess
from utils.widefield_utils import *
from pynwb.ophys import OpticalChannel, Device, OnePhotonSeries
from utils.server_paths import get_widefield_file, get_subject_analysis_folder, get_subject_data_folder


def convert_widefield_recording(nwb_file, config_file, wf_frame_timestamps):
    """
    Function to process widefield data
    Args:
        nwb_file:
        config_file:

    Returns:

    """

    if not importlib.util.find_spec('av'): # Check to have pyav plugin installed and otherwise install it
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'av'])

    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    wf_metadata = config["widefield_metadata"]
    subject_id = config["subject_metadata"]["subject_id"]
    session_name = config["session_metadata"]["session_id"]
    data_folder = get_subject_data_folder(subject_id)

    analysis_folder = get_subject_analysis_folder(subject_id=subject_id)
    file_names = get_widefield_file(config_file=config_file)
    if not file_names:
        return
    frames, fps = read_motion_jpeg_2000_movie(file_names[0])

    if len(file_names)>1:
        file_names = [file for file in file_names if session_name in file.split("\\")[-1]][0]

    analysis_folder = fr'M:\analysis\Pol_Bech\data\{subject_id}'
    F_file = os.path.join(analysis_folder, session_name, 'F_data.h5')

    if not os.path.exists(F_file):
        concat_widefield_data(file_names,
                               wf_frame_timestamps,
                               output_folder=os.path.join(analysis_folder, session_name))
    else:
        with h5py.File(F_file) as f:
            print(" ")
            print(f"Found F_data file with shape {f['F'].shape}")

    dff0_data = compute_dff0(data_folder=os.path.join(analysis_folder, session_name), method='percentile')

    print(" ")
    print(f"dFF0 calculated, final shape = {dff0_data.shape}")

    # Create the 'ophys module'
    if 'ophys' in nwb_file.processing:
        ophys_module = nwb_file.processing['ophys']
    else:
        ophys_module = nwb_file.create_processing_module('ophys', 'contains optical physiology processed data')

    device = Device("HamamatsuOrcaFlash4.0v3")
    nwb_file.add_device(device)
    excitation_wv = 567.0 if wf_metadata['LED567'] else 488.0
    optical_channel = OpticalChannel("optical_channel", "Green_channel", excitation_wv)
    indicator = "jRGECO1a" if wf_metadata['LED567'] else "GFP"
    image_plane_location = "hemisphere"

    imaging_plane = nwb_file.create_imaging_plane(name="widefield_imaging_plane",
                                                  optical_channel=optical_channel,
                                                  description="Recording of the whole dorsal cortex in the left hemisphere",
                                                  device=device,
                                                  excitation_lambda=excitation_wv,
                                                  imaging_rate=float(wf_metadata["CameraFrameRate"]),
                                                  indicator=indicator,
                                                  location=image_plane_location)

    F_dset = h5py.File(F_file, 'r')
    F_data = da.from_array(F_dset['F'])

    F_wf_series = OnePhotonSeries(
        name="F",
        binning=64,
        dimension=F_data.shape,
        external_file=[os.path.join(analysis_folder, session_name, "F_data", 'F_data.h5')],
        imaging_plane=imaging_plane,
        starting_frame=[0],
        format="external",
        exposure_time=1 / wf_metadata["CameraFrameRate"] if "Synchronous" in wf_metadata["CameraTriggerMode"] else
        wf_metadata["CameraExposure"],
        timestamps=[timestamp[0] for timestamp in wf_frame_timestamps],
        unit="grayscale values"
    )

    dFF0_wf_series = OnePhotonSeries(
        name="dff0",
        binning=64,
        dimension=dff0_data.shape,
        data=dff0_data,
        imaging_plane=imaging_plane,
        starting_frame=[0],
        exposure_time=1 / wf_metadata["CameraFrameRate"] if "Synchronous" in wf_metadata["CameraTriggerMode"] else
        wf_metadata["CameraExposure"],
        timestamps=[timestamp[0] for timestamp in wf_frame_timestamps],
        unit="normalized amplitude"
    )

    F_dset.close()

    nwb_file.add_acquisition(F_wf_series)
    ophys_module.add(dFF0_wf_series)

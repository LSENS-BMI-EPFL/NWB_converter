import numpy as np
import yaml
from pynwb.ophys import Device, OpticalChannel, TwoPhotonSeries

from utils.server_paths import get_imaging_file
from utils.tiff_loading import get_tiff_movie_shape, load_tiff_movie_in_memory


def convert_ci_movie(nwb_file, config_file, movie_format, ci_frame_timestamps):

    """

    :param nwb_file: nwb file
    :param config_file: Main yaml config file including 2P imaging metadata
    :param movie_format: either 'tiff' or 'link' if link add the path to the NWB file if tiff add the data
    :param: ci_frame_timestamps: timestamps of each imaging frame
    :return: create ImagingPlane, add acquisition


    """

    motion_corrected_file_name = get_imaging_file(config_file)
    if motion_corrected_file_name is None:
        print(f"No calcium imaging movie to add, return")
        return

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    two_p_metadata = config['two_photon_metadata']

    device = Device(two_p_metadata['device'])
    nwb_file.add_device(device)
    optical_channel = OpticalChannel('optical_channel', 'GreenPMT', two_p_metadata['emission_lambda'])
    excitation_lambda = two_p_metadata['excitation_lambda']
    indicator = two_p_metadata['indicator']
    image_plane_location = two_p_metadata['image_plane_location']

    scanimage_dict = config['log_continuous_metadata']['scanimage_dict']
    ci_sampling_rate = float(scanimage_dict['theoretical_ci_sampling_rate'])

    imaging_plane = nwb_file.create_imaging_plane(name='my_imaging_plane',
                                                  optical_channel=optical_channel,
                                                  description='A very interesting part of the brain',
                                                  device=device,
                                                  excitation_lambda=excitation_lambda,
                                                  imaging_rate=float(ci_sampling_rate),
                                                  indicator=indicator,
                                                  location=image_plane_location)

    if movie_format != 'link':
        if len(motion_corrected_file_name) == 1:
            print("Load data from single tiff")
            file_name = motion_corrected_file_name[0]
            tiff_movie = load_tiff_movie_in_memory(file_name, frames_to_add=None)
        else:
            print(f"Load data from multi-tiff, (found {len(motion_corrected_file_name)} tiff files)")
            first_file = motion_corrected_file_name[0]
            tiff_movie = load_tiff_movie_in_memory(first_file, frames_to_add=None)
            for file_index, file_name in enumerate(motion_corrected_file_name[1:]):
                tif_file = load_tiff_movie_in_memory(file_name, frames_to_add=None)
                tiff_movie = np.concatenate(tiff_movie, tif_file, axis=0)

        n_frames, n_lines, n_cols = tiff_movie.shape
        print(f"Movie dimensions n_frames, n_lines, n_cols :{n_frames, n_lines, n_cols}")
        motion_corrected_img_series = TwoPhotonSeries(name='motion_corrected_ci_movie',
                                                      dimension=[n_frames, n_lines, n_cols],
                                                      data=tiff_movie,
                                                      imaging_plane=imaging_plane,
                                                      starting_frame=[0], format=movie_format,
                                                      timestamps=ci_frame_timestamps,
                                                      unit="lux")
    elif movie_format == 'link':
        print(f"Extract tiff movie shape:")
        n_frames, n_lines, n_cols, starting_frames = get_tiff_movie_shape(motion_corrected_file_name)
        print(f"Movie dimensions n_frames, n_lines, n_cols :{n_frames, n_lines, n_cols}")

        motion_corrected_img_series = TwoPhotonSeries(name='motion_corrected_ci_movie',
                                                      dimension=[n_frames, n_lines, n_cols],
                                                      external_file=motion_corrected_file_name,
                                                      imaging_plane=imaging_plane,
                                                      starting_frame=starting_frames, format='external',
                                                      timestamps=ci_frame_timestamps,
                                                      unit="lux")

    nwb_file.add_acquisition(motion_corrected_img_series)



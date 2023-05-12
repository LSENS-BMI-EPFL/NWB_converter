from pynwb.base import TimeSeries
from pynwb.device import Device
from pynwb.ophys import TwoPhotonSeries, OpticalChannel, CorrectedImageStack
from utils.tiff_loading import load_tiff_movie_in_memory, get_tiff_movie_shape
from PIL import ImageSequence
import PIL
import numpy as np
import os
import yaml


def convert_ci_movie(nwb_file, two_p_yaml_file, log_yaml_file, movie_format):
    """

    :param nwb_file: nwb file
    :param two_p_yaml_file: 2p config
    :param log_yaml_file: sampling rate for 2p imaging (deducted from continuous logging)
    :param movie_format: either 'tiff' or 'link' if link add the path to the NWB file if tiff add the data
    :return: create ImagingPlane, add acquisition


    """

    with open(two_p_yaml_file, 'r') as stream:
        two_p_data_yaml_file = yaml.safe_load(stream)

    device = Device(two_p_data_yaml_file.get('device'))
    nwb_file.add_device(device)
    optical_channel = OpticalChannel('optical_channel', 'GreenPMT', two_p_data_yaml_file.get('emission_lambda'))
    excitation_lambda = two_p_data_yaml_file.get('excitation_lambda')
    indicator = two_p_data_yaml_file.get('indicator')
    image_plane_location = two_p_data_yaml_file.get('image_plane_location')

    with open(log_yaml_file, 'r') as stream:
        log_yaml_file_data = yaml.safe_load(stream)

    scanimage_dict = log_yaml_file_data.get("scan_image_dict")
    ci_sampling_rate = float(scanimage_dict.get("theoretical_ci_sampling_rate"))

    imaging_plane = nwb_file.create_imaging_plane(name='my_imaging_plane',
                                                  optical_channel=optical_channel,
                                                  description='A very interesting part of the brain',
                                                  device=device,
                                                  excitation_lambda=excitation_lambda,
                                                  imaging_rate=float(ci_sampling_rate),
                                                  indicator=indicator,
                                                  location=image_plane_location)

    motion_corrected_file_name = log_yaml_file_data.get('ci_tiff_path')

    if movie_format != 'link':
        tiff_movie = load_tiff_movie_in_memory(motion_corrected_file_name,
                                               frames_to_add=None)

        n_frames, n_lines, n_cols = tiff_movie.shape
        print(f"Movie dimensions n_frames, n_lines, n_cols :{n_frames, n_lines, n_cols}")
        motion_corrected_img_series = TwoPhotonSeries(name='motion_corrected_ci_movie',
                                                      dimension=[n_frames, n_lines, n_cols],
                                                      data=tiff_movie,
                                                      imaging_plane=imaging_plane,
                                                      starting_frame=[0], format=movie_format,
                                                      rate=float(ci_sampling_rate),
                                                      unit="lux")
    elif movie_format == 'link':
        n_frames, n_lines, n_cols = get_tiff_movie_shape(motion_corrected_file_name)

        print(f"Movie dimensions n_frames, n_lines, n_cols :{n_frames, n_lines, n_cols}")
        motion_corrected_img_series = TwoPhotonSeries(name='motion_corrected_ci_movie',
                                                      dimension=[n_frames, n_lines, n_cols],
                                                      external_file=[motion_corrected_file_name],
                                                      imaging_plane=imaging_plane,
                                                      starting_frame=[0], format='external',
                                                      rate=float(ci_sampling_rate),
                                                      unit="lux")

    nwb_file.add_acquisition(motion_corrected_img_series)



import yaml
from pynwb.ophys import TwoPhotonSeries, OpticalChannel, Device
from utils.tiff_loading import load_tiff_movie_in_memory, get_tiff_movie_shape
from utils.server_paths import get_imaging_file


def convert_ci_movie(nwb_file, config_file, movie_format, ci_frame_timestamps):
    """

    :param nwb_file: nwb file
    :param config_file: Main yaml config file including 2P imaging metadata
    :param movie_format: either 'tiff' or 'link' if link add the path to the NWB file if tiff add the data
    :return: create ImagingPlane, add acquisition


    """

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    two_p_metadata = config['2P_metadata']

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

    motion_corrected_file_name = get_imaging_file(config_file)
    if motion_corrected_file_name is None:
        print(f"No calcium imaging movie to add")
        return

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
                                                      timestamps=ci_frame_timestamps,
                                                      unit="lux")
    elif movie_format == 'link':
        n_frames, n_lines, n_cols = get_tiff_movie_shape(motion_corrected_file_name)

        print(f"Movie dimensions n_frames, n_lines, n_cols :{n_frames, n_lines, n_cols}")
        motion_corrected_img_series = TwoPhotonSeries(name='motion_corrected_ci_movie',
                                                      dimension=[n_frames, n_lines, n_cols],
                                                      external_file=[motion_corrected_file_name],
                                                      imaging_plane=imaging_plane,
                                                      starting_frame=[0], format='external',
                                                      timestamps=ci_frame_timestamps,
                                                      unit="lux")

    nwb_file.add_acquisition(motion_corrected_img_series)



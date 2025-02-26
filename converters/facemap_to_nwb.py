import yaml
import numpy as np

from pynwb.base import TimeSeries
from pynwb.behavior import BehavioralTimeSeries
from pynwb.base import Images
from pynwb.image import GrayscaleImage

from utils.server_paths import get_facemap_file_path


def convert_facemap_data(nwb_file, config_file, video_timestamps):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    facemap_file = get_facemap_file_path(config_file)

    facemap_data = np.load(facemap_file, allow_pickle=True).item()

    motion_energy = facemap_data['motion'][1]

    motSVD = facemap_data['motSVD'][1]

    SVD_images = facemap_data['motMask_reshape'][1]

    print(f"Motion energy shape: {motion_energy.shape}")
    print(f"Motion SVD shape: {motSVD.shape}")
    print(f"SVD images: {SVD_images.shape}")

    if 'behavior' in nwb_file.processing:
        bhv_module = nwb_file.processing['behavior']
    else:
        print("Creating behaviour processing module")
        bhv_module = nwb_file.create_processing_module('behavior', 'contains behavioral processed data')

    # Get the behavioral timeseries module (that we will use for continuous data)
    try:
        behavior_t_series = bhv_module.get(name='BehavioralTimeSeries')
    except KeyError:
        behavior_t_series = BehavioralTimeSeries(name='BehavioralTimeSeries')
        bhv_module.add_data_interface(behavior_t_series)

    # Add Motion energy to NWB
    timeseries = TimeSeries(name='motion_energy',
                            data=motSVD,
                            unit='seconds',
                            timestamps=[timestamp[0] for timestamp in video_timestamps['cam1']],
                            starting_time=None,
                            rate=None,
                            comments='no comments',
                            description=f'no description',
                            control=None,
                            control_description=None,
                            continuity='continuous')

    behavior_t_series.add_timeseries(timeseries)

    # Add Motion SVD  to NWB
    timeseries = TimeSeries(name='motion_svd',
                            data=motion_energy,
                            unit='seconds',
                            timestamps=[timestamp[0] for timestamp in video_timestamps['cam1']],
                            starting_time=None,
                            rate=None,
                            comments='no comments',
                            description=f'no description',
                            control=None,
                            control_description=None,
                            continuity='continuous')

    behavior_t_series.add_timeseries(timeseries)

    # Add SVD components to NWB
    svd_gs_images = []
    for svd_im in range(SVD_images.shape[2]):
        gs_svd_im = GrayscaleImage(
            name=f"SVD_im_{svd_im}",
            data=SVD_images[:, :, svd_im],
            description=f"Grayscale version of the SVd {svd_im} image.")
        svd_gs_images.append(gs_svd_im)

    images = Images(
        name="facemap_svd_images",
        images=svd_gs_images,
        description="A collection SVD images of the mouse.",
    )

    nwb_file.add_acquisition(images)

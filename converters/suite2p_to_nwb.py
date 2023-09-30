from pynwb.ophys import ImageSegmentation, Fluorescence
import os
import numpy as np
from PIL import ImageSequence
import PIL
from utils.server_paths import get_suite2p_folder


def convert_suite2p_data(nwb_file, config_file, ci_frame_timestamps):
    """
    :param nwb_file: nwb file
    :param config_file : general config file allowing to reconstruct path to suite2p folder
    :param ci_frame_timestamps: timestamps of each imaging frame obtained from analyse of continuous log
    :return:
    """

    # First check whether there is data to add in the ophys module
    suite2p_folder = get_suite2p_folder(config_file)
    if suite2p_folder is None:
        print(f"No suite2p folder to add")
        return

    if 'ophys' in nwb_file.processing:
        module = nwb_file.processing['ophys']
    else:
        module = nwb_file.create_processing_module('ophys', 'contains optical physiology processed data')

    image_series = nwb_file.acquisition.get("motion_corrected_ci_movie")
    if image_series is None:
        print(f"No calcium imaging movie named 'motion_corrected_ci_movie' found in nwb_file")

    img_seg = ImageSegmentation(name="all_cells")
    module.add_data_interface(img_seg)
    imaging_plane = nwb_file.get_imaging_plane("my_imaging_plane")
    ci_sampling_rate = imaging_plane.imaging_rate

    ps = img_seg.create_plane_segmentation(description='output from segmenting',
                                           imaging_plane=imaging_plane, name='my_plane_segmentation',
                                           reference_images=image_series)

    # Load Suite2p data
    suite2p_raw = None
    suite2p_deconvolve = None

    suite2p_folder = os.path.join(suite2p_folder, "plane0")
    if not os.path.isfile(os.path.join(suite2p_folder, "stat.npy")):
        print(f"Stat file missing in {suite2p_folder}")
        return
    else:
        stat = np.load(os.path.join(suite2p_folder, "stat.npy"), allow_pickle=True)
    if os.path.isfile(os.path.join(suite2p_folder, "iscell.npy")):
        is_cell = np.load(os.path.join(suite2p_folder, "iscell.npy"), allow_pickle=True)
    else:
        is_cell = None
    if os.path.isfile(os.path.join(suite2p_folder, "spks.npy")):
        suite2p_deconvolve = np.load(os.path.join(suite2p_folder, "spks.npy"), allow_pickle=True)
    if os.path.isfile(os.path.join(suite2p_folder, "F.npy")):
        suite2p_raw = np.load(os.path.join(suite2p_folder, "F.npy"), allow_pickle=True)

    # Check image dimension
    if image_series is not None:
        if image_series.format == "tiff":
            dim_y, dim_x = image_series.data.shape[1:]
            n_frames = image_series.data.shape[0]
            print(f"Dimensions double check: n_lines, n_cols: {image_series.data.shape[1:]}")
        elif image_series.format == "external":
            im = PIL.Image.open(image_series.external_file[0])
            n_frames = len(list(ImageSequence.Iterator(im)))
            dim_y, dim_x = np.array(im).shape
            print(f"Dimensions double check: n_lines, n_cols: {np.array(im).shape}")
        else:
            raise Exception(f"Format of calcium movie imaging {image_series.format} not yet implemented")

    # Add suite2p pixel mask
    pixel_masks = []
    n_cells = 0
    for cell in np.arange(len(stat)):
        if is_cell is not None:
            if is_cell[cell][0] == 0:
                continue
        n_cells += 1
        pix_mask = [(y, x, 1) for x, y in zip(stat[cell]["xpix"], stat[cell]["ypix"])]
        image_mask = np.zeros((dim_y, dim_x))  # recently inverted (12th of may 2020)
        for pix in pix_mask:
            image_mask[int(pix[0]), int(pix[1])] = pix[2]  # pix[0] pix[1] recently inverted (12th of may 2020)
        # we can id to identify the cell (int) otherwise it will be incremented at each step
        pixel_masks.append(pix_mask)
        ps.add_roi(pixel_mask=pix_mask, image_mask=image_mask)

    fl = Fluorescence(name=f"fluorescence_all_cells")
    module.add_data_interface(fl)
    rt_region = ps.create_roi_table_region('all cells', region=list(np.arange(n_cells)))
    if ci_frame_timestamps is None:
        timestamps = np.arange(n_frames)
    else:
        timestamps = ci_frame_timestamps

    # Add Suite2p_raw traces
    if suite2p_raw is not None:
        suite2p_raw_filtered = np.zeros((n_cells, suite2p_raw.shape[1]))
        real_cell_index = 0
        for cell in np.arange(suite2p_raw.shape[0]):
            if is_cell is not None:
                if is_cell[cell][0] == 0:
                    continue
            suite2p_raw_filtered[real_cell_index] = suite2p_raw[cell]
            real_cell_index += 1

        rrs = fl.create_roi_response_series(name='suite2p_raw',
                                            data=np.transpose(suite2p_raw_filtered),
                                            unit='lumens',
                                            rois=rt_region, timestamps=timestamps,
                                            description="raw traces from suite2p")

        print(f"- Creating Roi Response Series with: suite2p raw traces of "
              f"shape: {(np.transpose(suite2p_raw_filtered)).shape}")

    # Add Suite2p_deconvolved traces
    if suite2p_deconvolve is not None:
        suite2p_deconvolved_filtered = np.zeros((n_cells, suite2p_deconvolve.shape[1]))
        real_cell_index = 0
        for cell in np.arange(suite2p_deconvolve.shape[0]):
            if is_cell is not None:
                if is_cell[cell][0] == 0:
                    continue
            suite2p_deconvolved_filtered[real_cell_index] = suite2p_deconvolve[cell]
            real_cell_index += 1

        rrs = fl.create_roi_response_series(name='suite2p_deconvolve',
                                            data=np.transpose(suite2p_deconvolved_filtered),
                                            unit='lumens',
                                            rois=rt_region, timestamps=timestamps,
                                            description="raw traces from suite2p")

        print(f"- Creating Roi Response Series with: suite2p deconvolved traces of "
              f"shape: {(np.transpose(suite2p_deconvolved_filtered)).shape}")


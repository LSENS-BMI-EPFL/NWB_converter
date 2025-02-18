import gc
gc.collect()
import sys
import yaml
import importlib
import subprocess
import dask.array as da
from utils.widefield_utils import *
from pynwb.ophys import OpticalChannel, Device, OnePhotonSeries, ImageSegmentation, Fluorescence
import utils.server_paths as server_paths
import utils.ci_processing as ci_processing


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

    analysis_folder = server_paths.get_subject_analysis_folder(subject_id=subject_id)
    file_names = get_widefield_file(config_file=config_file)
    if not file_names:
        return
    frames, fps = read_motion_jpeg_2000_movie(file_names[0])

    if len(file_names) > 1:
        file_names = [file for file in file_names if session_name in os.path.basename(file)]
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
        external_file=[os.path.join(analysis_folder, session_name, 'F_data.h5')],
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

    # ADD the segmentation from multiple brain areas
    # Get file containing segmentation of brain regions
    roi_file = server_paths.get_wf_fiji_rois_file(config_file)

    if roi_file is not None:
        print(f" ")
        print(f"Found ROIs file : Add fluorescence traces from ROIs in roi file ")
        img_shape = dff0_data.shape[1:]

        # Extract list of region mask
        print(f"Extract pixel masks from roi file")
        area_names, brain_region_pixel_masks, brain_region_image_masks = \
            ci_processing.get_wf_roi_pixel_mask(roi_file, img_shape)

        # Create an ImageSegmentation & PlaneSegmentation
        img_seg = ImageSegmentation(name="brain_areas")
        ophys_module.add_data_interface(img_seg)

        ps = img_seg.create_plane_segmentation(description='brain area segmentation',
                                               imaging_plane=imaging_plane, name='brain_area_segmentation',
                                               reference_images=dFF0_wf_series if dFF0_wf_series is not None else None)

        # Add rois to plane segmentation
        print(f"Add masks to plane segmentation")
        ci_processing.add_wf_roi(ps, pix_masks=brain_region_pixel_masks, img_masks=brain_region_image_masks)

        # Create Fluorescence object to store fluorescence data
        fl = Fluorescence(name="brain_area_fluorescence")
        ophys_module.add_data_interface(fl)
        n_cells = len(area_names)
        rt_region = ps.create_roi_table_region('brain areas', region=list(np.arange(n_cells)))

        # Compute dff0 traces
        print(f"Compute traces")
        dff0_traces = np.zeros((n_cells, dff0_data.shape[0]))
        for cell in np.arange(n_cells):
            img_mask = ps['image_mask'][cell]
            img_mask = img_mask.astype(bool)
            dff0_traces[cell, :] = np.nanmean(dff0_data[:, img_mask], axis=1)

        # Add fluorescence data to roi response series.
        rrs = fl.create_roi_response_series(name='dff0_traces', data=np.transpose(dff0_traces), unit='lumens',
                                            rois=rt_region,
                                            timestamps=[timestamp[0] for timestamp in wf_frame_timestamps],
                                            description="dff0 traces", control=[cell for cell in range(n_cells)],
                                            control_description=area_names)
        print(f"Creating Roi Response Series with dff0 traces of shape: {(np.transpose(dff0_traces)).shape}")

    # Add grid like ROI
    add_grid_rrs = True
    if add_grid_rrs:
        grid_img_seg = ImageSegmentation(name="grid_areas")
        ophys_module.add_data_interface(grid_img_seg)
        grid_ps = grid_img_seg.create_plane_segmentation(description='brain grid area segmentation',
                                                         imaging_plane=imaging_plane,
                                                         name='brain_grid_area_segmentation',
                                                         reference_images=dFF0_wf_series if dFF0_wf_series is not None else None)
        # Extract list of grid mask
        img_shape = dff0_data.shape[1:]
        grid_coords, brain_grid_pixel_masks, brain_grid_image_masks = ci_processing.get_wf_grid_pixel_mask(img_shape)

        # Add rois to plane segmentation
        print(f"Add masks to plane segmentation")
        ci_processing.add_wf_roi(grid_ps, pix_masks=brain_grid_pixel_masks, img_masks=brain_grid_image_masks)

        # Create Fluorescence object to store fluorescence data
        fl = Fluorescence(name="brain_grid_fluorescence")
        ophys_module.add_data_interface(fl)
        n_cells = len(grid_coords)
        rt_grid = grid_ps.create_roi_table_region('brain grid', region=list(np.arange(n_cells)))

        # Compute dff0 traces
        print(f"Compute traces")
        dff0_grid_traces = np.zeros((n_cells, dff0_data.shape[0]))
        for cell in np.arange(n_cells):
            img_mask = grid_ps['image_mask'][cell]
            dff0_grid_traces[cell, :] = np.nanmean(dff0_data[:, img_mask], axis=1)

        # Add fluorescence data to roi response series.
        rrs = fl.create_roi_response_series(name='dff0_grid_traces', data=np.transpose(dff0_grid_traces), unit='lumens',
                                            rois=rt_grid,
                                            timestamps=[timestamp[0] for timestamp in wf_frame_timestamps],
                                            description="dff0 grid traces", control=[coord for coord in grid_coords],
                                            control_description=grid_coords)
        print(f"Creating Roi Response Series with dff0 grid traces of shape: {(np.transpose(dff0_grid_traces)).shape}")

    gc.collect()


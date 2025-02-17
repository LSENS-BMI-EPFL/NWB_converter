import os
import cv2
import numpy as np
import scipy
from scipy import ndimage
from read_roi import read_roi_file, read_roi_zip


def set_merged_roi_to_non_cell(stat, is_cell):
    # Set merged cells to 0 in is_cell.
    if 'inmerge' in stat[0].keys():
        for i, st in enumerate(stat):
            # 0: no merge; -1: output of a merge; index > 0: merged cell.
            if st['inmerge'] not in [0, -1]:
                is_cell[i] = 0.0

    return is_cell


# def compute_F0(F, fs, window):

#     # Parameters --------------------------------------------------------------
#     nfilt = 30  # Number of taps to use in FIR filter
#     fw_base = 1  # Cut-off frequency for lowpass filter, in Hz
#     base_pctle = 5  # Percentile to take as baseline value

#     # Main --------------------------------------------------------------------
#     # Ensure array_like input is a numpy.ndarray
#     F = np.asarray(F)

#     # For short measurements, we reduce the number of taps
#     nfilt = min(nfilt, max(3, int(F.shape[1] / 3)))

#     if fs <= fw_base:
#         # If our sampling frequency is less than our goal with the smoothing
#         # (sampling at less than 1Hz) we don't need to apply the filter.
#         filtered_f = F
#     else:
#         # The Nyquist rate of the signal is half the sampling frequency
#         nyq_rate = fs / 2.0

#         # Cut-off needs to be relative to the nyquist rate. For sampling
#         # frequencies in the range from our target lowpass filter, to
#         # twice our target (i.e. the 1Hz to 2Hz range) we instead filter
#         # at the Nyquist rate, which is the highest possible frequency to
#         # filter at.
#         cutoff = min(1.0, fw_base / nyq_rate)

#         # Make a set of weights to use with our taps.
#         # We use an FIR filter with a Hamming window.
#         b = scipy.signal.firwin(nfilt, cutoff=cutoff, window='hamming')

#         # The default padlen for filtfilt is 3 * nfilt, but in case our
#         # dataset is small, we need to make sure padlen is not too big
#         padlen = min(3 * nfilt, F.shape[1] - 1)

#         # Use filtfilt to filter with the FIR filter, both forwards and
#         # backwards.
#         filtered_f = scipy.signal.filtfilt(b, [1.0], F, axis=1,
#                                            padlen=padlen)

#     # Take a percentile of the filtered signal and windowed signal
#     F0 = scipy.ndimage.percentile_filter(filtered_f, percentile=base_pctle, size=(1,(fs*2*window + 1)), mode='constant', cval=+np.inf)

#     # Ensure filtering doesn't take us below the minimum value which actually
#     # occurs in the data. This can occur when the amount of data is very low.
#     F0 = np.maximum(F0, np.nanmin(F, axis=1, keepdims=True))

#     return F0


# def compute_dff(F_raw, F_neu, fs, window=60):

#     F_res = F_raw - .7 * F_neu  # Neuropil correction.
#     F0 = compute_F0(Fcor, fs, window)
#     dff = (Fcor - F0) / F0

#     return F0, dff


def add_suite2p_roi(ps, stat, is_cell, dim_x, dim_y):
    pixel_masks = []
    for cell in np.arange(len(stat)):
        if is_cell[cell][0] == 0:
            continue
        pix_mask = [(y, x, 1) for x, y in zip(stat[cell]["xpix"], stat[cell]["ypix"])]
        image_mask = np.zeros((dim_y, dim_x))  # recently inverted (12th of may 2020)
        for pix in pix_mask:
            image_mask[int(pix[0]), int(pix[1])] = pix[2]  # pix[0] pix[1] recently inverted (12th of may 2020)
        # we can id to identify the cell (int) otherwise it will be incremented at each step
        pixel_masks.append(pix_mask)
        ps.add_roi(pixel_mask=pix_mask, image_mask=image_mask)


def get_processed_ci(suite2p_folder):

    suite2p_folder = os.path.join(suite2p_folder, "plane0")
    if not os.path.isfile(os.path.join(suite2p_folder, "stat.npy")):
        print(f"Stat file missing in {suite2p_folder}")
        return
    else:
        stat = np.load(os.path.join(suite2p_folder, "stat.npy"), allow_pickle=True)
        is_cell = np.load(os.path.join(suite2p_folder, "iscell.npy"), allow_pickle=True)
        F_raw = np.load(os.path.join(suite2p_folder, "F_raw.npy"), allow_pickle=True)
        F_neu = np.load(os.path.join(suite2p_folder, "F_neu.npy"), allow_pickle=True)
        F0_cor = np.load(os.path.join(suite2p_folder, "F0_cor.npy"), allow_pickle=True)
        F0_raw = np.load(os.path.join(suite2p_folder, "F0_raw.npy"), allow_pickle=True)
        dff = np.load(os.path.join(suite2p_folder, "dff.npy"), allow_pickle=True)
        # spks = np.load(os.path.join(suite2p_folder, "spks.npy"), allow_pickle=True)
        # if os.path.isfile(os.path.join(suite2p_folder, "separated.npz")):
        #     fissa_output = np.load(os.path.join(suite2p_folder, "separated.npz"), allow_pickle=True)
        # else:
        #     fissa_output = None

    return stat, is_cell, F_raw, F_neu, F0_cor, F0_raw, dff


def get_roi_labels(rois_label_folder):
    far_red_rois = np.load(os.path.join(rois_label_folder, 'FarRedRois.npy'), allow_pickle=True)
    red_rois = np.load(os.path.join(rois_label_folder, 'RedRois.npy'), allow_pickle=True)
    un_rois = np.load(os.path.join(rois_label_folder, 'UNRois.npy'), allow_pickle=True)
    info_file = os.path.join(rois_label_folder, 'CTBInjectionsInfo.txt')

    info = {}
    with open(info_file) as f:
        for line in f:
            key, val = line.split()
            info[key] = val
    info = {color: area for area, color in info.items()}

    return far_red_rois, red_rois, un_rois, info


def get_wf_roi_pixel_mask(roi_file, img_shape):
    roi_file = roi_file[0] if roi_file is not None else None
    dim_y, dim_x = img_shape

    # Variables to return
    pix_masks = []
    image_masks = []
    area_names = []

    if roi_file.endswith("zip"):
        zip_data = read_roi_zip(roi_file)
        for area, roi_data in zip_data.items():
            # Skip bregma or add area name and keep going
            if area == 'bregma':
                continue
            area_names.append(area)

            # Check for ROI drawing type
            if roi_data['type'] == 'oval':
                circle_radius = int(roi_data['width'] / 2)
                circle_center = (int(roi_data['left'] + circle_radius), int(roi_data['top'] + circle_radius))
                black_image = np.zeros((dim_y, dim_x), dtype=np.uint8)
                image = cv2.circle(black_image, circle_center, circle_radius, (255, 255, 255), 1)
                image_mask = ndimage.binary_fill_holes(image).astype(int)
                pix_mask = np.argwhere(image_mask)
                pix_mask = [(pix[0], pix[1], 1) for pix in pix_mask]

                pix_masks.append(pix_mask)
                image_masks.append(image_mask)

            if roi_data['type'] == 'polygon':
                n_points = len(roi_data['x'])
                contours = np.zeros((2, n_points), dtype="int16")
                contours[0] = roi_data['x']
                contours[1] = roi_data['y']
                if len(contours[0]) == 0:
                    print(f'Error: {area} contours length {len(contours[0]), len(contours[1])}')
                    continue
                image_mask = np.zeros((dim_y, dim_x), dtype=np.uint8)
                contours_to_draw = np.array([(contours[1][i], contours[0][i]) for i in range(n_points)]).astype(int)
                image_mask = cv2.drawContours(image_mask, [contours_to_draw], 0, (255, 255, 255), 1)
                # we  use morphology.binary_fill_holes to build pixel_mask from coord
                image_mask = ndimage.binary_fill_holes(image_mask).astype(int)
                pix_mask = np.argwhere(image_mask)
                pix_mask = [(pix[0], pix[1], 1) for pix in pix_mask]

                pix_masks.append(pix_mask)
                image_masks.append(image_mask)

    elif roi_file.endswith("roi"):
        roi = read_roi_file(roi_file)
        area_names.append(os.path.basename(roi_file).split('.')[0])

        if roi['type'] == 'oval':
            circle_radius = int(roi['width'] / 2)
            circle_center = (int(roi['left'] + circle_radius), int(roi['top'] + circle_radius))
            black_image = np.zeros((dim_y, dim_x), dtype=np.uint8)
            image = cv2.circle(black_image, circle_center, circle_radius, (255, 255, 255), 1)
            image_mask = ndimage.binary_fill_holes(image).astype(int)
            pix_mask = np.argwhere(image_mask)
            pix_mask = [(pix[0], pix[1], 1) for pix in pix_mask]

            pix_masks.append(pix_mask)
            image_masks.append(image_mask)

        elif roi['type'] == 'polygon':
            roi = roi[list(roi.keys())[0]]
            n_points = len(roi['x'])
            contours = np.zeros((2, n_points), dtype="int16")
            contours[0] = roi['x']
            contours[1] = roi['y']

            image_mask = np.zeros((dim_y, dim_x), dtype=np.uint8)
            contours_to_draw = np.array([(contours[1][i], contours[0][i]) for i in range(n_points)]).astype(int)
            image_mask = cv2.drawContours(image_mask, [contours_to_draw], 0, (255, 255, 255), 1)
            # we  use morphology.binary_fill_holes to build pixel_mask from coord
            image_mask = ndimage.binary_fill_holes(image_mask).astype(int)
            pix_mask = np.argwhere(image_mask)
            pix_mask = [(pix[0], pix[1], 1) for pix in pix_mask]

            pix_masks.append(pix_mask)
            image_masks.append(image_mask)

    else:
        return None, None, None

    return area_names, pix_masks, image_masks


def add_wf_roi(ps, pix_masks, img_masks):
    for cell in np.arange(len(pix_masks)):
        pix_mask = pix_masks[cell]
        image_mask = img_masks[cell]
        ps.add_roi(pixel_mask=pix_mask, image_mask=image_mask)

import numpy as np
from PIL import ImageSequence
from ScanImageTiffReader import ScanImageTiffReader
import PIL.Image as pil_image
import time


def load_tiff_movie_in_memory_using_pil(tif_movie_file_name, frames_to_add=None):
    """
        Load tiff movie from filename using PIL library

        Args:
            tif_movie_file_name (str) : Absolute path to tiff movie
            frames_to_add: dict with key an int representing the frame index after which add frames.
                the value is the number of frames to add (integer)

        Returns:
            tiff_movie (array) : Tiff movie as 3D-array
    """
    if frames_to_add is None:
        frames_to_add = dict()

    start_time_timer = time.time()
    im = pil_image.open(tif_movie_file_name)
    n_frames = len(list(ImageSequence.Iterator(im)))
    dim_y, dim_x = np.array(im).shape
    print(f"n_frames {n_frames}, dim_x {dim_x}, dim_y {dim_y}")

    if len(frames_to_add) > 0:
        n_frames += np.sum(list(frames_to_add.values()))
    tiff_movie = np.zeros((n_frames, dim_y, dim_x), dtype="uint16")
    frame_index = 0
    for frame, page in enumerate(ImageSequence.Iterator(im)):
        tiff_movie[frame_index] = np.array(page)
        frame_index += 1
        # adding blank frames
        if frame in frames_to_add:
            frame_index += frames_to_add[frame]
    stop_time_timer = time.time()
    print(f"Time for loading movie: "
          f"{np.round(stop_time_timer - start_time_timer, 3)} s")
    return tiff_movie


def load_tiff_movie_in_memory(tif_movie_file_name, frames_to_add=None):
    """
        Load tiff movie from filename using Scan Image Tiff

        Args:
            tif_movie_file_name (str) : Absolute path to tiff movie

        Returns:
            tiff_movie (array) : Tiff movie as 3D-array
    """

    if tif_movie_file_name is not None:
        print(f"Loading movie")
        try:
            if (frames_to_add is not None) and (len(frames_to_add) > 0):
                return load_tiff_movie_in_memory_using_pil(tif_movie_file_name, frames_to_add)
            else:
                raise AttributeError()
        except AttributeError:
            try:
                start_time = time.time()
                tiff_movie = ScanImageTiffReader(tif_movie_file_name).data()
                stop_time = time.time()
                print(f"Time for loading movie with scan_image_tiff: "
                      f"{np.round(stop_time - start_time, 3)} s")
            except Exception as e:
                return load_tiff_movie_in_memory_using_pil(tif_movie_file_name)

        return tiff_movie


def get_tiff_movie_shape(tif_movie_file_name):
    tiff_movie_shape = ScanImageTiffReader(tif_movie_file_name).shape()

    return tiff_movie_shape


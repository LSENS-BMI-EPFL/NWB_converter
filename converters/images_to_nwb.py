import os

from pynwb.base import Images
from pynwb.image import GrayscaleImage

from utils import server_paths, tiff_loading


def convert_images_data(nwb_file, config_file):
    anatomical_images = server_paths.get_anat_images_files(config_file)
    if anatomical_images is None:
        print('No anatomical images to include')
        return
    anatomical_images_to_add = []
    for image_index, image_path in enumerate(anatomical_images):
        image_data = tiff_loading.load_tiff_image(tiff_image_path=image_path)
        image_name = os.path.split(image_path)[1]
        image_to_add = GrayscaleImage(name=image_name,
                                      data=image_data,
                                      description='description',
                                      resolution=None)
        anatomical_images_to_add.append(image_to_add)

    images = Images(
        name="Anatomical images",
        images=anatomical_images_to_add,
        description="A collection of logo images presented to the subject.",
    )

    nwb_file.add_acquisition(images)


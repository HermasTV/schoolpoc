"""Face alignment module wrapper

    @author : Ahmed Hermas
    @copyright : Tahaluf UAE
"""
import logging
import numpy as np
import SCRFD.cv_align as cv_align

Array = np.ndarray

# diable pylint too-few-public-methods
# pylint: disable=R0903


class FaceAlign:
    """
    Face alignment class for face detection engine
    """

    def __init__(self, mode="default"):
        self.mode = mode
        self.logger = logging.getLogger("Face-Detector-Package")
        self._load_aligners()

    def _load_aligners(self):
        """load aligners models if any"""
        # if there are alignment methods needs loading
        return None

    def align(self, img: Array, landmarks: Array, image_size=112):
        """align face image with landmark points
        Args:
            img (): input image
            landmark: landmark points
        Returns:
            aligned image
        """
        # Add alignment methods call here
        return cv_align.norm_crop(img, landmarks, image_size)

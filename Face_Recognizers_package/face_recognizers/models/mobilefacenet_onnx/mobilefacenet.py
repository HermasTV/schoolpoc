""" Onnx_base responsible for loading onnx model
    and perform prediction over one image.

    @author : Mahnoud Ewaisha
    @copyright : Tahaluf UAE
"""
import cv2
import onnxruntime
import numpy as np


class MBFOnnx:
    """Interface for Arcface Onnx model"""

    def __init__(self, logger, configs: dict):
        self._cfg = configs
        self.logger = logger
        self.set_device(self._cfg["device"])
        self.set_thresh(self._cfg["thresh"])
        self.input_name = self.session.get_inputs()[0].name
        self.database = None

    def get_device(self):
        """get device
        Returns:
        str : device name
        """
        return self._cfg["device"]

    def set_device(self, device: str) -> None:
        """set device
        Args:
        device (str): device name
        """
        self._cfg["device"] = device
        if self._cfg["device"] == "cpu":
            self.logger.info("LOADING MODEL ON CPU")
            self.session = onnxruntime.InferenceSession(
                self._cfg["model"], providers=["CPUExecutionProvider"]
            )
        else:
            self.logger.info("LOADING MODEL ON GPU")
            self.session = onnxruntime.InferenceSession(
                self._cfg["model"], providers=["CUDAExecutionProvider"]
            )

    def get_thresh(self):
        """get threshold
        Returns:
        float : det_thresh
        """
        return self._cfg["thresh"]

    def set_thresh(self, thresh: float) -> None:
        """set det_thresh
        Args:
            det_thresh (float): threshold
        """
        self._cfg["thresh"] = thresh

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        """Image pre-processing function
        Args:
            img (np.ndarray): a numpy image in RGB format

        Returns:
            np.ndarray: processed image
        """
        blob = cv2.dnn.blobFromImages(
            [img[:, :, ::-1]],
            1.0 / 127.5,
            (112, 112),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        return blob

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Main prediction function
        Args:
            img: an image in RGB format

        Returns:
            np.ndarray: a (1, 512) embedding
        """
        outputs = self.session.run([], {self.input_name: img})[0]
        return outputs

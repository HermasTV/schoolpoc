""" Onnx_base responsible for loading onnx model
    and perform prediction over one image.

    @author : Mahnoud Ewaisha
              Anwar Alsheikh
    @copyright : Tahaluf UAE

"""

import onnxruntime
import numpy as np
import cv2


class AdafaceOnnx:
    """Interface for Arcface Onnx model"""

    def __init__(self, logger, configs: dict):
        self._cfg = configs
        self.logger = logger
        self.set_device(self._cfg["device"])
        self.set_thresh(self._cfg["thresh"])
        # self.set_thresh(self._cfg["thresh"])
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
       
        (batch_size,hight,width,channel) = img.shape
        print(img.shape)
        instack = img.transpose((1,2,3,0)).reshape((hight,width,channel*batch_size))
        outstack = cv2.resize(instack,(112,112))
        img = outstack.reshape((hight,width,channel,batch_size)).transpose((3,0,1,2))
        img = (((img[:, :, :, ::-1].astype(np.float32))/ 255.0) - 0.5) / 0.5
        img = np.transpose(img, (0, 3, 1, 2)) 
        img = img.astype("float32")
       

        return img

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Main prediction function
        Args:
            img: an image in RGB format

        Returns:
            np.ndarray: a (1, 512) embedding
        """
        
        outputs = self.session.run([], {self.input_name: img})[0]
        
        return outputs

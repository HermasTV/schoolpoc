from SCRFD.detect import SCRFD
from SCRFD.scrfd_inferance import SCRFD_TRT
from SCRFD.scrfd_trt_batch import SCRFD_TRT_BATCH
import utils.face_utils as utils
from SCRFD.alignment import FaceAlign
import utils.utils as utils_ 
from Face_Recognizers_package.face_recognizers.recognizer import Recognizer
from recognizer_sbatch_trt import ArcFace_TRT
import pdb
# Triton Version
from SCRTrT import faceDetectionSCRFD as FD_Triton
from arcTrT import faceRecognizer as FR_Triton

class Models():
    def __init__(self) -> None:
        self.aligner = FaceAlign()
        self.config = utils_.load_configs("../config.yaml")

        self.detector_config = self.config[self.config["detector"]]
        self.recognizer_config = self.config[self.config["recognizer"]]
        self.recognizer = self._init_recognizer()
        self.detector = self._init_detector()


    def _init_detector(self):
            '''
            Args:
                none
            Output:
                none
            '''
            if self.config["detector"]== 'detector2.5-trt':
                return SCRFD_TRT(self.detector_config)
            elif self.config["detector"]== 'detector10-trt':
                return SCRFD_TRT(self.detector_config)
            elif self.config["detector"]== 'detector2.5-trt-batch':
                return SCRFD_TRT_BATCH(self.detector_config)
            elif self.config["detector"]== 'detector10-trt-batch':
                return SCRFD_TRT_BATCH(self.detector_config)
                
            elif self.config["detector"]== 'detector10-onnx':
                return SCRFD(self.detector_config)
            elif self.config["detector"]== 'detector2.5-onnx':
                return SCRFD(self.detector_config)
            elif self.config["detector"]== "detector-triton":
                return FD_Triton("configs/SCR_TRT.yaml",version="two")
                
            else:
                raise Exception("Model type not supported")
            

    def _init_recognizer(self):
        '''
        Args:
            none
        Output:
            none
        '''
        if self.config["recognizer"]== 'arcface-r50-trt':
            return ArcFace_TRT(self.recognizer_config)
        elif self.config["recognizer"]== 'arcface-batched-trt':
            return ArcFace_TRT(self.recognizer_config)
        elif self.config["recognizer"]== 'arcface-r18-trt':
            return ArcFace_TRT(self.recognizer_config)
        elif self.config["recognizer"]== 'arcface-r18-batched-trt':
            return ArcFace_TRT(self.recognizer_config)
        elif self.config["recognizer"]== 'mobilenet-trt':
            return ArcFace_TRT(self.recognizer_config)
        elif self.config["recognizer"]== 'mobilenet-batched-trt':
            return ArcFace_TRT(self.recognizer_config)
        elif self.config["recognizer"]== 'adaface-r50-trt':
            return ArcFace_TRT(self.recognizer_config)
        elif self.config["recognizer"]== 'adaface-r50-batched-trt':
            return ArcFace_TRT(self.recognizer_config)
        elif self.config["recognizer"]== 'adaface-r18-trt':
            return ArcFace_TRT(self.recognizer_config)
        elif self.config["recognizer"]== 'adaface-r18-batched-trt':
            return ArcFace_TRT(self.recognizer_config)
        

        elif self.config["recognizer"]== 'arcface50-onnx':
            return Recognizer(self.recognizer_config["model"])
        elif self.config["recognizer"]== 'arcface18-onnx':
            return Recognizer(self.recognizer_config["model"])
        elif self.config["recognizer"]== 'adaface50-onnx':
            return Recognizer(self.recognizer_config["model"])
        elif self.config["recognizer"]== 'adaface18-onnx':
            return Recognizer(self.recognizer_config["model"])
        elif self.config["recognizer"]== 'mobilenet-onnx':
            return Recognizer(self.recognizer_config["model"])
        elif self.config["recognizer"]== "arcface-triton":
            return FR_Triton("configs/arc.yaml",version="trt")
        else:
            raise Exception("Model type not supported")
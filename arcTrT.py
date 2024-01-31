import sys

sys.path.append("./")
import numpy as np
from typing import Any,List
from concurrent.futures import ThreadPoolExecutor
import grpc
from tritonclient.utils import InferenceServerException
import tritonclient.grpc as grpcclient
from utils_e import *
import pdb

class faceRecognizer:
    
    def __init__(self,config_path,version="trt") -> None:

        self.config = load_configs(config_path)
        self.engine = self.config["engine"]
        self.model_name = self.config["model_name"][version]
        self.verbose = self.config["verbose"]
        self.model_info = self.config["model_info"]

        self.processing_pool = ThreadPoolExecutor()
        self.infer_pool = ThreadPoolExecutor()
        self.postprocessing_pool = ThreadPoolExecutor()

        self.backend = self.config["backend"] if "backend" in self.config.keys() else "tritonserver"

        if self.backend == "tritonserver":
            self.url = self.config["url"]
            self.model_version = self.config["model_version"] if "model_version" in self.config.keys() else ""
            self.client_timeout = self.config["client_timeout"] if "client_timeout" in self.config.keys() else None

            # Create gRPC stub for communicating with the server
            channel = grpc.insecure_channel(self.url)
            grpc_stub = grpcclient.service_pb2_grpc.GRPCInferenceServiceStub(
                channel)

            metadata_request = grpcclient.service_pb2.ModelMetadataRequest(
                name=self.model_name, version=self.model_version)
            metadata_response = grpc_stub.ModelMetadata(metadata_request)

            self.INPUT_NAMES = [
                input.name for input in metadata_response.inputs]
            self.OUTPUT_NAMES = [
                output.name for output in metadata_response.outputs]

            # Create server context
            try:
                self.triton_client = grpcclient.InferenceServerClient(
                    url=self.url,
                    verbose=self.verbose)
            except Exception as e:
                print("context creation failed: " + str(e))
                sys.exit()

            # Health check
            if not self.triton_client.is_server_live():
                print("FAILED : is_server_live")
                sys.exit(1)

            if not self.triton_client.is_server_ready():
                print("FAILED : is_server_ready")
                sys.exit(1)

            if not self.triton_client.is_model_ready(self.model_name):
                print("FAILED : is_model_ready")
                sys.exit(1)

            if self.model_info:
                # Model metadata
                try:
                    metadata = self.triton_client.get_model_metadata(
                        self.model_name)
                    print(metadata)
                except InferenceServerException as ex:
                    if "Request for unknown model" not in ex.message():
                        print("FAILED : get_model_metadata")
                        print("Got: {}".format(ex.message()))
                        sys.exit(1)
                    else:
                        print("FAILED : get_model_metadata")
                        sys.exit(1)

                # Model configuration
                try:
                    config = self.triton_client.get_model_config(
                        self.model_name)
                    if not (config.config["name"] == self.model_name):
                        print("FAILED: get_model_config")
                        sys.exit(1)
                    print(config)
                except InferenceServerException as ex:
                    print("FAILED : get_model_config")
                    print("Got: {}".format(ex.message()))
                    sys.exit(1)

            self.infer = self.infer_TRITONSERVER_batch

        else:
            raise NotImplementedError(
                "Only supporting TritonServer and ONNXRT backend, while you selected {}".format(self.backend))
       
        print(
            "{} {} Successfully Initialized".format(
                self.engine, self.model_name
            )
        )

    def prepare_TRITONSERVER(self, batch_size):
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(self.INPUT_NAMES[0], [batch_size,3,112,112], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[0]))
        return inputs, outputs
    
    def preprocess_sample(self, img: np.ndarray):
        """Image pre-processing function
        Args:
            img (np.ndarray): a numpy image in RGB format

        Returns:
            np.ndarray: processed image
        """
        img_res = img.copy()
        img_res = np.reshape(img_res, (112, 112,3))
        img_res = ((img_res[:, :, ::-1]/255.0) - 0.5) / 0.5
        img_res = np.transpose(img_res, (2, 0, 1))
        img_res = np.expand_dims(img_res, axis=0)
        img_res = img_res.astype("float32")
        return img_res
    
    def preprocess(self, images: List[np.ndarray]): 
        if type(images) == np.ndarray:
            images = [images]
        
        processed_images= []
        outputs= self.processing_pool.map(self.preprocess_sample, images)
        for output in outputs:
            processed_images.append(output[0])

        processed_images = np.asarray(processed_images)
        return processed_images

    def infer_TRITONSERVER_batch(self, processed_images, input_ports, output_ports):
        input_ports[0].set_data_from_numpy(processed_images)
        results = self.triton_client.infer(
            model_name = self.model_name,
            inputs=input_ports,
            outputs=output_ports,
            client_timeout=self.client_timeout
        )

        if self.model_info:
            statistics = self.triton_client.get_inference_statistics(
                model_name=self.model_name)
            if len(statistics.model_stats) != 1:
                print("FAILED: get_inference_statistics")
                sys.exit(1)
            print(statistics)

        output0 = results.as_numpy(
            self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("683")])

        return output0  


    def predict(self, images):
        processed_images = self.preprocess(images)
        input_ports, output_ports = self.prepare_TRITONSERVER(batch_size=len(processed_images))
        embeds = self.infer_TRITONSERVER_batch(processed_images, input_ports, output_ports)
        return embeds

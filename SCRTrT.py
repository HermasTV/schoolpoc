import sys
from tritonclient.utils import InferenceServerException
from utils_e import *
import glob
# from commons.utils.utils import get_config_from_json
import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import grpc
from concurrent.futures import ThreadPoolExecutor
from typing import List,Any
import time
import os
sys.path.append("./")


    
class faceDetectionSCRFD:
    
    def __init__(self, config_path,version="two"):
        self.config = load_configs(config_path)["TRT"]
        self.engine = self.config["engine"]
        self.model_name = self.config["model_name"][version]
        self.input_size = (
            self.config['input_size']['width'],
            self.config['input_size']['height'],
        )
        self.mean = self.config['mean']
        self.fmc = self.config['fmc']
        self.fpn_stride = self.config['fpn_stride']
        self.num_anchors = self.config['num_anchors']
        self.conf_thresh = self.config['conf_threshold']

        # Create a multiprocessing pool with automatically determined number of  processes
        self.processing_pool = ThreadPoolExecutor()
        self.infer_pool = ThreadPoolExecutor()
        self.postprocessing_pool = ThreadPoolExecutor()

        self.backend = self.config["backend"] if "backend" in self.config.keys() else "tritonserver"


        if self.backend == "tritonserver":
            self.url = self.config["url"]
            self.model_version = self.config["model_version"] if "model_version" in self.config.keys() else ""
            self.verbose = self.config["verbose"] if "verbose" in self.config.keys() else False
            self.model_info = self.config["model_info"] if "model_info" in self.config.keys() else False
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
        inputs.append(grpcclient.InferInput(self.INPUT_NAMES[0], [batch_size,3,self.input_size[0], self.input_size[1]], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[0]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[1]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[2]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[3]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[4]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[5]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[6]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[7]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[8]))

        return inputs, outputs

    def infer_TRITONSERVER_sample(self, input_ports, output_ports):
        def run_sample(processed_images):
            input_ports[0].set_data_from_numpy(processed_images)
            results = self.triton_client.infer(model_name=self.model_name,
                                               inputs=input_ports,
                                               outputs=output_ports,
                                               client_timeout=self.client_timeout)
            if self.model_info:
                statistics = self.triton_client.get_inference_statistics(
                    model_name=self.model_name)
                if len(statistics.model_stats) != 1:
                    print("FAILED: get_inference_statistics")
                    sys.exit(1)
                print(statistics)

            output0 = results.as_numpy(
                self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("score_8")])
            output1 = results.as_numpy(
                self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("score_16")])
            output2 = results.as_numpy(
                self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("score_32")])
            output3 = results.as_numpy(
                self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("bbox_8")])
            output4 = results.as_numpy(
                self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("bbox_16")])
            output5 = results.as_numpy(
                self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("bbox_32")])
            output6 = results.as_numpy(
                self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("kps_8")])
            output7 = results.as_numpy(
                self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("kps_16")])
            output8 = results.as_numpy(
                self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("kps_32")])

            return output0, output1, output2, output3, output4, output5, output6, output7, output8
        return run_sample

    def infer_TRITONSERVER(self, processed_images, input_ports, output_ports):
        func = self.infer_TRITONSERVER_sample(input_ports, output_ports)
        outputs = self.infer_pool.map(func, processed_images)
        return [output for output in outputs]

    def infer_TRITONSERVER_batch(self, processed_images, input_ports, output_ports):
        import pdb
        input_ports[0].set_data_from_numpy(processed_images)
        # pdb.set_trace()

        results = self.triton_client.infer(model_name = self.model_name,inputs=input_ports,outputs=output_ports,client_timeout=self.client_timeout
        )
        # pdb.set_trace()

        if self.model_info:
            statistics = self.triton_client.get_inference_statistics(
                model_name=self.model_name)
            if len(statistics.model_stats) != 1:
                print("FAILED: get_inference_statistics")
                sys.exit(1)
            print(statistics)

        output0 = results.as_numpy(
            self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("score_8")])
        output1 = results.as_numpy(
            self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("score_16")])
        output2 = results.as_numpy(
            self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("score_32")])
        output3 = results.as_numpy(
            self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("bbox_8")])
        output4 = results.as_numpy(
            self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("bbox_16")])
        output5 = results.as_numpy(
            self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("bbox_32")])
        output6 = results.as_numpy(
            self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("kps_8")])
        output7 = results.as_numpy(
            self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("kps_16")])
        output8 = results.as_numpy(
            self.OUTPUT_NAMES[self.OUTPUT_NAMES.index("kps_32")])

        return [output0, output1, output2, output3, output4, output5, output6, output7, output8]
    
    def preprocess_sample(self, image: np.ndarray) -> np.ndarray:
        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(self.input_size[1]) / self.input_size[0]
        if im_ratio>model_ratio:
            new_height = self.input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros( (self.input_size[1], self.input_size[0], 3), dtype=np.float32 )
        det_img[:new_height, :new_width, :] = resized_img

        return det_img, det_scale

    def preprocess(self, images: List[np.ndarray]):
        if type(images) == np.ndarray:
            images = [images]
        
        processed_images= []
        det_scales = []
        outputs= self.processing_pool.map(self.preprocess_sample, images)
        for output in outputs:
            processed_images.append(output[0])
            det_scales.append(output[1])

        processed_images = np.asarray(processed_images)
        det_scales = np.asarray(det_scales)
        processed_images = cv2.dnn.blobFromImages(processed_images, 1.0/128, self.input_size, self.mean, swapRB=True)
        return processed_images,det_scales

    def postprocess_sample(self, outputs: List) -> List[np.ndarray]:
        image, predictions, scale = outputs
        scores_list = []
        bboxes_list = []
        kpss_list = []
        for idx, stride in enumerate(self.fpn_stride):
            if self.backend == 'ort':
                scores = predictions[idx][0]
                bbox_preds = predictions[idx + self.fmc][0] * stride
                kps_preds = predictions[idx + self.fmc * 2][0] * stride
            else:
                scores = predictions[idx]
                bbox_preds = predictions[idx + self.fmc] * stride
                kps_preds = predictions[idx + self.fmc * 2] * stride

            height = self.input_size[1] // stride
            width = self.input_size[0] // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
            anchor_centers = np.stack([anchor_centers]*self.num_anchors, axis=1).reshape( (-1,2) )

            pos_inds = np.where(scores>=self.conf_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / scale
        kpss = np.vstack(kpss_list) / scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = nms(pre_det)
        det = pre_det[keep, :]
        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:].reshape(-1,5,2).astype(int)
        scores = det[:,4]
        dets = det[:,:4].astype(int)
        
        # found_faces = crop_faces(image, dets)
        # return [dict(bbox=dets[i], found_faces=found_faces[i], landmarks=kpss[i]) for i in range(dets.shape[0])]
        return [dets, kpss]
    
    def postprocess(self, images, outputs: List[List], scales) -> List:
        if len(np.asarray(images).shape)==3:
            images = [images]
        
        if self.backend != 'ort':
            batch_outputs = []
            for i in range(len(outputs[0])):
                batch_outputs.append([outputs[0][i], outputs[1][i], outputs[2][i], outputs[3][i], outputs[4][i],
                                    outputs[5][i], outputs[6][i], outputs[7][i], outputs[8][i]])
        else:
            batch_outputs = outputs
        outputs = self.postprocessing_pool.map(
            self.postprocess_sample, list(zip(images, batch_outputs, scales)))
        return [output for output in outputs]

    def predict(self, images):
        import pdb
        # pdb.set_trace()
        images = images[..., ::-1]
        
        processed_images, det_scales = self.preprocess(images)
        # pdb.set_trace()
        
        input_ports, output_ports = self.prepare_TRITONSERVER(batch_size=len(processed_images))
        # pdb.set_trace()

        batch_predictions = self.infer_TRITONSERVER_batch(processed_images, input_ports, output_ports)
        # pdb.set_trace()

        batch_detections = self.postprocess(images, batch_predictions, scales=det_scales)
        # pdb.set_trace()

        return batch_detections[0]


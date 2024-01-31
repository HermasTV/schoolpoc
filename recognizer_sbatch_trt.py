'''recognizer trt inferance
    @authors:Anwar Alsheikh     
    @lisence: Tahaluf 2023
'''
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import tensorrt as trt
import time
import os


class ArcFace_TRT:
    def __init__(self,configs):

        self.configs = configs
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins

        with open(self.configs["model"], "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)

        self.num_bindings = engine.num_bindings
        binding_names = []
        binding_shapes = []
        for i in range(self.num_bindings):
            binding_names.append(engine.get_binding_name(i))
            binding_shapes.append(engine.get_binding_shape(i))
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def pre_process(self, img):
        """Image pre-processing function
        Args:
            img (np.ndarray): a numpy image in RGB format

        Returns:
            np.ndarray: processed image
        """
        img_res = img.copy()
        img_res = np.reshape(img_res, (112, 112,3))
        # img_res = cv2.resize(img_res, (112, 112))
        img_res = ((img_res[:, :, ::-1]/255.0) - 0.5) / 0.5
        img_res = np.transpose(img_res, (2, 0, 1))
        img_res = np.expand_dims(img_res, axis=0)
        img_res = img_res.astype("float32")
        # img_res = None
        # (batch_size,channel,hight,width) = img.shape
        # if hight == width == 112:
        #     # normalize image of shape (1,3,112,112)
        #     img_res = (((img[:,::-1, :, : ].astype(np.float32))/ 255.0) - 0.5) / 0.5
        #     # print("img_res", img_res.shape)
        # else :
        #     instack = img.transpose((2,3,1,0)).reshape((hight,width,channel*batch_size))
        #     # print("instack", instack.shape)
        #     outstack = cv2.resize(instack,(112,112))
        #     # print("outstack", outstack.shape)
        #     img = outstack.reshape((hight,width,channel,batch_size)).transpose((3,0,1,2))
        #     # print("img after outstack-reshape", img.shape)
        #     img = (((img[:, :, :, ::-1].astype(np.float32))/ 255.0) - 0.5) / 0.5
        #     # print("img after normalize", img.shape)
        #     img = np.transpose(img, (0, 3, 1, 2)) 
        #     # print("img after transpose", img.shape)
        #     img = img.astype("float32")
        return img_res


    def predict(self, input_data):
        # Preprocess input data
        preprocessed_input = self.pre_process(input_data)

        self.inputs[0]['host'] = np.ravel(preprocessed_input)

        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]

        if len(data) == 1:
            data = data[0].reshape(1,512)
        else:
            data = data[1].reshape(1,512)

        return data
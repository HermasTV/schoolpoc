'''scrfd trt batch inferance
    @authors:Anwar Alsheikh     
    @lisence: Tahaluf 2023
'''
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import time
import os
import concurrent.futures as cf


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class SCRFD_TRT_BATCH():
    def __init__(self, configs):
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
        # Print binding names and shapes
        for i in range(self.num_bindings):
            print(f"Binding Name {i}: {binding_names[i]}, Binding Shape {i}: {binding_shapes[i]}")
        # self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong

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
        self.center_cache = {}
        self.nms_thresh = 0.4

    def dummy(self,img):
        
        self.inputs[0]['host'] = np.ravel(img)
        
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        start = time.time()
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        print(f'dummy inferance FD execution took: {(time.time()-start) * 1000} ms')
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        data = [out['host'] for out in self.outputs]

        return data

    def pre_process_single(self,img,input_size):
        new_height, new_width = input_size
        
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        
        det_img[:new_height, :new_width, :] = resized_img
        
        blob = cv2.dnn.blobFromImage(det_img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        
        return blob
    def predict(self, images, thresh=0.5, input_size = None):
        
        input_size = (640, 640)
        new_height, new_width = input_size
        
        start_preprocess = time.time()
        det_imgs = []
        det_scale = float(new_height) / images[0].shape[0]
        with cf.ThreadPoolExecutor() as executor:
            output1 = executor.submit(self.pre_process_single,images[0],input_size)
            output2 = executor.submit(self.pre_process_single,images[1],input_size)
            output3 = executor.submit(self.pre_process_single,images[2],input_size)
            output4 = executor.submit(self.pre_process_single,images[3],input_size)
    
            result_one = output1.result()
            result_two = output2.result()
            result_three = output3.result()
            result_foure = output4.result()

            det_imgs = [result_one,result_two,result_three,result_foure]
        
        
        # print(f'pre_processing took: {(time.time()-start_preprocess) * 1000} ms')
        
        output = self.forward(det_imgs,det_scale)

        return output
    

    def forward(self, imgs,det_scale):

        imgs = np.array(imgs)
        reshaped_imgs= np.reshape(imgs,(4,3,640,640))
        self.inputs[0]['host'] = np.ravel(reshaped_imgs)
        
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        start = time.time()
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # print(f'inferance execution took: {(time.time()-start) * 1000} ms')
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        data = [out['host'] for out in self.outputs]
        
        start_postprocess = time.time()
        scr8 = data[0].reshape(4, 12800, 1)
        bb8 = data[1].reshape(4, 12800, 4)
        kps8 = data[2].reshape(4, 12800, 10)
        scr16 = data[3].reshape(4, 3200, 1)
        bb16 =data[4].reshape(4, 3200, 4)
        kps16 = data[5].reshape(4, 3200, 10)
        scr32= data[6].reshape(4, 800, 1)
        bb32=data[7].reshape(4, 800, 4)
        kps32=data[8].reshape(4, 800, 10)

        img1 = [scr8[0].reshape(1,-1,1),scr16[0].reshape(1,-1,1),scr32[0].reshape(1,-1,1),bb8[0].reshape(1,-1,4),bb16[0].reshape(1,-1,4),bb32[0].reshape(1,-1,4),kps8[0].reshape(1,-1,10),kps16[0].reshape(1,-1,10),kps32[0].reshape(1,-1,10)]
        img2 = [scr8[1].reshape(1,-1,1),scr16[1].reshape(1,-1,1),scr32[1].reshape(1,-1,1),bb8[1].reshape(1,-1,4),bb16[1].reshape(1,-1,4),bb32[1].reshape(1,-1,4),kps8[1].reshape(1,-1,10),kps16[1].reshape(1,-1,10),kps32[1].reshape(1,-1,10)]
        img3 = [scr8[2].reshape(1,-1,1),scr16[2].reshape(1,-1,1),scr32[2].reshape(1,-1,1),bb8[2].reshape(1,-1,4),bb16[2].reshape(1,-1,4),bb32[2].reshape(1,-1,4),kps8[2].reshape(1,-1,10),kps16[2].reshape(1,-1,10),kps32[2].reshape(1,-1,10)]
        img4 = [scr8[3].reshape(1,-1,1),scr16[3].reshape(1,-1,1),scr32[3].reshape(1,-1,1),bb8[3].reshape(1,-1,4),bb16[3].reshape(1,-1,4),bb32[3].reshape(1,-1,4),kps8[3].reshape(1,-1,10),kps16[3].reshape(1,-1,10),kps32[3].reshape(1,-1,10)]
        
        
        with cf.ThreadPoolExecutor() as executor:

            output1 = executor.submit(self.post_process_single,reshaped_imgs,img1,det_scale,thresh=0.5,max_num=0,metric='default')
            output2 = executor.submit(self.post_process_single,reshaped_imgs,img2,det_scale,thresh=0.5,max_num=0,metric='default')
            output3 = executor.submit(self.post_process_single,reshaped_imgs,img3,det_scale,thresh=0.5,max_num=0,metric='default')
            output4 = executor.submit(self.post_process_single,reshaped_imgs,img4,det_scale,thresh=0.5,max_num=0,metric='default')
    
            result_one = output1.result()
            result_two = output2.result()
            result_three = output3.result()
            result_foure = output4.result()

            output = [result_one,result_two,result_three,result_foure]

            

        # print(f'post_processing took: {(time.time()-start_postprocess) * 1000} ms')

        return output
    
    def post_process_single(self,image,output,det_scale,thresh=0.5,max_num=0,metric='default'):

        scores_list = []
        bboxes_list = []
        kpss_list = []

        if len(output[0].shape) == 3: 
            batched = True
        use_kps = False
        num_anchors = 1
        
        if len(output)==6:
            fmc = 3
            feat_stride_fpn = [8, 16, 32]
            num_anchors = 2
        elif len(output)==9:
            fmc = 3
            feat_stride_fpn = [8, 16, 32]
            num_anchors = 2
            use_kps = True
        elif len(output)==10:
            fmc = 5
            feat_stride_fpn = [8, 16, 32, 64, 128]
            num_anchors = 1
        elif len(output)==15:
            fmc = 5
            feat_stride_fpn = [8, 16, 32, 64, 128]
            num_anchors = 1
            use_kps = True
        

        input_height = image.shape[2]
        input_width = image.shape[3]
        
        
        for idx, stride in enumerate(feat_stride_fpn):
            # If model support batch dim, take first output
            if batched:
                scores = output[idx][0]
                bbox_preds = output[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                
                if use_kps:
                    kps_preds = output[idx + fmc * 2][0] * stride
                
            # If model doesn't support batching take output as is
            else:
                scores = output[idx]
                bbox_preds = output[idx + fmc]
                bbox_preds = bbox_preds * stride
                
                if use_kps:
                    kps_preds = output[idx + fmc * 2] * stride
           
            
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
        
            
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                #solution-1, c style:
                #anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                #for i in range(height):
                #    anchor_centers[i, :, 1] = i
                #for i in range(width):
                #    anchor_centers[:, i, 0] = i

                #solution-2:
                #ax = np.arange(width, dtype=np.float32)
                #ay = np.arange(height, dtype=np.float32)
                #xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                #anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                #solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                #print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers
                    
                

            pos_inds = np.where(scores>=thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            if use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

       
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        
        if use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if use_kps:
            
            kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            
            
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        
        
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

        
        
# if __name__ == "__main__":
    
#     # Path to the TensorRT engine file
#     engine_path = "/data/Biometri-dev/assets/models/FD/tensorrt/scrfd2.5_32_4.engine"
#     # Resize images to a specific width and height
#     data_path = "/data/Biometri-dev/Anwar/TAH_Packages/FR_tensort/SCRFD/tests/data/"
#     images = [] 
#     for img_path in os.listdir(data_path):
#         image_path = os.path.join(data_path, img_path)
#         img = cv2.imread(image_path)
#         resized_image = cv2.resize(img, (640, 640))
#         images.append(resized_image)
    
#     dummy_input_det = np.random.rand(4,640,640,3).astype(np.float32)
#     image_tensor = np.array(images)
#     inference = SCRFD_TRT_BATCH(engine_path) 
#     inference.dummy(dummy_input_det)
#     start = time.time()
#     outputs = inference.predict(image_tensor,thresh = 0.5,input_size = (640, 640))
#     print(outputs)
#     print(f'inferance pipline took: {(time.time()-start) * 1000} ms')
#     for idx, img in enumerate(images):
#         # Assuming each image corresponds to one output in the list
#         output = outputs[idx]

#         for bbox, kps in zip(output[0], output[1]):
#             x1, y1, x2, y2, score = bbox.astype(np.int32)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

#             for kp in kps:
#                 kp = kp.astype(np.int32)
#                 cv2.circle(img, tuple(kp), 1, (0, 0, 255), 2)

#         cv2.imwrite(f'/data/Biometri-dev/Anwar/TAH_Packages/FR_tensort/cropped/output2.5_{idx}.jpg', img)
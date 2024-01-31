'''scrfd trt inferance
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
import pdb


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

class SCRFD_TRT():
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
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.thresh = 0.5

    def predict(self, img, thresh=0.5, input_size= None, max_num=0, metric='default'):

        input_size = (640,640)
        
        #pre-processing resize and normaliz
        
        im_ratio = float(img.shape[0]) / img.shape[1]

        if im_ratio> float(input_size[1]) / input_size[0]:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        # print(resized_img.shape)
        # cv2.imshow('resized_img',resized_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # resize img to 1/4 of its original size using fx and fy
        
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        det_img[:new_height, :new_width, :] = resized_img
        input_size = tuple(det_img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(det_img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        
        # fd_preproc.append(time.time()-start)

        self.inputs[0]['host'] = np.ravel(blob)
        
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
        
        start= time.time()
        data_u = [np.reshape(data[0],(1,-1,1)),np.reshape(data[3],(1,-1,1)),np.reshape(data[6],(1,-1,1)),np.reshape(data[1],(1,-1,4)),np.reshape(data[4],(1,-1,4)),np.reshape(data[7],(1,-1,4)),np.reshape(data[2],(1,-1,10)),np.reshape(data[5],(1,-1,10)),np.reshape(data[8],(1,-1,10))]
        

        scores_list = []
        bboxes_list = []
        kpss_list = []

        if len(data_u[0].shape) == 3:
            batched = True

        use_kps = False
        num_anchors = 1
        
        if len(data_u)==6:
            fmc = 3
            feat_stride_fpn = [8, 16, 32]
            num_anchors = 2
        elif len(data_u)==9:
            fmc = 3
            feat_stride_fpn = [8, 16, 32]
            num_anchors = 2
            use_kps = True
        elif len(data_u)==10:
            fmc = 5
            feat_stride_fpn = [8, 16, 32, 64, 128]
            num_anchors = 1
        elif len(data_u)==15:
            fmc = 5
            feat_stride_fpn = [8, 16, 32, 64, 128]
            num_anchors = 1
            use_kps = True
        

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        
        for idx, stride in enumerate(feat_stride_fpn):
            
           
            # If model support batch dim, take first output
            if batched:
                scores = data_u[idx][0]
                bbox_preds = data_u[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if use_kps:
                    kps_preds = data_u[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = data_u[idx]
                bbox_preds = data_u[idx + fmc]
                bbox_preds = bbox_preds * stride
                if use_kps:
                    kps_preds = data_u[idx + fmc * 2] * stride
           

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
                

            pos_inds = np.where(scores>=self.thresh)[0]
            
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                #kpss = kps_preds
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
#     engine_path = "/data/Biometri-dev/Anwar/TAH_Packages/FR_tensort/scrfd.engine"
#     image_path = "/data/Biometri-dev/Anwar/Face_Recognizers_package/test/assets/fixtures/players/Mo Salah/side.jpg"
#     image= cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
#     # Initialize the InferenceHelper and load the model
#     inference = infer(engine_path)
#     bboxes, kpss = inference.infer(image,thresh = 0.5,input_size = (640, 640))
#     print(bboxes, kpss)
#     for i in range(bboxes.shape[0]):
#             bbox = bboxes[i]
#             x1,y1,x2,y2,score = bbox.astype(np.int32)
#             cv2.rectangle(image, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
#             if kpss is not None:
#                 kps = kpss[i]
#                 for kp in kps:
#                     kp = kp.astype(np.int32)
#                     cv2.circle(image, tuple(kp) , 1, (0,0,255) , 2)
#     filename = image_path.split('/')[-1]
#     print('output:', filename)
#     cv2.imwrite('./data%s'%filename, image)

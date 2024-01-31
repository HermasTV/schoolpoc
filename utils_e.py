# pylint: skip-file
import numpy as np
import cv2
from easydict import EasyDict
import json
import os
from typing import Any
import yaml
import csv
from skimage import transform as trans

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

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

def read_embd(csv_file)-> np.ndarray:
    '''
    Output:
        ids: Array, database ids .
        Embeddings: Array, database embeddings.
    '''
    ids= []
    embeddings= []
    with open(csv_file, mode='r') as file:
        reader= csv.reader(file)
        for row in reader:
            embedding= row[:-1]
            ids.append(row[-1])
            embeddings.append(embedding)
    return np.array(ids), np.array(embeddings)

def nms(dets,nms_thresh=0.4):
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

        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def load_configs(configs: Any) -> dict:
        """load configs from yaml file

        Args:
            config_path (any): path to yaml file or dict
        """
       
        if (isinstance(configs, str)) and (configs.endswith(".yaml")):
            with open(
                os.path.join(
                    os.path.dirname(__file__),
                    configs,
                ),
                encoding="utf-8",
            ) as file:
                try:
                    configs = yaml.safe_load(file)
                except (ValueError, KeyError) as exc:
                    raise ValueError(" WRONG MODEL CONFIG NAME!") from exc
        else:
            raise ValueError("CONFIGS MUST BE DICT OR YAML FILE")
        
        return configs



def crop_faces(image, dets):
    found_faces = []
    for det in dets:
        [xmin, ymin, xmax, ymax] = det
        found_faces.append(
            image[ymin: ymax, xmin: xmax, :]
        )
    return found_faces

def visualize_detections(image, detections):
    for [xmin, ymin, xmax, ymax] in detections:
        image = cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymax), (255, 0, 0), 2)

Array = np.ndarray
arcface_src = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def norm_crop(img, landmark, image_size=112):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
    else:
        ratio = float(image_size) / 128.0
    dst = arcface_src * ratio
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def align(img: Array, landmarks: Array, image_size=112):
        """align face image with landmark points
        Args:
            img (): input image
            landmark: landmark points
        Returns:
            aligned image
        """
        # Add alignment methods call here
        return norm_crop(img, landmarks, image_size)
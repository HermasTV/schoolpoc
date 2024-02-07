""" Utils for faces, crop and visualize dets """

from traceback import print_exc
import cv2
import numpy as np
import datetime


Array = np.ndarray


def crop_face(img: Array, bbox: list, ratio: float = 1)-> Array:
    """Crop face from image

    Args:
        img (Array): image
        bbox (list): bounding box
        ratio (float, optional): ratio of the cropped image. Defaults to 1.

    Returns:
        Array: cropped image
    """

    # get the width and height of the image
    img_h, img_w, _ = img.shape
    # calculate the width and height of the bounding box
    width = bbox[2] - bbox[0]  # x2 - x1
    height = bbox[3] - bbox[1]  # y2 - y1
    # calculate the center of the bounding box
    center_x = bbox[0] + width / 2
    center_y = bbox[1] + height / 2
    # calculate the width and height of the cropped image
    length = width * ratio
    # calculate the coordinates of the cropped image
    # make sure the cropped image is within the image
    crop_x1 = max(0, (center_x - length / 2))
    crop_y1 = max(0, (center_y - length / 2))
    crop_x2 = min(img_w, (center_x + length / 2))
    crop_y2 = min(img_h, (center_y + length / 2))

    crop_img = img[int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)]

    return crop_img, np.array([crop_x1, crop_y1, crop_x2, crop_y2])


def dashboard(frame: Array, width: int, login: set, logout: set) -> None:
    """Visualize the student counter dashboard

    Args:
        frame: Array, 
        width: int, 
        login: set, 
        logout: set,
    """
    # Dashboard segment
    cv2.rectangle(frame, (0, 0), (width, 20), (255, 255, 255), -1)
    cv2.rectangle(frame, (0, 0), (width, 20), (0, 0, 0), 1)  # Add border

    # Present count
    present_count = len(login - logout)
    cv2.putText(frame, f'Present: {present_count}', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Date/time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f'Date/Time: {current_datetime}', (width - 200, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Present students
    y = 40
    for name in login:
        cv2.putText(frame, f'Present: {name}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        y += 15

def boundboxes(img: Array, bbox: list, id: str, sim: float, save_path: str = None)-> None:
    """visualize results on source image

    Args:
        img: Array,
        bbox: list,
        id: str,
        sim: float,
        save_path: str,

    """
    
    if id == 'u':
        color = (0, 0, 255)  # Red color for unknown
    else:
        color = (0, 255, 0)  # Green color for known

    # Draw bounding box
    try:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 3)
    except:
        print_exc()

    # Draw text
    text = f'{id}, score: {int(sim*100)}%'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8  # Increase the font scale to make the text bigger
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    cv2.rectangle(img, (int(bbox[0]), int(bbox[1]) - text_size[1] - 5),
                  (int(bbox[0]) + text_size[0], int(bbox[1]) - 5), color, -1)
    cv2.putText(img, text, (int(bbox[0]), int(bbox[1]) - 5), font, font_scale, (255, 255, 255), thickness)
    
    if save_path:
        cv2.imwrite(save_path, img)
    

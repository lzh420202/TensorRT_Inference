import numpy as np
import cv2

color_map = [
    (54, 67, 244),
    (99, 30, 233),
    (176, 39, 156),
    (183, 58, 103),
    (181, 81, 63),
    (243, 150, 33),
    (212, 188, 0),
    (136, 150, 0),
    (80, 175, 76),
    (74, 195, 139),
    (57, 220, 205),
    (59, 235, 255),
    (0, 152, 255),
    (34, 87, 255),
    (72, 85, 121),
    (139, 125, 96),
    (246, 229, 39)]

def draw_result(boxes, labels, image):
    for i in range(boxes.shape[0]):
        bbox = boxes[i, :-1].reshape(-1, 2).round().astype(np.int32)
        idx = labels[i]
        color = color_map[idx]
        image = cv2.polylines(image, [bbox], True, color, 2)

    return image
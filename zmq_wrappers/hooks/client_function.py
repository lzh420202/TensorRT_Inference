import numpy as np
import cv2

color_map = {
    'plane': (54, 67, 244),
    'baseball-diamond': (99, 30, 233),
    'bridge': (176, 39, 156),
    'ground-track-field': (183, 58, 103),
    'small-vehicle': (181, 81, 63),
    'large-vehicle': (243, 150, 33),
    'ship': (212, 188, 0),
    'tennis-court': (136, 150, 0),
    'basketball-court': (80, 175, 76),
    'storage-tank': (74, 195, 139),
    'soccer-ball-field': (57, 220, 205),
    'roundabout': (59, 235, 255),
    'harbor': (0, 152, 255),
    'swimming-pool': (34, 87, 255),
    'helicopter': (72, 85, 121)
}

def draw_zmq_result(image, results):
    for result in results:
        bbox = np.array(result['box']).reshape(-1, 2).round().astype(np.int32)
        color = color_map[result['label']]
        image = cv2.polylines(image, [bbox], True, color, 2)
    return image

def client_payload(result: dict):
    if result:
        if result.get('TEST') == 'SUCCESS':
            print('ZeroMQ Server is Running!')
            return
        else:
            print(f'Image: {result["image"]}')
            print(f"Detect {len(result['objects'])} objects.")

    else:
        print('ZeroMQ Server is crashed!')

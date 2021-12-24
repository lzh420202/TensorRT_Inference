import cv2
import os


def server_payload(data: dict):
    if data:
        for k, v in data.items():
            if k != 'image':
                print(f'{k}: \t{v}')
        print('writing object into file.')
        print(f"Shape: {data['image'].shape}")
        cv2.imwrite(os.path.join('save', data['name']), data['image'])

    return dict(result='ok')
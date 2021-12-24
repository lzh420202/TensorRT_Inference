from multiprocessing import (Process, Queue)
import numpy as np
import cv2
import os
import math
from utils.nms import multiclass_poly_nms_rbbox
from utils.visualize import draw_result


def preprogress_data_unit(image_list, queue, normalization, mean, stdinv):
    for image_path in image_list:
        image = cv2.imread(image_path)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
        image = np.asarray(image, np.float32)
        if normalization:
            cv2.subtract(image, mean, image)  # inplace
            cv2.multiply(image, stdinv, image)  # inplace
        image = np.expand_dims(np.transpose(image, [2, 0, 1]), 0).astype(np.float32)
        queue.put(dict(image=image, name=image_path))
    queue.put(None)


def preprogress_data(image_dir, data_queue: Queue, num_processor, normalization=dict(enable=True,
                                                                                mean=[123.675, 116.28, 103.53],
                                                                                std=[58.395, 57.12, 57.375])):
    mean_ = np.array(normalization['mean'])
    mean = np.float64(mean_.reshape(1, -1))
    std = np.array(normalization['std'])
    stdinv = 1.0 / np.float64(std.reshape(1, -1))

    def is_image(path):
        return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if is_image(os.path.join(image_dir, f))]
    images.sort()
    per_list_num = math.ceil(len(images) / num_processor)
    processors = []
    for i in range(num_processor):
        lists_path = images[i*per_list_num: (i+1) * per_list_num]
        p = Process(target=preprogress_data_unit, args=(lists_path,
                                                        data_queue,
                                                        normalization['enable'],
                                                        mean,
                                                        stdinv))
        processors.append(p)
        p.start()
    return processors, len(images)


def load_images_unit(image_list, queue):
    for image_path in image_list:
        image = cv2.imread(image_path)
        queue.put(dict(image=image, name=image_path))
    queue.put(None)


def load_images(image_dir, data_queue: Queue, num_processor: int):
    def is_image(path):
        return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if is_image(os.path.join(image_dir, f))]
    images.sort()
    per_list_num = math.ceil(len(images) / num_processor)
    processors = []
    for i in range(num_processor):
        lists_path = images[i*per_list_num: (i+1) * per_list_num]
        p = Process(target=load_images_unit, args=(lists_path, data_queue))
        processors.append(p)
        p.start()
    return processors, len(images)

def preprogress_images_unit(ori_queue, out_queue, normalization, mean, stdinv):
    while True:
        data = ori_queue.get()
        if data:
            image = data['image']
            image_path = data['name']
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            image = np.asarray(image, np.float32)
            if normalization:
                cv2.subtract(image, mean, image)  # inplace
                cv2.multiply(image, stdinv, image)  # inplace
            image = np.expand_dims(np.transpose(image, [2, 0, 1]), 0).astype(np.float32)
            out_queue.put(dict(image=image, name=image_path))
        else:
            pass

    # queue.put(None)


def preprogress_images(ori_queue: Queue, out_queue: Queue, num_processor, normalization=dict(enable=True,
                                                                                             mean=[123.675, 116.28, 103.53],
                                                                                             std=[58.395, 57.12, 57.375])):
    mean_ = np.array(normalization['mean'])
    mean = np.float64(mean_.reshape(1, -1))
    std = np.array(normalization['std'])
    stdinv = 1.0 / np.float64(std.reshape(1, -1))

    processors = []
    count_n = num_processor
    count_queue = Queue(num_processor)
    for i in range(num_processor):
        p = Process(target=preprogress_images_unit, args=(ori_queue,
                                                          out_queue,
                                                          normalization['enable'],
                                                          mean,
                                                          stdinv))
        processors.append(p)
        p.start()
    return processors

def post_process_unit(input_queue, output_queue, score_thre, nms_thre, max_det):
    while True:
        input_data = input_queue.get()
        if input_data:
            boxes_, labels_ = multiclass_poly_nms_rbbox(input_data['box'], input_data['score'], score_thre, nms_thre, max_det)
            output_queue.put(dict(rboxes=boxes_, labels=labels_, file_path=input_data['file_path']))
        else:
            input_queue.put(None)
            break

def post_process(num_processor, input_queue, output_queue, score_thre, nms_thre, max_det):
    processors = []
    for i in range(num_processor):
        p = Process(target=post_process_unit, args=(input_queue,
                                                    output_queue,
                                                    score_thre,
                                                    nms_thre,
                                                    max_det))
        processors.append(p)
        p.start()
    return processors

def collect_result(cache_dir, ALL_LABEL, result_queue, draw_cfg=dict(enable=False, output_dir=None, num=20)):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    all_result = {key: "" for key in ALL_LABEL}
    draw_count = 0
    if draw_cfg['num'] == 'all':
        draw_limit = 1e9
    else:
        assert isinstance(draw_cfg['num'], int)
        assert draw_cfg['num'] > 0
        draw_limit = draw_cfg['num']
    while True:
        result = result_queue.get()
        if result:
            box = result['rboxes']
            label = result['labels']
            file_path = result['file_path']
            basename = os.path.splitext(os.path.basename(file_path))[0]
            for j, label_ in enumerate(label):
                name = ALL_LABEL[label_]
                box_str = [str(value) for value in box[j, :-1].tolist()]
                box_str_ = ' '.join(box_str)
                all_result[name] += f'{basename} {str(box[j, -1])} {box_str_}\n'
            if (draw_cfg['enable'] and draw_cfg['output_dir']) and draw_count < draw_limit:
                image = draw_result(box, label, cv2.imread(file_path))
                output_path = os.path.join(draw_cfg['output_dir'], "{}_det.png".format(basename))
                cv2.imwrite(output_path, image)
                draw_count += 1
        else:
            break
    for key, value in all_result.items():
        with open(os.path.join(cache_dir, f'Task_{key}.txt'), 'w') as f:
            f.write(value)
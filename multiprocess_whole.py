from multiprocessing import (Manager, Pool, Pipe)
import numpy as np
import cv2
import os
import math
import time
from nms import multiclass_poly_nms_rbbox, multiclass_poly_nms_rbbox_patches
from visualize import draw_result
DEBUG = False

def generate_split_box(image_shape, split_size, gap):
    height, width = image_shape
    stride_length = split_size - gap
    n_h = max(1, height // stride_length)
    n_w = max(1, width // stride_length)
    if n_h * stride_length + gap < height:
        n_h += 1
    if n_w * stride_length + gap < width:
        n_w += 1
    boxes = []
    for i in range(n_h):
        for j in range(n_w):
            offset_h = i * stride_length
            offset_w = j * stride_length
            if offset_h + split_size > height:
                offset_h = max(0, height - split_size)
            if offset_w + split_size > width:
                offset_w = max(0, width - split_size)
            boxes.append([offset_h, min(height, offset_h + split_size),
                          offset_w, min(width, offset_w + split_size)])

            # boxes.append([i * stride_length, min(height, i * stride_length + split_size),
            #               j * stride_length, min(width, j * stride_length + split_size)])
    return boxes


def preprogress_data_unit(pipe, queue, normalization):
    while True:
        data = pipe.recv()
        if data:
            src_image, boxes, meta = data
            for box in boxes:
                image = src_image[box[0]:box[1], box[2]:box[3], :].copy()
                h, w, _ = image.shape
                pad_h = meta['patch_size'] - h
                pad_w = meta['patch_size'] - w
                if pad_w > 0 or pad_h > 0:
                    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, 0)
                if DEBUG:
                    new = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    base_name = os.path.splitext(os.path.basename(meta["image_path"]))[0]
                    cv2.imwrite(os.path.join(f'/data/cache/{base_name}_{box[0]}_{box[2]}.jpg'), new)

                image = np.asarray(image, np.float32)
                if normalization['enable']:
                    cv2.subtract(image, normalization['mean'], image)
                    cv2.multiply(image, normalization['std'], image)
                image = np.expand_dims(np.transpose(image, [2, 0, 1]), 0).astype(np.float32)
                queue.put(dict(image=image,
                               image_path=meta['image_path'],
                               offset=(box[2], box[0]),
                               patch_num=meta['patch_num']))
        else:
            break

def preprogress_data_imread(image_list,
                            pipes,
                            lock: Manager().Lock,
                            split_cfg=dict(subsize=1024, gap=200)):
    for image in image_list:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        h, w, _ = img.shape
        boxes = generate_split_box((h, w), split_cfg['subsize'], split_cfg['gap'])
        per_list_num = math.ceil(len(boxes) / len(pipes))
        image_meta = dict(image_path=image, patch_size=split_cfg['subsize'], gap=split_cfg['gap'], patch_num=len(boxes))
        lock.acquire()
        t = time.time()
        for i, pipe in enumerate(pipes):
            per_boxes = boxes[i * per_list_num: (i + 1) * per_list_num]
            pipe.send((img, per_boxes, image_meta))

    for pipe in pipes:
        pipe.send(None)

def preprogress_data(image_list,
                     data_queue: Manager().Queue,
                     num_processor,
                     lock: Manager().Lock,
                     normalization=dict(enable=True,
                                        mean=[123.675, 116.28, 103.53],
                                        std=[58.395, 57.12, 57.375]),
                     split_cfg=dict(subsize=1024,
                                    gap=200)):
    mean_ = np.array(normalization['mean'])
    mean = np.float64(mean_.reshape(1, -1))
    std = np.array(normalization['std'])
    stdinv = 1.0 / np.float64(std.reshape(1, -1))
    norm_cfg = dict(enable=normalization['enable'], mean=mean, std=stdinv)


    pool = Pool(num_processor + 1)
    pipe_sends = []
    pipe_recvs = []
    for i in range(num_processor):
        send, recv = Pipe()
        pipe_sends.append(send)
        pipe_recvs.append(recv)
    pool.apply_async(preprogress_data_imread, (image_list, pipe_sends, lock, split_cfg))
    for i in range(num_processor):
        pool.apply_async(preprogress_data_unit, (pipe_recvs[i], data_queue, norm_cfg))

    return pool

def post_process_unit(input_queue: Manager().Queue, cache_queue: Manager().Queue, det_cfg):
    while True:
        input_data = input_queue.get()
        if input_data:
            boxes_, labels_ = multiclass_poly_nms_rbbox(input_data['box'],
                                                        input_data['score'],
                                                        det_cfg['score_threshold'],
                                                        det_cfg['nms_threshold'],
                                                        det_cfg['max_det_num'])
            boxes_[:, 0:8:2] += input_data['offset'][0]
            boxes_[:, 1:8:2] += input_data['offset'][1]

            cache_queue.put(dict(rboxes=boxes_,
                                 labels=labels_,
                                 image_path=input_data['image_path'],
                                 patch_num=input_data['patch_num'],
                                 class_num=input_data['score'].shape[1]))
        else:
            input_queue.put(None)
            cache_queue.put(None)
            break


def post_process_collect(cache_queue: Manager().Queue, output_queue: Manager().Queue, lock: Manager().Lock, det_cfg):
    cache_box = []
    cache_label = []
    patch_count = 0
    image_path = ''
    while True:
        cache_data = cache_queue.get()
        if cache_data:
            if image_path == '':
                image_path = cache_data['image_path']
            assert image_path == cache_data['image_path']
            cache_box.append(cache_data['rboxes'])
            cache_label.append(cache_data['labels'])
            patch_count += 1

            if patch_count == cache_data['patch_num']:
                boxes_ = np.concatenate(cache_box, axis=0)
                labels_ = np.concatenate(cache_label, axis=0)
                boxes, labels = multiclass_poly_nms_rbbox_patches(boxes_,
                                                                  labels_,
                                                                  cache_data['class_num'],
                                                                  det_cfg['nms_threshold'])
                output_queue.put(dict(rboxes=boxes, labels=labels, image_path=image_path))
                cache_box = []
                cache_label = []
                patch_count = 0
                image_path = ''
                lock.release()
                continue
        else:
            print('Close post collector.')
            output_queue.put(None)
            break


def post_process(num_processor,
                 input_queue: Manager().Queue,
                 output_queue: Manager().Queue,
                 cache_size: int,
                 lock: Manager().Lock,
                 det_cfg):
    pool = Pool(num_processor + 1)
    cache_queue = Manager().Queue(cache_size)
    for i in range(num_processor):
        pool.apply_async(func=post_process_unit, args=(input_queue, cache_queue, det_cfg))
    pool.apply_async(func=post_process_collect, args=(cache_queue, output_queue, lock, det_cfg))
    return pool


def output_result(output_dir, ALL_LABEL, result_queue, draw_cfg=dict(enable=False, num=20)):
    def make_dirs(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
    detection_result_folder = os.path.join(output_dir, 'detection')
    visualization_folder = os.path.join(output_dir, 'visualization')
    make_dirs(detection_result_folder)
    make_dirs(visualization_folder)
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
            det_str = ''
            box = result['rboxes']
            label = result['labels']
            image_path = result['image_path']
            basename = os.path.splitext(os.path.basename(image_path))[0]
            for j, label_ in enumerate(label):
                name = ALL_LABEL[label_]
                box_str = [f'{value: .3f}' for value in box[j, :-1].tolist()]
                box_str_ = ' '.join(box_str)
                det_str += f'{name} {box[j, -1]: .3f} {box_str_}\n'
            with open(os.path.join(detection_result_folder, basename + '.txt'), 'w') as f:
                f.write(det_str)
            if draw_cfg['enable'] and draw_count < draw_limit:
                image = draw_result(box, label, cv2.imread(image_path))
                output_path = os.path.join(visualization_folder, "{}_det.jpg".format(basename))
                cv2.imwrite(output_path, image)
                draw_count += 1
        else:
            break
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import sys
import time
import argparse
from model_inference import TensorRTInfer
from multiprocessing import (Manager, Pool)
from multiprocess_whole import (preprogress_data, post_process, output_result)
import copy


def parse_label_file(file):
    with open(file, 'r') as f:
        labels = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return labels


def is_image(path):
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

def main(cfg):
    assert cfg['whole_mode'] == 1
    lock = Manager().Lock()
    cfg_io = cfg['io']
    cfg_model = cfg['model']
    cfg_preprocess = cfg['preprocess']
    cfg_postprocess = cfg['postprocess']

    output_dir = os.path.realpath(cfg_io['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    ALL_LABEL = []
    if cfg_model['labels']:
        ALL_LABEL = parse_label_file(cfg_model['labels'])

    trt_infer = TensorRTInfer(cfg_model['engine_file'])
    preprogress_queue = Manager().Queue(cfg_preprocess['queue_length'])
    pre_processor_num = cfg_preprocess['num_process']

    cfg_norm = cfg_preprocess['normalization']
    normalization = dict(enable=bool(cfg_norm['enable']),
                         mean=cfg_norm['mean'],
                         std=cfg_norm['std'])
    split_cfg = cfg_preprocess['split']

    images = [os.path.join(cfg_io['input_dir'], f) for f in os.listdir(cfg_io['input_dir']) if is_image(os.path.join(cfg_io['input_dir'], f))]
    images.sort()

    preprecess_pool = preprogress_data(images, preprogress_queue, pre_processor_num, lock, normalization, split_cfg)
    post_processor_num = cfg_postprocess['num_process']

    post_process_input_queue = Manager().Queue(cfg_postprocess['queue_length'])
    post_process_output_queue = Manager().Queue(-1)
    det_cfg = dict(score_threshold=cfg_postprocess['score_threshold'],
                   nms_threshold=cfg_postprocess['nms_threshold'],
                   max_det_num=cfg_postprocess['max_det_num'])
    post_processor = post_process(post_processor_num,
                                  post_process_input_queue,
                                  post_process_output_queue,
                                  cfg_postprocess['queue_length'],
                                  lock,
                                  det_cfg)
    cfg_draw = cfg_postprocess['draw_image']
    cfg_draw = dict(enable=bool(cfg_draw['enable']), num=cfg_draw['num'])
    collector = Pool(1)
    collector.apply_async(func=output_result, args=(output_dir, ALL_LABEL, post_process_output_queue, cfg_draw))
    count = 0
    while True:
        data = preprogress_queue.get()
        if not data:
            count += 1
            if count == pre_processor_num:
                break
            else:
                continue
        bboxes, labels = trt_infer.infer(data['image'])
        post_process_input_queue.put(dict(box=bboxes,
                                          score=labels,
                                          image_path=data['image_path'],
                                          offset=data['offset'],
                                          patch_num=data['patch_num']))
    preprecess_pool.close()
    preprecess_pool.join()
    post_process_input_queue.put(None)
    post_processor.close()
    collector.close()
    post_processor.join()
    collector.join()
    print("Done!")

if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config_file", default=None, help="The serialized TensorRT engine")
    args = parser.parse_args()

    if not args.config_file:
        parser.print_help()
        print("\nThese arguments are required: -f")
        sys.exit(1)
    file = args.config_file
    with open(file, 'r') as f:
        cfg = yaml.load(f)
    main(cfg)

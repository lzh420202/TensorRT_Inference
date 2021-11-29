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
import argparse
from model_inference import TensorRTInfer
from multiprocessing import (Queue, Process, Lock, Pipe)
from multiprocess_whole import (preprocess_data, postprocess, output_result)


def parse_label_file(file):
    with open(file, 'r') as f:
        labels = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return labels


def is_image(path):
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

def print_cfg(cfg):
    print(f'{"-"*10 + " configure " + "-"*10}')

    if cfg['whole_mode']:
        print(f"Whole mode: True")
    else:
        print(f"Whole mode: False")
    print(f"Model file: {cfg['model']['engine_file']}")
    if cfg['model']['labels']:
        ALL_LABEL = parse_label_file(cfg['model']['labels'])
        label_str = '[' + ', '.join(ALL_LABEL) + ']'
        print(f"Labels: {label_str}")
    print(f"Input directory: {os.path.abspath(cfg['io']['input_dir'])}")
    print(f"Output directory: {os.path.abspath(cfg['io']['output_dir'])}")
    cfg_preprocess = cfg['preprocess']
    print(f"Preprocessor: {cfg_preprocess['num_process']}")
    print(f"Preprocessor queue: {cfg_preprocess['queue_length']}")
    cfg_norm = cfg_preprocess['normalization']
    if cfg_norm['enable']:
        print("Normalization: True")
        print(f"Normalization mean: [{', '.join([str(m) for m in cfg_norm['mean']])}]")
        print(f"Normalization std: [{', '.join([str(s) for s in cfg_norm['std']])}]")
    else:
        print("Normalization: False")
    if cfg['whole_mode']:
        cfg_split = cfg_preprocess['split']
        print(f"Split size: {cfg_split['subsize']}")
        print(f"Split gap: {cfg_split['gap']}")
    cfg_postprocess = cfg['postprocess']
    print(f"Postprocessor: {cfg_postprocess['num_process']}")
    print(f"Postprocessor queue: {cfg_postprocess['queue_length']}")
    print(f"Postprocess score threshold: {cfg_postprocess['score_threshold']}")
    print(f"Postprocess nms threshold: {cfg_postprocess['nms_threshold']}")
    print(f"Single picture max detection number: {cfg_postprocess['max_det_num']}")
    if cfg_postprocess['draw_image']['enable']:
        print("Draw result: True")
        print(f"Draw number: {cfg_postprocess['draw_image']['num']}")
    else:
        print("Draw result: False")
    print(f'{"-" * 31}')


def main(cfg):
    assert cfg['whole_mode'] == 1
    lock = Lock()
    cfg_io = cfg['io']
    cfg_model = cfg['model']
    cfg_preprocess = cfg['preprocess']
    cfg_postprocess = cfg['postprocess']

    output_dir = os.path.realpath(cfg_io['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    ALL_LABEL = []
    if cfg_model['labels']:
        ALL_LABEL = parse_label_file(cfg_model['labels'])

    print(f"Load TensorRT engine: {cfg_model['engine_file']}")
    trt_infer = TensorRTInfer(cfg_model['engine_file'])
    preprocess_queue = Queue(cfg_preprocess['queue_length'])
    preprocessor_num = cfg_preprocess['num_process']

    cfg_norm = cfg_preprocess['normalization']
    normalization = dict(enable=bool(cfg_norm['enable']),
                         mean=cfg_norm['mean'],
                         std=cfg_norm['std'])
    split_cfg = cfg_preprocess['split']

    images = [os.path.join(cfg_io['input_dir'], f) for f in os.listdir(cfg_io['input_dir']) if is_image(os.path.join(cfg_io['input_dir'], f))]
    images.sort()
    print(f"Scan input folder \"{cfg_io['input_dir']}\", include {len(images)} images.")

    result_log_recv, result_log_send = Pipe(duplex=False)
    print(f"Create preprocessor")
    preprecessor = preprocess_data(images, preprocess_queue, result_log_recv, preprocessor_num, lock, normalization, split_cfg)
    postprocessor_num = cfg_postprocess['num_process']

    postprocess_input_queue = Queue(cfg_postprocess['queue_length'])
    postprocess_output_recv, postprocess_output_send = Pipe(duplex=False)
    det_cfg = dict(score_threshold=cfg_postprocess['score_threshold'],
                   nms_threshold=cfg_postprocess['nms_threshold'],
                   max_det_num=cfg_postprocess['max_det_num'])
    print(f"Create postprocessor")
    postprocessor = postprocess(postprocessor_num,
                                postprocess_input_queue,
                                postprocess_output_send,
                                result_log_send,
                                cfg_postprocess['queue_length'],
                                lock,
                                det_cfg)
    cfg_draw = cfg_postprocess['draw_image']
    cfg_draw = dict(enable=bool(cfg_draw['enable']), num=cfg_draw['num'])
    print(f"Create result collector")
    collector = Process(target=output_result, args=(output_dir, ALL_LABEL, postprocess_output_recv, cfg_draw))

    print(f"Run preprocessor")
    for p in preprecessor:
        p.start()
    print(f"Run postprocessor")
    for p in postprocessor:
        p.start()
    print(f"Run result collector")
    collector.start()
    count = 0
    while True:
        data = preprocess_queue.get()
        if data:
            bboxes, labels = trt_infer.infer(data['image'])
            postprocess_input_queue.put(dict(box=bboxes,
                                             score=labels,
                                             image_path=data['image_path'],
                                             offset=data['offset'],
                                             patch_num=data['patch_num']))
        else:
            count += 1
            if count == preprocessor_num:
                break

    postprocess_input_queue.put(None)
    print(f"Stop preprocessor")
    for p in preprecessor:
        p.join()
    print(f"Stop postprocessor")
    for p in postprocessor:
        p.join()
    print(f"Stop result collector")
    collector.join()
    print("Done!")


if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", default=None, help="The serialized TensorRT engine")
    args = parser.parse_args()

    if not args.config_file:
        parser.print_help()
        print("\nThese arguments are required: config file")
        sys.exit(1)
    file = args.config_file
    with open(file, 'r') as f:
        cfg = yaml.load(f)
        print(f'Configure file: {os.path.abspath(file)}')
        print(f'Parse configure')
        print_cfg(cfg)

    main(cfg)

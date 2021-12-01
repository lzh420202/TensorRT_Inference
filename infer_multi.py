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
from multiprocessing import (Process, Queue)
from multiprocess_m import (preprogress_data, post_process, collect_result)
from DOTA_devkit.ResultMerge_multi_process import mergebypoly as mergebypoly_multi_process

from tqdm import tqdm
import shutil
import copy


def parse_label_file(file):
    with open(file, 'r') as f:
        labels = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return labels


def main(cfg):
    assert cfg['whole_mode'] == 0
    cfg_io = cfg['io']
    cfg_model = cfg['model']
    cfg_preprocess = cfg['preprocess']
    cfg_postprocess = cfg['postprocess']

    speed_n = cfg_model['num_speed']

    output_dir = os.path.realpath(cfg_io['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    ALL_LABEL = []
    if cfg_model['labels']:
        ALL_LABEL = parse_label_file(cfg_model['labels'])

    trt_infer = TensorRTInfer(cfg_model['engine_file'])
    queue_data = Queue(cfg_preprocess['queue_length'])
    processor_num = cfg_preprocess['num_process']

    cfg_norm = cfg_preprocess['normalization']
    normalization = dict(enable=bool(cfg_norm['enable']),
                         mean=cfg_norm['mean'],
                         std=cfg_norm['std'])
    process, image_num = preprogress_data(cfg_io['input_dir'], queue_data, processor_num, normalization)
    post_processor_num = cfg_postprocess['num_process']
    out_queue = Queue(cfg_postprocess['queue_length'])
    result_queue = Queue(-1)
    post_processor = post_process(post_processor_num,
                                  out_queue,
                                  result_queue,
                                  cfg_postprocess['score_threshold'],
                                  cfg_postprocess['nms_threshold'],
                                  cfg_postprocess['max_det_num'])
    cache_dir = os.path.join(output_dir, 'cache')
    cfg_draw = cfg_postprocess['draw_image']
    collector = Process(target=collect_result, args=(cache_dir,
                                                     ALL_LABEL,
                                                     result_queue,
                                                     dict(enable=bool(cfg_draw['enable']),
                                                          output_dir=cfg_draw['output_dir'],
                                                          num=cfg_draw['num'])))
    collector.start()
    t = time.time()
    start_time = copy.deepcopy(t)
    with tqdm(total=image_num) as pbar:
        count = 0
        while True:
            data = queue_data.get()
            if not data:
                count += 1
                if count == processor_num:
                    break
                else:
                    continue
            bboxes, labels = trt_infer.infer(data['image'])
            out_queue.put(dict(box=bboxes, score=labels, file_path=data['name']))
            pbar.update(1)
            if pbar.n == speed_n:
                print(f'\rInference {speed_n} AVG FPS: {speed_n / (time.time() - start_time)}\n', end='', flush=True)
        out_queue.put(None)
        for p in process:
            p.join()
        for p in post_processor:
            p.join()
    result_queue.put(None)
    collector.join()
    cost_time = time.time() - start_time
    print(f'Summary: Image count-{image_num}, Process time-{cost_time}, Average inference time-{cost_time/image_num}, Average FPS-{image_num/cost_time}')
    print("Finished Collecting")
    mergebypoly_multi_process(cache_dir, output_dir)
    shutil.rmtree(cache_dir)
    print("Finished Processing")

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
        cfg = yaml.safe_load(f)
    main(cfg)

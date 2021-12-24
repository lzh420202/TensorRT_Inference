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
import time

from model_inference import TensorRTInfer
from multiprocessing import (Queue, Process, Lock, Pipe)
from process.multiprocess_server import (preprocess_data, postprocess, output_result)

from utils.tools import (parse_label_file, print_cfg)
from zmq_wrappers import custom_server


def main(cfg):
    assert cfg['mode'].lower() == 'server'
    lock = Lock()
    cfg_model = cfg['model']
    cfg_preprocess = cfg['preprocess']
    cfg_postprocess = cfg['postprocess']

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

    zmq_data_queue = Queue(10)
    zmq_progressbar_queue = Queue(20)
    zmq_result_queue = Queue(10)

    server = custom_server(int(cfg['port']), zmq_data_queue, zmq_result_queue, zmq_progressbar_queue, True)

    result_log_recv, result_log_send = Pipe(duplex=False)
    print(f"Create preprocessor")
    preprecessor = preprocess_data(zmq_data_queue, preprocess_queue, result_log_recv, preprocessor_num, lock, normalization, split_cfg)
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
    print(f"Create result collector")
    output_processor = Process(target=output_result, args=(zmq_result_queue, ALL_LABEL, postprocess_output_recv))

    print(f"Run preprocessor")
    for p in preprecessor:
        p.start()
    print(f"Run postprocessor")
    for p in postprocessor:
        p.start()
    print(f"Run result collector")
    output_processor.start()
    count = 0
    current = 0
    while True:
        data = preprocess_queue.get()
        if data:
            bboxes, labels = trt_infer.infer(data['image'])
            current += 1
            start_time = data['start_time']
            postprocess_input_queue.put(dict(box=bboxes,
                                             score=labels,
                                             image_path=data['image_path'],
                                             offset=data['offset'],
                                             patch_num=data['patch_num'],
                                             start_time=start_time))
            progressbar = dict(type='detect',
                               current=current,
                               total=data['patch_num'],
                               used_time=time.time()-start_time,
                               description='detection task')
            if current == data['patch_num']:
                progressbar.update(dict(done=True, run=True))
                current = 0
            else:
                progressbar.update(dict(done=False, run=True))
            zmq_progressbar_queue.put(progressbar)
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
    output_processor.join()
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
        cfg = yaml.safe_load(f)
        print(f'Configure file: {os.path.abspath(file)}')
        print(f'Parse configure')
        print_cfg(cfg)

    main(cfg)

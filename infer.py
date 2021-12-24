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
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda

from image_batcher import ImageBatcher
from utils.nms import multiclass_poly_nms_rbbox
# from visualize import visualize_detections
from tqdm import tqdm

# dota_10 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
#            'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor',
#            'swimming-pool', 'helicopter']

SPEED_MODE = False

class TensorRTInfer:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch, score_threshold, nms_threshold, max_det=2000):
        # Prepare the output data
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])

        rboxes = outputs[0]
        scores = outputs[1]
        boxes = []
        labels = []
        for i in range(rboxes.shape[0]):
            boxes_, labels_ = multiclass_poly_nms_rbbox(rboxes[i], scores[i], score_threshold, nms_threshold, max_det)
            boxes.append(boxes_)
            labels.append(labels_)

        return boxes, labels

def parse_label_file(file):
    with open(file, 'r') as f:
        labels = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return labels

def main(args):
    output_dir = os.path.realpath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    ALL_LABEL = []
    if args.labels:
        ALL_LABEL = parse_label_file(args.labels)

    trt_infer = TensorRTInfer(args.engine)
    batcher = ImageBatcher(args.input, *trt_infer.input_spec())
    all_result = {key: "" for key in ALL_LABEL}
    t = time.time()
    section = 200
    with tqdm(total=batcher.num_batches) as pbar:
        for batch, images, scales in batcher.get_batch():
            # print("Processing Image {} / {}".format(batcher.image_index, batcher.num_images), end="\r")
            bboxes, labels = trt_infer.infer(batch, 0.1, args.nms_threshold, 2000)
            if not SPEED_MODE:
                for i in range(len(images)):
                    basename = os.path.splitext(os.path.basename(images[i]))[0]

                    box = bboxes[i]
                    label = labels[i]
                    for j, label_ in enumerate(label):
                        name = ALL_LABEL[label_]
                        box_str = [str(value) for value in box[j, :-1].tolist()]
                        box_str_ = ' '.join(box_str)
                        all_result[name] += f'{basename} {str(box[j, -1])} {box_str_}\n'
                    # image = draw_result(box, label, cv2.imread(images[i]))
                    # output_path = os.path.join(output_dir, "{}.png".format(basename))
                    # cv2.imwrite(output_path, image)
            pbar.update(1)
            if pbar.n % section == 0:
                print('FPS: {}.'.format(section / (time.time() - t)))
                t = time.time()

    # cache = os.path.join(output_dir, 'cache')
    # os.makedirs(cache, exist_ok=True)
    # for key, value in all_result.items():
    #     with open(os.path.join(cache, f'Task_{key}.txt'), 'w') as f:
    #         f.write(value)
    # mergebypoly_multi_process(cache, output_dir)
    # shutil.rmtree(cache)

    print("Finished Processing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default=None, help="The serialized TensorRT engine")
    parser.add_argument("-i", "--input", default=None, help="Path to the image or directory to process")
    parser.add_argument("-o", "--output", default=None, help="Directory where to save the visualization results")
    parser.add_argument("-l", "--labels", default="./labels_coco.txt", help="File to use for reading the class labels "
                                                                            "from, default: ./labels_coco.txt")
    parser.add_argument("-t", "--nms_threshold", type=float, help="Override the score threshold for the NMS operation, "
                                                                  "if higher than the threshold in the engine.")
    args = parser.parse_args()
    args.engine = '/home/nvidia/Desktop/FCOSR/model/epoch_36_16_lite_nx_74.trt'
    args.input = '/home/nvidia/DOTA_TEST/images/'
    args.output = 'result'
    args.labels = 'labels.txt'
    args.nms_threshold = 0.1

    if not all([args.engine, args.input, args.output]):
        parser.print_help()
        print("\nThese arguments are required: --engine  --input and --output")
        sys.exit(1)
    main(args)

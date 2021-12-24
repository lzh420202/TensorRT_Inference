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
from zmq_wrappers import custom_client
import cv2
import os


if __name__ == "__main__":
    save_dirs = r'D:\Code'
    infer_client = custom_client('192.168.2.38', 10000, True)
    image = cv2.imread(r'F:\BaiduNetdiskDownload\test\BIG\P0031.png')
    infer_client.sendData(dict(image=image, name='P0031.png'))
    result = infer_client.draw_result(image)
    cv2.imwrite(os.path.join(save_dirs, 'P0031.png'), result)

    image = cv2.imread(r'F:\BaiduNetdiskDownload\test\BIG\P0112.png')
    infer_client.sendData(dict(image=image, name='P0112.png'))
    result = infer_client.draw_result(image)
    cv2.imwrite(os.path.join(save_dirs, 'P0112.png'), result)



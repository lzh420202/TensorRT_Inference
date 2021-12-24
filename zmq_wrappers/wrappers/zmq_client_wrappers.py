import time

from ..base.zmq_base import (zmq_client_base, zmq_client_complex_base)
from queue import Queue
from ..hooks.message_hooks import (sendDataHooks, sendMultipartDataHooks, sendMultipartDataComplexHooks)
from ..hooks.client_function import client_payload
from threading import Thread
import copy


class zmq_data_client(zmq_client_base):
    def __init__(self, dst_ip: str, port: int, input_queue: Queue):
        super(zmq_data_client, self).__init__(dst_ip, port, input_queue, sendDataHooks, client_payload)


class zmq_multipart_data_client(zmq_client_base):
    def __init__(self, dst_ip: str, port: int, input_queue: Queue, progressbar=None):
        super(zmq_multipart_data_client, self).__init__(dst_ip, port, input_queue, sendMultipartDataHooks, client_payload)
        self.progressbar = progressbar


class zmq_multipart_data_complex_client(zmq_client_complex_base):
    def __init__(self, dst_ip: str, port: int, input_queue: Queue, output_queue: Queue, progressbar=None):
        super(zmq_multipart_data_complex_client, self).__init__(dst_ip, port, input_queue, output_queue, sendMultipartDataComplexHooks)
        self.progressbar = progressbar


class zmq_client(zmq_client_base):
    def __init__(self, dst_ip: str, port: int, input_queue: Queue, message_callback=None, function_callback=None, progressbar=None):
        if message_callback is None:
            message_callback = sendMultipartDataHooks
        if function_callback is None:
            function_callback = client_payload
        super(zmq_client, self).__init__(dst_ip, port, input_queue, message_callback, function_callback)
        self.progressbar = progressbar


class monitorThread(Thread):
    def __init__(self, variate, interval_time=0.1):
        Thread.__init__(self)
        self.variate = variate
        self.interval_time = interval_time
        self.bar_width = 50
        self.bar_offset = 9
        self.space = '\033[42m{}\033[0m'

    def generate_progressbar_base(self, info, with_describe):
        rate = info['current'] / info['total']
        bar_length = min(self.bar_width, round(rate * self.bar_width))
        process = self.space.format(''.ljust(bar_length))
        process = process.ljust(self.bar_width + self.bar_offset)
        if with_describe:
            return f"\r{info['description']} {rate: 5.0%} {process} "
        else:
            return f"\r{rate: 5.0%} {process} "


    def generate_transfer_progressbar(self, info, with_describe=False):
        bar_base = self.generate_progressbar_base(info, with_describe)
        speed = info['current'] / ((1 << 20) * info['used_time'])
        eta = (info['total'] - info['current']) * info['used_time'] / info['current']
        bar = bar_base + f"{info['current'] / (1 << 20):.1f}/{info['total'] / (1 << 20):.1f}MB " \
                         f"{speed: 5.2f}MB/s " \
                         f"{info['used_time']: .1f}s " \
                         f"ETA: {eta: .1f}s"
        return bar

    def generate_detect_progressbar(self, info, with_describe=False):
        bar_base = self.generate_progressbar_base(info, with_describe)
        fps = info['current'] / info['used_time']
        eta = (info['total'] - info['current']) * info['used_time'] / info['current']
        bar = bar_base + f"{info['current']}/{info['total']} " \
                         f"{fps: 5.2f}FPS " \
                         f"{info['used_time']: .1f}s" \
                         f"ETA: {eta: .1f}s"
        return bar

    def run(self) -> None:
        while True:
            if self.variate:
                if self.variate['run']:
                    stat_copy = copy.deepcopy(self.variate)
                    if stat_copy['used_time'] == 0:
                        continue
                    if stat_copy['type'] == 'transfer':
                        bar = self.generate_transfer_progressbar(stat_copy, True)
                    elif stat_copy['type'] == 'detect':
                        bar = self.generate_detect_progressbar(stat_copy, True)
                    else:
                        raise AttributeError(f"Unsupport progress bar type: {stat_copy['type']}")
                    print(bar, end='', flush=True)
                    if stat_copy['done']:
                        self.variate.update(dict(current=0, total=0, used_time=0, done=False, run=False))
                        print('')
            time.sleep(self.interval_time)


from ..base.zmq_base import (zmq_server_base, zmq_server_complex_base)
from ..hooks.message_hooks import (recvDataHooks, recvMultipartDataHooks, recvMultipartDataComplexHooks)
from ..hooks.server_function import server_payload
from multiprocessing import Queue

class zmq_data_server(zmq_server_base):
    def __init__(self, port: int):
        super(zmq_data_server, self).__init__(port, recvDataHooks, server_payload)


class zmq_multipart_data_server(zmq_server_base):
    def __init__(self, port: int):
        super(zmq_multipart_data_server, self).__init__(port, recvMultipartDataHooks, server_payload)


class zmq_server(zmq_server_base):
    def __init__(self, port: int, message_hooks=None, payload_hooks=None):
        if message_hooks is None:
            message_hooks = recvMultipartDataHooks
        if payload_hooks is None:
            payload_hooks = server_payload
        super(zmq_server, self).__init__(port, message_hooks, payload_hooks)

class zmq_multipart_data_complex_server(zmq_server_complex_base):
    def __init__(self, port: int, data_queue: Queue, result_queue: Queue, progressbar_queue: Queue=None, with_progressbar=False):
        super(zmq_multipart_data_complex_server, self).__init__(port, recvMultipartDataComplexHooks, data_queue, result_queue, progressbar_queue, with_progressbar)
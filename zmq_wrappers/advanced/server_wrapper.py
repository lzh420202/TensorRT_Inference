from ..wrappers.zmq_server_wrappers import (zmq_multipart_data_server, zmq_multipart_data_complex_server)
from multiprocessing import Queue

class custom_server():
    def __init__(self, port: int, data_queue: Queue, result_queue: Queue, progressbar_queue: Queue=None, with_progressbar=False):
        self.server = zmq_multipart_data_complex_server(port, data_queue, result_queue, progressbar_queue, with_progressbar)
        self.server.start()
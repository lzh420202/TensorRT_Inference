import zmq
from threading import Thread
from queue import Queue
from multiprocessing import Queue as mQueue
from abc import abstractmethod

class _zmq_client_base(Thread):
    def __init__(self, dst_ip: str, port: int, input_queue: Queue, message_callback):
        Thread.__init__(self)
        if message_callback is None:
            raise ValueError('message callback can not be None.')
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.set(zmq.LINGER, 0)
        self.addr = f"tcp://{dst_ip}:{port}"
        self.message_callback = message_callback
        self.input_queue = input_queue
        self.progressbar = None

    def sendMessage(self, message: dict):
        self.socket.connect(self.addr)
        result = self.message_callback(self.socket, message, self.progressbar)
        self.socket.disconnect(self.addr)
        return result

    @abstractmethod
    def run(self):
        raise NotImplementedError


class zmq_client_base(_zmq_client_base):
    def __init__(self, dst_ip: str, port: int, input_queue: Queue, message_callback, function_callback):
        super(zmq_client_base, self).__init__(dst_ip, port, input_queue, message_callback)
        if function_callback is None:
            raise ValueError('function callback can not be None.')
        self.function_callback = function_callback

    def run(self):
        while True:
            message = self.input_queue.get()
            if message:
                result = self.sendMessage(message)
                self.function_callback(result)
            else:
                break
        self.context.destroy()


class zmq_client_complex_base(_zmq_client_base):
    def __init__(self, dst_ip: str, port: int, input_queue: Queue, output_queue: Queue, message_callback):
        super(zmq_client_complex_base, self).__init__(dst_ip, port, input_queue, message_callback)
        self.output_queue = output_queue

    def run(self):
        while True:
            message = self.input_queue.get()
            if message:
                result = self.sendMessage(message)
                self.output_queue.put(result)
            else:
                break
        self.context.destroy()


class _zmq_server_base(Thread):
    def __init__(self, port: int, message_callback):
        Thread.__init__(self)
        if message_callback is None:
            raise ValueError('message callback can not be None.')
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.message_callback = message_callback

    @abstractmethod
    def run(self):
        raise NotImplementedError


class zmq_server_base(_zmq_server_base):
    def __init__(self, port: int, message_callback, function_callback):
        super(zmq_server_base, self).__init__(port, message_callback)
        self.function_callback = function_callback

    def close_server(self):
        self.socket.close()
        self.context.destroy()

    def listening(self):
        self.message_callback(self.socket, self.function_callback)

    def run(self):
        self.listening()


class zmq_server_complex_base(_zmq_server_base):
    def __init__(self, port: int,
                 message_callback,
                 data_queue: mQueue,
                 result_queue: mQueue,
                 progressbar_queue: mQueue = None,
                 with_progressbar=False):
        super(zmq_server_complex_base, self).__init__(port, message_callback)
        self.data_queue = data_queue
        self.result_queue = result_queue
        self.progressbar_queue = progressbar_queue
        self.with_progressbar = with_progressbar

    def close_server(self):
        self.socket.close()
        self.context.destroy()

    def listening(self):
        self.message_callback(self.socket, self.data_queue, self.result_queue, self.progressbar_queue, self.with_progressbar)

    def run(self):
        self.listening()
from queue import Queue
from ..wrappers.zmq_client_wrappers import zmq_multipart_data_client, zmq_multipart_data_complex_client, monitorThread
from ..hooks.client_function import draw_zmq_result

class custom_client():
    def __init__(self, ip, port, with_monitor=False):
        self.input_queue = Queue(10)
        self.output_queue = Queue(10)
        if with_monitor:
            self.process_info = dict(type='transfer', current=0, total=0, used_time=0, description='network transfer', done=False, run=False)
        else:
            self.process_info = None
        self.client = zmq_multipart_data_complex_client(ip, port, self.input_queue, self.output_queue, self.process_info)
        self.client.start()
        if with_monitor:
            self.monitor = monitorThread(self.process_info, 0.1)
            self.monitor.start()
        else:
            self.monitor = None

    def sendData(self, data):
        self.input_queue.put(data)

    def testServer(self):
        self.input_queue.put(dict(TEST=True))

    def draw_result(self, image):
        result = self.output_queue.get()
        img = draw_zmq_result(image, result['objects'])
        return img



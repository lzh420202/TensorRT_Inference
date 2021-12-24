import zmq
# from queue import Queue
from multiprocessing import Queue
import time
import pickle
import math


def sendDataHooks(socket: zmq.Socket, message: dict, progress_bar_info=None):
    t = time.time()
    socket.send_pyobj(message)
    result = socket.recv_pyobj()
    return result


def recvDataHooks(socket: zmq.Socket, callback):
    while True:
        message = socket.recv_pyobj()
        if message:
            reply = callback(message)
        else:
            reply = None
        socket.send_pyobj(reply)


def sendMultipartDataHooks(socket: zmq.Socket, message: dict, progress_bar_info=None):
    if message.get('TEST') is not None:
        socket.send_string('START')
        _ = socket.recv()
        socket.send_string('1')
        _ = socket.recv()
        return dict(TEST='SUCCESS')
    else:
        message_bytes = pickle.dumps(message)
        split_n = 100
        min_block_size = 1 << 15 # 32kb
        max_block_size = 1 << 20 # 1024kb
        total = len(message_bytes)
        block_size = math.ceil(total / split_n)
        if block_size < min_block_size:
            split_n = max(1, math.floor(total / min_block_size))
            block_size = math.ceil(total / split_n)
        elif block_size > max_block_size:
            split_n = math.ceil(total / max_block_size)
            block_size = math.ceil(total / split_n)

        if progress_bar_info:
            assert isinstance(progress_bar_info, dict)
            progress_bar_info.update(dict(type='transfer',
                                          current=0,
                                          total=total,
                                          used_time=0,
                                          description='network transfer',
                                          done=False,
                                          run=False))
        t = time.time()
        socket.send_string('START')
        _ = socket.recv()
        socket.send_string(f'{split_n}')
        _ = socket.recv()
        for i in range(split_n):
            msg = message_bytes[i*block_size:(i+1)*block_size]
            socket.send(msg)
            if progress_bar_info:
                current = min(block_size * (i + 1), total)
                progress_bar_info.update(dict(current=current, used_time=time.time() - t, run=True))
            _ = socket.recv()
        if progress_bar_info:
            progress_bar_info.update(dict(current=total, used_time=time.time() - t, done=True))
        socket.send_string('done')
        result = socket.recv_pyobj()
        return result


def recvMultipartDataHooks(socket: zmq.Socket, callback):
    renew = False
    while True:
        msg = b'ok'
        parts = []
        if renew:
            renew = False
        else:
            assert socket.recv_string() == 'START'
            socket.send(msg)

        n_str = socket.recv_string()
        n = int(n_str)
        socket.send(msg)
        for i in range(n):
            msg_recv = socket.recv()
            if len(msg_recv) == 5:
                if msg_recv == b'START':
                    renew = True
                    socket.send(msg)
                    break
            parts.append(msg_recv)
            socket.send(msg)
        if renew:
            continue
        _ = socket.recv_string()
        message = pickle.loads(b''.join(parts))
        if message:
            reply = callback(message)
        else:
            reply = None

        socket.send_pyobj(reply)


def recvMultipartDataComplexHooks(socket: zmq.Socket, data_queue: Queue, result_queue: Queue, progressbar_queue: Queue, with_progressbar=False):
    renew = False
    while True:
        msg = b'ok'
        parts = []
        if renew:
            renew = False
        else:
            assert socket.recv_string() == 'START'
            socket.send(msg)

        n_str = socket.recv_string()
        n = int(n_str)
        socket.send(msg)
        for i in range(n):
            msg_recv = socket.recv()
            if len(msg_recv) == 5:
                if msg_recv == b'START':
                    renew = True
                    socket.send(msg)
                    break
            parts.append(msg_recv)
            socket.send(msg)
        if renew:
            continue
        _ = socket.recv_string()
        message = pickle.loads(b''.join(parts))

        data_queue.put(message)
        socket.send_pyobj(dict(with_progressbar=True))
        _ = socket.recv()
        if with_progressbar:
            while True:
                progressbar_info = progressbar_queue.get()
                flag = (progressbar_info['current'] == progressbar_info['total']) and (progressbar_info['current'] > 0)
                if flag:
                    progressbar_info.update(dict(done=True))
                else:
                    progressbar_info.update(dict(done=False))
                socket.send_pyobj(progressbar_info)
                _ = socket.recv()
                if flag:
                    break
        result = result_queue.get()
        socket.send_pyobj(result)


def sendMultipartDataComplexHooks(socket: zmq.Socket, message: dict, progress_bar_info=None):
    if message.get('TEST') is not None:
        socket.send_string('START')
        _ = socket.recv()
        socket.send_string('1')
        _ = socket.recv()
        return dict(TEST='SUCCESS')
    else:
        message_bytes = pickle.dumps(message)
        split_n = 100
        min_block_size = 1 << 15 # 32kb
        max_block_size = 1 << 20 # 1024kb
        total = len(message_bytes)
        block_size = math.ceil(total / split_n)
        if block_size < min_block_size:
            split_n = max(1, math.floor(total / min_block_size))
            block_size = math.ceil(total / split_n)
        elif block_size > max_block_size:
            split_n = math.ceil(total / max_block_size)
            block_size = math.ceil(total / split_n)

        if progress_bar_info:
            assert isinstance(progress_bar_info, dict)
            progress_bar_info.update(dict(type='transfer',
                                          current=0,
                                          total=total,
                                          used_time=0,
                                          description='network transfer',
                                          done=False,
                                          run=False))
        t = time.time()
        socket.send_string('START')
        _ = socket.recv()
        socket.send_string(f'{split_n}')
        _ = socket.recv()
        for i in range(split_n):
            msg = message_bytes[i*block_size:(i+1)*block_size]
            socket.send(msg)
            if progress_bar_info:
                current = min(block_size * (i + 1), total)
                progress_bar_info.update(dict(current=current, used_time=time.time() - t, run=True))
            _ = socket.recv()
        if progress_bar_info:
            progress_bar_info.update(dict(current=total, used_time=time.time() - t))
        socket.send_string('done')
        progress_bar_flag = socket.recv_pyobj()
        if progress_bar_flag['with_progressbar']:
            socket.send(b'0')
            while True:
                progress_bar_msg = socket.recv_pyobj()
                if progress_bar_info:
                    progress_bar_info.update(progress_bar_msg)
                socket.send(b'0')
                if progress_bar_msg['done']:
                    break
        result = socket.recv_pyobj()
        return result
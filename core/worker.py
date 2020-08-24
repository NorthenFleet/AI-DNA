import threading
import struct
import json
import numpy as np
import time
import hashlib
from ai_company.core import logger
from ai_company.core.debug import get_transmission_logger


class Connection: 
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.int) or isinstance(obj, np.int16) or isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            if isinstance(obj) == np.ndarray:
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    def __init__(self, conn, timeout=100):
        conn.settimeout(timeout)
        self.conn = conn
        self.cache = b''

    def send_json(self, data):
        raw = bytes(json.dumps(data, cls=Connection.NumpyEncoder), encoding="utf-8")
        header = struct.pack('>I', len(raw))
        self.conn.sendall(header)
        self.conn.sendall(raw)

    def recv_json(self):
        length = 0
        raw = self.cache

        while True:
            if len(raw) >= 4:
                length = struct.unpack('>I', raw[:4])[0]
                raw = raw[4:]
                break
            raw += self.conn.recv(4)
        while True:
            raw += self.conn.recv(1024)
            if len(raw) >= length:
                self.cache = raw[length:]
                break
        return json.loads(str(raw[:length], encoding="utf-8"))

    def close(self):
        self.conn.close()


class ThreadController:
    def __init__(self):
        self.threads_info = []

    def create_work(self, addr, conn, handler_factory):
        worker = Worker(addr, conn, handler_factory)
        thread = threading.Thread(target=worker.work, args=())
        self.threads_info.append((worker, thread))
        thread.start()

    def join(self):
        for w, t in self.threads_info:
            t.join()

    def request_stop(self):
        for w, t in self.threads_info:
            w.stop()


STOP = -1
INITIALIZE = 0
RESET = 1
STEP = 2


class Worker:
    def __init__(self, addr, conn, handler_factory, record_transmission=True):
        self.addr = addr
        self.conn = Connection(conn)
        self.handler = handler_factory()
        self.loop_counter = 0
        self._running = True

        code = f'{time.time()}:{addr[0]}:{addr[1]}'
        sha1 = hashlib.sha1(bytes(code, encoding="utf-8"))
        self.worker_id = sha1.hexdigest()[:6]

        self.record_transmission = record_transmission
        if self.record_transmission:
            self.transmission_logger = get_transmission_logger(self.worker_id)

    def _loop(self):
        logger.info(f"<{self.worker_id}> Worker正在启动")

        while self._running:
            recv = self.conn.recv_json()

            if self.record_transmission:
                self.transmission_logger.debug(json.dumps(recv))
                
            if recv['type'] == INITIALIZE:
                self.handler.handle_init_data(recv['data'])
            elif recv['type'] == RESET:
                self.handler.handle_reset_data()
            elif recv['type'] == STEP:
                result = self.handler.handle_step_data(recv['data'])
                self.conn.send_json(result)
            elif recv['type'] == STOP:
                break
            self.loop_counter += 1

    def work(self):
        try:
            self._loop()
        except ConnectionResetError as e:
            logger.error(f"<{self.worker_id}> 与客户端的连接已断开，连接信息: {self.addr}")
        except TimeoutError as e:
            logger.error(f"<{self.worker_id}> 与客户端的连接超时，连接信息: {self.addr}")
        except Exception as e:
            logger.exception(f"<{self.worker_id}> Worker内部发生异常，循环计数: {self.loop_counter}")
        finally:
            logger.info(f"<{self.worker_id}> Worker已停止运行")
            self.handler.handle_stop()
            self.conn.close()

    def stop(self):
        self._running = False
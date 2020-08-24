import socket

from .worker import ThreadController, Worker
from ai_company.core import logger

class AIServer:
    def __init__(self, host, port, handler_factory, multi_worker=False):
        self.host = host
        self.port = port
        self.multi_worker = multi_worker
        self.handler_factory = handler_factory
        self.thread_controller = None

    def run(self):
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(10)
        logger.info(f"准备接受远程连接，监听地址{self.host}:{self.port}")
        try:
            if self.multi_worker:
                self.thread_controller = ThreadController()
                while True:
                    conn, addr = listener.accept()
                    self.thread_controller.create_work(addr, conn, self.handler_factory)
            else:
                conn, addr = listener.accept()
                worker = Worker(addr, conn, self.handler_factory)
                worker.work()
        except KeyboardInterrupt as e:
            logger.info("AIServer已被用户终止")
            if self.thread_controller:
                self.thread_controller.request_stop()
            else:
                worker.stop()
        finally:
            logger.info("AIServer正在尝试退出，请耐心等待...")
            listener.close()
            if self.thread_controller:
                self.thread_controller.join()



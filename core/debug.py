
import logging
import time
import json
import os
import pdb
from ai_company.core import TRANSMISSIN_LOG_DIR

def get_transmission_logger(worker_id):
    logger_name = 'debug'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter('%(message)s')

    filename = os.path.join(TRANSMISSIN_LOG_DIR, f'{worker_id}.log')
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(file_formatter)
    logger.addHandler(handler)

    return logger

STOP = -1
INITIALIZE = 0
RESET = 1
STEP = 2
class DebugWorker:
    def __init__(self, worker_id: str, stop_at: int, handler_factory):
        self.worker_id = worker_id
        self.handler = handler_factory()
        self.filename = os.path.join(TRANSMISSIN_LOG_DIR, f'{worker_id}.log')
        self.loop_counter = 0
        self.stop_at = stop_at

    def _loop(self):
        print(f"<{self.worker_id}> DebugWorker正在启动")

        with open(self.filename) as f:
            while True:
                recv = f.readline().strip('\n')
                if not recv:
                    break
                recv = json.loads(recv)

                if self.loop_counter == self.stop_at:
                    print('>>> 进入调试模式 <<<')
                    pdb.set_trace()

                if recv['type'] == INITIALIZE:
                    self.handler.handle_init_data(recv['data'])
                elif recv['type'] == RESET:
                    self.handler.handle_reset_data()
                elif recv['type'] == STEP:
                    result = self.handler.handle_step_data(recv['data'])
                    # self.conn.send_json(result)
                elif recv['type'] == STOP:
                    break
                self.loop_counter += 1

    def work(self):
        self._loop()

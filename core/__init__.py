import os

LOG_DIR = 'log'
CONFIG_DIR = 'config'

SERVER_LOG_DIR = os.path.join(LOG_DIR, 'server')
TRANSMISSIN_LOG_DIR = os.path.join(LOG_DIR, 'worker')

SERVER_INI = os.path.join(CONFIG_DIR, 'server.ini')


def _set_dir():
    import os
    import shutil

    if not os.path.exists(LOG_DIR):
       os.mkdir(LOG_DIR)
    if not os.path.exists(CONFIG_DIR):
        os.mkdir(CONFIG_DIR)
    if not os.path.exists(TRANSMISSIN_LOG_DIR):
        os.mkdir(TRANSMISSIN_LOG_DIR)
    if not os.path.exists(SERVER_LOG_DIR):
        os.mkdir(SERVER_LOG_DIR)
    if not os.path.exists(SERVER_INI):
        SAMPLE = os.path.join(CONFIG_DIR, "server.ini.sample")
        shutil.copyfile(SAMPLE, SERVER_INI)
        print(f"{SERVER_INI}不存在，已生成默认server配置文件，请修改后重新启动")
        exit()


def _clean_old_log():
    pass


def _get_logger():
    import logging
    import os
    import time
    import logging.handlers

    file_formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] <%(threadName)s> %(message)s (%(filename)s:%(lineno)s)', datefmt="%Y-%m-%d %H:%M:%S")
    console_formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] <%(threadName)s> %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    logger_name = 'core'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # filename = os.path.join(SERVER_LOG_DIR, f'debug.log')
    # handler = logging.handlers.RotatingFileHandler(filename, maxBytes=1024*1024*5, backupCount=40)
    # handler.setLevel(logging.DEBUG)
    # handler.setFormatter(file_formatter)
    # logger.addHandler(handler)

    error_filename = os.path.join(SERVER_LOG_DIR, f'error.log')
    error_handler = logging.FileHandler(error_filename)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(console_formatter)
    logger.addHandler(console)

    return logger


_set_dir()
_clean_old_log()
logger = _get_logger()

occupy_scenario_1231 = False

from .handler import BaseHandler
from .server import AIServer

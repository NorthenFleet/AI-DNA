import sys
import configparser

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from ai_company.core import AIServer, logger
from ai_company.handler import TransmissionTest, ParallelTrainer
from ai_company.algorithm import ParallelPPO

if __name__ == "__main__":
    save_net = False
    save_model_num = '2010211129_red_new'
    load_net = False
    load_model_num = '2010211129_red_new'
    save_interval_steps = 10000
    is_Train = True
    max_map_size_x = 77*2
    max_map_size_y = 92*2
    step_num = 2
    move_space = sum(range(step_num + 1)) * 6
    map_chanel = 1
    input_entity_size = 384
    batch = 128
    IS_MULTI_PROCESSING = True

    logger.info("*** SERVER RUNNING ***")
    cf = configparser.ConfigParser()
    cf.read("./config/server.ini")

    host = cf.get('Server', 'host')
    port = cf.getint('Server', 'port')
    transmission_test = cf.getboolean('Handler', 'transmission_test')

    ppo = None

    if transmission_test:
        func = lambda: TransmissionTest()
    else:
        ppo = ParallelPPO(max_map_size_x=max_map_size_x, max_map_size_y=max_map_size_y, move_space=move_space,
                          map_chanel=map_chanel, input_scale_size=input_entity_size, save_net=save_net,
                          load_net=load_net, save_steps=save_interval_steps, is_train=is_Train, batch=batch,
                          save_model_num=save_model_num, load_model_num=load_model_num, use_gpu=False)
        func = lambda: ParallelTrainer(step_num, ppo)

    server = AIServer(host, port, func, multi_worker=IS_MULTI_PROCESSING)
    server.run()
    
    logger.info("*** SERVER EXIT ***")

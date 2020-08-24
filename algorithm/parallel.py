import threading
import time
import numpy as np
from .PPO import PPO
from ai_company.core import logger
from ai_company.data_process.game_data import ValidActionInterface

class ParallelPPO:
    def __init__(self, max_map_size_x, max_map_size_y, move_space, map_chanel, input_scale_size, save_net, load_net,
                 save_steps, is_train, batch, save_model_num, load_model_num, use_gpu=True):
        assert(threading.current_thread() == threading.main_thread())

        # 参数输出
        import inspect
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        arg_list = []
        for i in args:
            if i == 'self':
                continue
            arg_list.append(f'{i}={values[i]}')
        logger.debug(f'神经网络初始化参数：{", ".join(arg_list)}')

        self.save_net = save_net
        self.save_model_num = save_model_num
        self.save_steps = save_steps

        # 地图数据初始化
        self.actions_space = 4964
        # self.map_chanel = 50
        self.ppo = PPO(self.actions_space, max_map_size_x, max_map_size_y, move_space, map_chanel, input_scale_size,
                       load_net, is_train, batch, load_model_num, use_gpu)
        self._lock = threading.Lock()
        self.buffer_spatial_actor = []
        self.buffer_spatial_critic = []
        self.buffer_entity_actor = []
        self.buffer_entity_critic = []
        self.buffer_scalar_state = []
        self.buffer_action_type = []
        self.buffer_my_id = []
        self.buffer_obj_id = []
        self.buffer_r = []
        self.buffer_move = []
        # self.buffer_x = []
        # self.buffer_y = []
        self.buffer_move_type = []
        self.BATCH = batch
        self.GAMMA = 0.9
        self.METHOD = [
            dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
            dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
        ][1]  # choose the method for optimization

        self.current_step = 0

    def choose_action_discrete(self, spatial_actor, spatial_critic, entity_actor, entity_critic, scalar_state,
                               interface: ValidActionInterface):
        result = self.ppo.choose_action(spatial_actor, entity_actor, spatial_critic, entity_critic,
                                        scalar_state, interface)
        result["thread_id"] = threading.currentThread().ident
        return result

    # 不知道在哪里用
    # def dispose_last_data(self, scenario, spatial_actor, spatial_critic, entity_actor, entity_critic, scalar_state, r, result):
    #     with self._lock:
    #         self.ep_r += r
    #         self.buffer_spatial_actor.append(spatial_actor)
    #         self.buffer_spatial_critic.append(spatial_critic)
    #         self.buffer_entity_actor.append(entity_actor)
    #         self.buffer_entity_critic.append(entity_critic)
    #         self.buffer_scalar_state.append(scalar_state)
    #         self.buffer_action_type.append(result["action_type"])
    #         self.buffer_my_id.append(result["my_id"])
    #         self.buffer_obj_id.append( result["target_id"])
    #         self.buffer_move.append(result["move"])
    #         # self.buffer_x.append(result["x"])
    #         # self.buffer_y.append(result["y"])
    #         self.buffer_move_type.append(result["move_type"])
    #         self.buffer_r.append(r) # normalize reward, find to be useful
    #
    #         # 更新网络
    #         if (self.current_step + 1) % (self.BATCH) == 0:
    #             logger.info(f'更新网络')
    #             v_s_ = self.ppo.get_v(spatial_critic, entity_critic)
    #             discounted_r = []
    #             for r in self.buffer_r[::-1]:
    #                 v_s_ = r + self.GAMMA * v_s_
    #                 discounted_r.append(v_s_)
    #             discounted_r.reverse()
    #
    #             b_spatial_actor = np.vstack(self.buffer_spatial_actor)
    #             b_spatial_critic = np.vstack(self.buffer_spatial_critic)
    #             b_entity_actor = np.vstack(self.buffer_entity_actor)
    #             b_entity_critic = np.vstack(self.buffer_entity_critic)
    #             b_scalar_state = np.vstack(self.buffer_scalar_state)
    #             ba = np.vstack(self.buffer_action_type)
    #             bmy_id = np.vstack(self.buffer_my_id)
    #             bobj_id = np.vstack(self.buffer_obj_id)
    #             bmove = np.vstack(self.buffer_move)
    #             # bx = np.vstack(self.buffer_x)
    #             # by = np.vstack(self.buffer_y)
    #             bmove_type = np.vstack(self.buffer_move_type)
    #             # br = np.array(discounted_r[-32:])[:, np.newaxis]
    #             br = np.array(discounted_r)
    #
    #             self.buffer_spatial_actor, self.buffer_entity_actor = [], []
    #             self.buffer_spatial_critic, self.buffer_entity_critic = [], []
    #             self.buffer_scalar_state = 0,
    #             self.buffer_action_type, self.buffer_r = [], []
    #             self.buffer_my_id, self.buffer_obj_id = [], []
    #             self.buffer_x, self.buffer_y = [], []
    #             self.buffer_move_type = []
    #
    #             begining_step = self.current_step - self.BATCH
    #             # fixme
    #             self.ppo.store_transition(begining_step, b_spatial_actor, b_spatial_critic, b_entity_actor, b_entity_critic,
    #                             b_scalar_state, ba, bmy_id, bobj_id, bmove, bmove_type, np.squeeze(br, axis=2))
    #
    #         if (self.current_step) % (self.save_steps - 1) == 0 and self.save_net:
    #             logger.info("保存网络")
    #             # if self.save_model_num == 10:
    #             #     self.save_model_num = 0
    #             # self.ppo.save_model(self.current_step, self.save_model_num)
    #             # self.save_model_num += 1
    #             self.ppo.save_model(self.current_step, self.save_model_num)
    #
    #         if self.current_step == 0:
    #             self.all_ep_r.append(self.ep_r)
    #         else:
    #             self.all_ep_r.append(self.all_ep_r[-1] * 0.9 + self.ep_r * 0.1)
    #
    #         logger.info('({}) current_step: {}, ep_r: {}'.format(scenario, self.current_step, self.ep_r))
    #         self.current_step += 1



    def store_transition(self, buffer_spatial_actor,
                     buffer_spatial_critic,
                     buffer_entity_actor,  
                     buffer_entity_critic, 
                     buffer_scalar_state,  
                     buffer_action_type,   
                     buffer_my_id,         
                     buffer_obj_id,        
                     buffer_move,
                     buffer_move_type,
                     entity_critic,
                     scalar_state,
                     buffer_r,
                     buffer_basic):
        with self._lock:
            # 更新网络
            v_s_ = self.ppo.get_v(entity_critic, scalar_state)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + self.GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            v_s_basic = self.ppo.get_v(entity_critic, scalar_state)
            discounted_basic = []
            for basic in buffer_basic[::-1]:
                v_s_basic = basic + self.GAMMA * v_s_basic
                discounted_basic.append(v_s_basic)
            discounted_basic.reverse()

            b_spatial_actor = np.vstack(buffer_spatial_actor)
            b_spatial_critic = np.vstack(buffer_spatial_critic)
            b_entity_actor = np.vstack(buffer_entity_actor)
            b_entity_critic = np.vstack(buffer_entity_critic)
            b_scalar_state = np.vstack(buffer_scalar_state)
            ba = np.vstack(buffer_action_type)
            bmy_id = np.vstack(buffer_my_id)
            bobj_id = np.vstack(buffer_obj_id)
            bmove = np.vstack(buffer_move)
            bmove_type = np.vstack(buffer_move_type)
            # br = np.array(discounted_r[-32:])[:, np.newaxis]
            br = np.array(discounted_r)
            bbasic = np.array(discounted_basic)

            begining_step = self.current_step - self.BATCH
            # fixme
            self.ppo.store_transition(begining_step, b_spatial_actor, b_spatial_critic, b_entity_actor, b_entity_critic,
                b_scalar_state, ba, bmy_id, bobj_id, bmove, bmove_type, np.squeeze(br, axis=2), np.squeeze(bbasic, axis=2))



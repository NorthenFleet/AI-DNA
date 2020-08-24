import sys

sys.path.append("../")
import os

from ai_company.net.ACNet import ACNet
import numpy as np
import matplotlib.pyplot as plt
import math

from ai_company.data_process.game_data import ValidActionInterface
from ai_company.core import logger
# from ai_company.algorithm.PPO_MOVE import PPO_MOVE
from ai_company.move.DPPO_move import PPONet
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
if type(tf.contrib) != type(tf): tf.contrib._warning = None

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

class PPO(object):
    def __init__(self, actions_space, max_map_size_x, max_map_size_y, move, map_chanel, input_entity_size, load_net, is_train,
                 batch, load_model_num, use_gpu):
        self._device_str1 = '/gpu:0' if use_gpu else '/cpu:0'
        self._device_str2 = '/gpu:1' if use_gpu else '/cpu:0'
        logger.info(f"PPO初始化")
        logger.info(f"使用设备：{self._device_str1}")
        logger.info(f"使用设备：{self._device_str2}")
        self.A_LR = 0.0001
        self.C_LR = 0.0002
        self.batch = batch
        self.A_UPDATE_STEPS = 10
        self.C_UPDATE_STEPS = 10
        self.METHOD = [
            dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
            dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
        ][1]  # choose the method for optimization

        # 设置Sess运算环境
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 使用GPU显存的比率
        config.gpu_options.allow_growth = True  # 按需求使用GPU
        config.log_device_placement = False
        # config.device_count = {'cpu': 0}
        # config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.C_DIM = 1

        # 添加的参数
        self.memory_entity_critic, self.memory_scalar_state, self.memory_r = None, None, None
        self.memory_action, self.memory_my_id, self.memory_obj_id, self.memory_move = None, None, None, None
        self.actions_space = actions_space
        self.max_map_size_x = max_map_size_x
        self.max_map_size_y = max_map_size_y
        self.move = move
        self.map_chanel = map_chanel
        self.action_type_size = 14
        self.my_id_size = 34
        self.obj_id_size = 34
        self.input_entity_size = input_entity_size
        self.move_type = 10
        self.input_scalar_size = 20
        self.update_interval = 1
        self.lstm_batch_size = 1
        self.eps = 0.000001

        # define the input of the state
        # self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], "state")

        # saving and loading the network
        self.model_path = 'PredictAI_model/'
        self.load_model_num = load_model_num
        self.save_interval = 0

        self.adv_list = []
        self.critic_list = []
        self.fig = plt.figure()
        self.display_loss_interval = 10
        self.dislpay_loss_counter = 0

        # # define the ppo_move
        # self.ppo_move = PPO_MOVE(batch, step_num=2)
        self.ppo_move = PPONet(step_num=3)
        self.is_ppo_move = False
        # self.loss_display = fig.add_subplot(1, 1, 1)
        # self.loss_display.scatter(self.current_step, self.loss)
        plt.ion()  # 本次运行请注释，全局运行不要注释
        plt.show()

        self.get_on_flage = False

        # actor & critic
        logger.info(f'初始化神经网络')
        with tf.variable_scope('ACnet'):
            self.ACNet = ACNet(self.max_map_size_y, self.max_map_size_x, self.map_chanel, self.lstm_batch_size,
                               self.actions_space, self.action_type_size, self.my_id_size, self.obj_id_size,
                               self.move, self.A_LR, use_gpu)
            # self.tf_spatial_critic = tf.placeholder(tf.float32, [None, self.max_map_size_y, self.max_map_size_x,
            #                                                      self.map_chanel], name='inputs')
            self.tf_entity_critic = tf.placeholder(tf.float32, [None, self.input_entity_size], name='inputs_entity')
            self.tf_scalar_state = tf.placeholder(tf.float32, [None, self.input_scalar_size], name='inputs_scale')
            self.tf_spatial_critic = tf.placeholder(tf.float32,
                                                   [None, self.max_map_size_y, self.max_map_size_x, self.map_chanel],
                                                   name='actor_inputs')
            # self.tf_entity_actor = tf.placeholder(tf.float32, [None, self.input_entity_size], name='actor_inputs_scale')

            self.tfdc_r = tf.placeholder(tf.float32, [None, self.C_DIM], 'discounted_r')
            self.ANet, self.old_ANet, self.critic = self.ACNet.build_ACNet(self.tf_entity_critic, self.tf_scalar_state,
                                                                           True)
            self.advantage = self.tfdc_r - self.critic
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            with tf.device('/cpu:0'):
                tf.summary.scalar('closs', self.closs)  # tensorflow >= 0.12
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)

            # with tf.device(self._device_str1):

        # 离散状态下使用
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.ANet['actor_params'],
                                                                      self.old_ANet['actor_params'])]

        # 从动作类型概率分布中选择已选动作进行更新
        # with tf.device(self._device_str):
        # fixme
        self.tfaction_type = tf.placeholder(tf.int32, [None, 1], 'action')
        # a_tfa = tf.reshape(self.tfaction_type, [self.A_DIM, -1])
        action_type_indices = tf.stack([tf.range(tf.shape(self.tfaction_type)[0], dtype=tf.int32),
                                        tf.squeeze(self.tfaction_type, axis=1)], axis=1)
        action_type_prob = tf.gather_nd(params=self.ANet['action_type_prob'], indices=action_type_indices)  # shape=(None, )
        oldaction_type_prob = tf.gather_nd(params=self.old_ANet['action_type_prob'], indices=action_type_indices)  # shape=(None, )

        # 从我方算子概率分布中选择我方算子进行更新
        self.tfmy_id = tf.placeholder(tf.int32, [None, 1], 'action')
        my_id_indices = tf.stack([tf.range(tf.shape(self.tfmy_id)[0], dtype=tf.int32),
                  tf.squeeze(self.tfmy_id, axis=1)], axis=1)
        my_id_prob = tf.gather_nd(params=self.ANet['my_id_prob'], indices=my_id_indices)  # shape=(None, )
        oldmy_id_prob = tf.gather_nd(params=self.old_ANet['my_id_prob'], indices=my_id_indices)  # shape=(None, )

        # 从敌方算子概率分布中选择敌方算子进行更新
        self.tfobj_id = tf.placeholder(tf.int32, [None, 1], 'action')
        obj_id_indices = tf.stack([tf.range(tf.shape(self.tfobj_id)[0], dtype=tf.int32),
                  tf.squeeze(self.tfobj_id, axis=1)], axis=1)
        obj_id_prob = tf.gather_nd(params=self.ANet['obj_id_prob'], indices=obj_id_indices)  # shape=(None, )
        oldobj_id_prob = tf.gather_nd(params=self.old_ANet['obj_id_prob'], indices=obj_id_indices)  # shape=(None, )

        # 从move分布中选择已选位置
        self.tfmove = tf.placeholder(tf.int32, [None, 1], 'move_x')
        move_indices = tf.stack([tf.range(tf.shape(self.tfmove)[0], dtype=tf.int32),
                  tf.squeeze(self.tfmove, axis=1)], axis=1)
        move_prob = tf.gather_nd(params=self.ANet['move_prob'], indices=move_indices)  # shape=(None, )
        oldmove_prob = tf.gather_nd(params=self.old_ANet['move_prob'], indices=move_indices)  # shape=(None, )

        # loss
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')  # 计算TD-error
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfaction_type) - oldpi.log_prob(self.tfaction_type))      # 连续情况下使用 OpenAI算法
                # ratio = pi.tf.divide(self.tfaction_type) / oldpi.prob(self.tfaction_type)                 # 连续情况下使用 DeepMind 算法

                # 三个比率分别计算，最后求和算总体损失
                # 离散情况下使用
                ratio_action_type = tf.divide(action_type_prob, tf.maximum(oldaction_type_prob, 1e-5))

                ratio_my_id = tf.divide(my_id_prob, tf.maximum(oldmy_id_prob, 1e-5))
                ratio_obj_id = tf.divide(obj_id_prob, tf.maximum(oldobj_id_prob, 1e-5))
                ratio_move = tf.divide(move_prob, tf.maximum(oldmove_prob, 1e-5))

                surr_action_type = ratio_action_type * self.tfadv
                surr_my_id = ratio_my_id * self.tfadv
                surr_obj_id = ratio_obj_id * self.tfadv
                surr_move = ratio_move * self.tfadv

            if self.METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(self.old_ANet['action_type_prob'], self.ANet['action_type_prob'])
                self.kl_mean = tf.reduce_mean(kl)
                with tf.name_scope('aloss'):
                    self.aloss = -(tf.reduce_mean(surr_action_type - self.tflam * kl))
            else:  # clipping method, find this is better
                self.aloss = -(
                        tf.reduce_mean(tf.minimum(surr_action_type,
                                                  tf.clip_by_value(ratio_action_type, 1. - self.METHOD['epsilon'],
                                                                   1. + self.METHOD['epsilon']) * self.tfadv)) +
                        tf.reduce_mean(tf.minimum(surr_my_id,
                                                  tf.clip_by_value(ratio_my_id, 1. - self.METHOD['epsilon'],
                                                                   1. + self.METHOD['epsilon']) * self.tfadv)) +
                        tf.reduce_mean(tf.minimum(surr_obj_id,
                                                  tf.clip_by_value(ratio_obj_id, 1. - self.METHOD['epsilon'],
                                                                   1. + self.METHOD['epsilon']) * self.tfadv)) +
                        tf.reduce_mean(tf.minimum(surr_move,
                                                  tf.clip_by_value(ratio_move, 1. - self.METHOD['epsilon'],
                                                                   1. + self.METHOD['epsilon']) * self.tfadv))
                )
        with tf.variable_scope('atrain'):
            with tf.device(self._device_str1):
                self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("log/tensorflow/", self.sess.graph)

        # initializing the global components
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        # loading the network model
        if load_net is True:
            self.load_model()

    def update(self, current_step, spatial_actor, spatial_critic, entity_actor, entity_critic, scalar_state, action,
               my_id, obj_id, move, move_type, r):
        if self.is_ppo_move:
            self.ppo_move.update(current_step, entity_critic, move_type, r)
            # return
        # with tf.device("/cpu:0"):
        logger.info(f'更新网络')
        with tf.device(self._device_str2):
            self.sess.run(self.update_oldpi_op)
            adv = self.sess.run(self.advantage, {
                                                 self.tf_entity_critic: entity_critic,
                                                 self.tf_scalar_state: scalar_state,
                                                 self.tfdc_r: r})
        # adv = (adv - adv.mean()) / (adv.std() + 1e-6)  # sometimes helpful
        # update actor
        with tf.device(self._device_str1):
            if self.METHOD['name'] == 'kl_pen':
                for _ in range(self.A_UPDATE_STEPS):
                    _, kl = self.sess.run(
                        [self.atrain_op, self.kl_mean],
                        {self.tf_spatial_critic: spatial_critic, self.tf_entity_critic: entity_critic,
                         self.tfaction_type: action,
                         self.tfadv: adv, self.tflam: self.METHOD['lam']})
                    if kl > 4 * self.METHOD['kl_target']:  # this in in google's paper
                        break
                if kl < self.METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                    self.METHOD['lam'] /= 2
                elif kl > self.METHOD['kl_target'] * 1.5:
                    self.METHOD['lam'] *= 2
                self.METHOD['lam'] = np.clip(self.METHOD['lam'], 1e-4,
                                             10)  # sometimes explode, this clipping is my solution
            else:  # clipping method, find this is better (OpenAI's paper)
                [self.sess.run(self.atrain_op, {
                                                self.tf_entity_critic: entity_critic,
                                                self.tf_scalar_state: scalar_state,
                                                # self.tfaction_type: np.squeeze(a, axis=1),
                                                self.tfaction_type: action,
                                                self.tfmy_id: my_id,
                                                self.tfobj_id: obj_id,
                                                self.tfmove: move,
                                                self.tfadv: adv}
                               ) for _ in range(self.A_UPDATE_STEPS)]
        # update critic
        with tf.device(self._device_str2):
            [self.sess.run(self.ctrain_op,
                           {
                            # self.tf_spatial_critic: spatial_critic,
                            self.tf_entity_critic: entity_critic,
                            self.tf_scalar_state: scalar_state,
                            self.tfdc_r: r}) for _ in
             range(self.C_UPDATE_STEPS)]
        # 显示损失函数
        adv_lis = np.squeeze(np.square(adv), axis=1)
        self.adv_list += adv_lis.tolist()
        critic_list = np.squeeze(self.get_v(entity_critic, scalar_state), axis=1)
        self.critic_list += critic_list.tolist()
        if self.dislpay_loss_counter == self.display_loss_interval:
            self.display_loss(self.adv_list, self.critic_list)
            self.dislpay_loss_counter = 0
        self.dislpay_loss_counter += 1

        # tensorbord 显示损失函数
        # with tf.device('/cpu:0'):
        #     loss = self.sess.run(self.merged,
        #                          feed_dict={
        #                                     self.tf_entity_critic: self.buffer_entity_critic,
        #                                     self.tf_scalar_state: self.buffer_scalar_state,
        #                                     self.tfdc_r:  self.buffer_r,
        #                                     self.tfaction_type: self.buffer_action,
        #                                     self.tfmy_id: self.buffer_my_id,
        #                                     self.tfobj_id: self.buffer_obj_id,
        #                                     self.tfmove: move,
        #                                     self.tfadv: adv,
        #                                     # self.tf_spatial_critic: spatial_critic,
        #                                     self.ppo_move.tfa: move,
        #                                     self.ppo_move.tfadv: adv,
        #                                     self.ppo_move.tfs: entity_critic,
        #                                     self.ppo_move.tfdc_r: r
        #                                     })
        #     self.writer.add_summary(loss, current_step)


    def choose_action(self, spatial_actor, entity_actor, spatial_critic, entity_critic, scalar_state,
                      interface: ValidActionInterface):
        result = {
            'Thread_id': -1,
            'action_type': -1,
            'my_id': -1,
            'my_id_output': -1,
            'target_id': -1,
            'target_id_output': -1,
            'target_state': -1,
            'move': -1,
            'move_type': -1
        }

        map_size_x = interface.map_size_x
        map_size_y = interface.map_size_y

        valid_x = np.zeros(shape=(1, self.max_map_size_x))
        valid_x[:, 0:map_size_x] = np.ones(shape=(1, map_size_x))
        valid_y = np.zeros(shape=(1, self.max_map_size_y))
        valid_y[:, 0:map_size_y] = np.ones(shape=(1, map_size_y))

        # 获取动作算子
        my_id_vector = interface.get_valid_my_operator_id_vector()
        prob_weights_my_id = self.sess.run(self.ANet['my_id_prob'], feed_dict={self.tf_spatial_critic: spatial_critic,
                                                                               self.tf_entity_critic: entity_critic,
                                                                               self.tf_scalar_state: scalar_state})
        if sum(my_id_vector) != 0:
            prob_weights_my_id_output = prob_weights_my_id[0:] * my_id_vector
            prob_weights_my_id_output[0] = prob_weights_my_id_output[0] / sum(prob_weights_my_id_output[0])
            my_id_output = np.random.choice(range(prob_weights_my_id_output.shape[1]),
                                            p=prob_weights_my_id_output[0])  # select action w.r.t the actions prob
            my_id = interface.query_my_id(my_id_output)
            result['my_id'] = my_id
        else:
            my_id_output = np.random.choice(range(prob_weights_my_id[0:].shape[1]),
                                            p=prob_weights_my_id[0:][0])  # select action w.r.t the actions prob
            my_id = -1
            result['my_id'] = my_id
        result['my_id_output'] = my_id_output

        # 获取可用动作类型：
        if my_id != -1:
            if my_id == 200:
                valid_type_vector = interface.get_type_vector_by_selected_operator_id(my_id)  # 输出的是长14的list
            else:
                valid_type_vector = interface.get_type_vector_by_selected_operator_id(my_id)  # 输出的是长14的list
            if sum(valid_type_vector) != 0:
                prob_weights_action_type = self.sess.run(self.ANet['action_type_prob'],
                                                         feed_dict={self.tf_spatial_critic: spatial_critic,
                                                         self.tf_entity_critic: entity_critic,
                                                         self.tf_scalar_state: scalar_state})
                prob_weights_action_type = prob_weights_action_type[0:] * valid_type_vector
                prob_weights_action_type[0] = prob_weights_action_type[0] / sum(prob_weights_action_type[0])
                action_type_No = np.random.choice(range(prob_weights_action_type.shape[1]),
                                                  p=prob_weights_action_type[0])  # select action w.r.t the actions prob
                # action_type_No = np.unravel_index(np.argmax(prob_weights_action_type[0]), prob_weights_action_type[0].shape)[0]
                # tf.multinomial(logits, num_samples, seed=None, name=None)
                action_type = interface.query_action_type(action_type_No)  # output应为11的list

                # if my_id == 200 and valid_type_vector[3] == 1 and self.get_on_flage is False:
                #     action_type = 3
                #     self.get_on_flage = True
                if valid_type_vector[2] == 1:
                    action_type = 2
                if valid_type_vector[9] == 1:
                    action_type = 9
                if valid_type_vector[5] == 1:
                    action_type = 5
                result['action_type'] = action_type
            else:
                action_type = 15
                result['action_type'] = action_type
        else:
            action_type = 15
            result['action_type'] = action_type

        # 获取可用算子：
        # my_id_vector = interface.get_operator_vector_of_specific_type(action_type)
        # prob_weights_my_id = self.sess.run(self.ANet['my_id_prob'], feed_dict={self.tf_spatial_critic: spatial_critic,
        #                                                                self.tf_entity_critic: entity_critic,
        #                                                                self.tf_scalar_state: scalar_state})
        # if sum(my_id_vector) != 0:
        #     prob_weights_my_id_output = prob_weights_my_id[0:] * my_id_vector
        #     prob_weights_my_id_output[0] = prob_weights_my_id_output[0] / sum(prob_weights_my_id_output[0])
        #     my_id_output = np.random.choice(range(prob_weights_my_id_output.shape[1]),
        #                                     p=prob_weights_my_id_output[0])  # select action w.r.t the actions prob
        #     my_id = interface.query_my_id(my_id_output)
        #     result['my_id'] = my_id
        #
        #     if my_id == 400 and valid_type_vector[3] == 1 and self.get_on_flage is False:
        #         action_type = 3
        #         result['action_type'] = action_type
        #         self.get_on_flage = True
        # else:
        #     my_id_output = np.random.choice(range(prob_weights_my_id[0:].shape[1]),
        #                                     p=prob_weights_my_id[0:][0])  # select action w.r.t the actions prob
        #     my_id = -1
        #     result['my_id'] = my_id
        # result['my_id_output'] = my_id_output

        if self.is_ppo_move and my_id_output != -1:
            self.move_type = self.ppo_move.choose_action(entity_critic)
            valid_type_vector = interface.get_type_vector()  # 输出的是长11的list
            if valid_type_vector[1] == 1:
                result['move_type'] = self.move_type
            else:
                result['move_type'] = -1
            print(self.move_type)

        # 获取坐标信息
        with tf.device('/gpu:0'):
            prob_move = self.sess.run(self.ANet['move_prob'], feed_dict={self.tf_spatial_critic: spatial_critic,
                                                                         self.tf_entity_critic: entity_critic,
                                                                         self.tf_scalar_state: scalar_state})
        result['move'] = np.random.choice(range(prob_move.shape[1]), p=prob_move[0])
        # result['move'] = np.unravel_index(np.argmax(prob_move[0]), prob_move[0].shape)[0]
        result['move_prob'] = prob_move[0]

        # 获取目标id参数：
        prob_weights_obj_id = self.sess.run(self.ANet['obj_id_prob'],
                                            feed_dict={self.tf_spatial_critic: spatial_critic,
                                            self.tf_entity_critic: entity_critic,
                                            self.tf_scalar_state: scalar_state})  # output应为34的list
        if my_id != -1 and action_type not in [0, 15]:
            param = interface.get_param_by_selected_action_type(action_type, my_id)
            if param is not None:
                tp = param['type']
                value = param['value']  # 输出的是长34的list
                prob_weights_obj_id_output = prob_weights_obj_id[0:] * value
                prob_weights_obj_id_output[0] = prob_weights_obj_id_output[0] / sum(prob_weights_obj_id_output[0])
                target_id_output = np.random.choice(range(prob_weights_obj_id_output.shape[1]),
                                          p=prob_weights_obj_id_output[0])  # select action w.r.t the actions prob
                result["target_id_output"] = target_id_output

                if tp == 'my_ops':  # 上下车动作
                    result['target_id'] = interface.query_my_id(target_id_output)
                elif tp == 'enemy_ops':  # 攻击动作
                    result['target_id'] = interface.query_enemy_id(target_id_output)
                elif tp == 'state':  # 动作6: CHANGE_STATE
                    result['target_state'] = interface.query_state()
            else:
                target_id_output = np.random.choice(range(prob_weights_obj_id.shape[1]),
                                                    p=prob_weights_obj_id[0])  # select action w.r.t the actions prob
                result["target_id_output"] = target_id_output
        else:
            target_id_output = np.random.choice(range(prob_weights_obj_id.shape[1]),
                                                p=prob_weights_obj_id[0])  # select action w.r.t the actions prob
            result["target_id_output"] = target_id_output
        return result

    def get_v(self, entity_critic, scalar_state):
        if self.is_ppo_move:
            self.ppo_move.get_v(entity_critic)
        # if spatial_critic.ndim < 2: s = spatial_critic[np.newaxis, :]
        # 通过.eval函数可以把tensor转化为numpy类数据
        # state = s.detach().numpy()
        # return self.sess.run(self.v, {self.tfs: s})[:, 0, 0, 0]
        return self.sess.run(self.critic, {
                                    # self.tf_spatial_critic: spatial_critic,
                                    self.tf_entity_critic: entity_critic,
                                    self.tf_scalar_state: scalar_state})

    def store_transition(self, current_step, spatial_actor, spatial_critic, entity_actor, entity_critic, scalar_state,
                         action,  my_id, obj_id, move, move_type, r, basic):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        if self.memory_entity_critic is None:
            self.memory_entity_critic = entity_critic
            self.memory_scalar_state = scalar_state
            self.memory_r = r
            self.memory_action = action
            self.memory_my_id = my_id
            self.memory_obj_id = obj_id
            self.memory_move = move
        else:
            self.memory_entity_critic = np.vstack([entity_critic, self.memory_entity_critic])
            self.memory_scalar_state = np.vstack([scalar_state, self.memory_scalar_state])
            self.memory_r = np.vstack([r, self.memory_r])
            self.memory_action = np.vstack([action, self.memory_action])
            self.memory_my_id = np.vstack([my_id, self.memory_my_id])
            self.memory_obj_id = np.vstack([obj_id, self.memory_obj_id])
            self.memory_move = np.vstack([move, self.memory_move])
        self.memory_counter += 1

        if len(self.memory_entity_critic) > 2048:
            # self.memory_entity_critic[0: -self.batch -1, :] = self.memory_entity_critic[self.batch: -1, :]
            # self.memory_scalar_state[0: -self.batch -1, :] = self.memory_scalar_state[self.batch: -1, :]
            # self.memory_r[0: -self.batch -1, :] = self.memory_r[self.batch: -1, :]
            # self.memory_action[0: -self.batch -1, :] = self.memory_action[self.batch: -1, :]
            # self.memory_my_id[0: -self.batch -1, :] = self.memory_my_id[self.batch: -1, :]
            # self.memory_obj_id[0: -self.batch -1, :] = self.memory_obj_id[self.batch: -1, :]
            # self.memory_move[0: -self.batch -1, :] = self.memory_move[self.batch: -1, :]

            delete_list = list(range(self.batch))
            self.memory_entity_critic = np.delete(self.memory_entity_critic, delete_list, 0)
            self.memory_scalar_state = np.delete(self.memory_scalar_state, delete_list, 0)
            self.memory_r = np.delete(self.memory_r, delete_list, 0)
            self.memory_action = np.delete(self.memory_action, delete_list, 0)
            self.memory_my_id = np.delete(self.memory_my_id, delete_list, 0)
            self.memory_obj_id = np.delete(self.memory_obj_id, delete_list, 0)
            self.memory_move = np.delete(self.memory_move, delete_list, 0)

        for i in range(1):
            # if len(self.memory_r) < self.batch * 50:
            #     index = np.random.choice(range(len(self.memory_r)), self.batch, False)
            # else:
            #     index = np.random.choice(range(self.batch * 50), self.batch, False)
            #     index = index + len(self.memory_r) - self.batch * 50
            if len(self.memory_entity_critic) > self.batch * 5:
                index = np.random.choice(range(len(self.memory_entity_critic)), self.batch * 5, False)
            else:
                index = np.random.choice(range(len(self.memory_entity_critic)), self.batch, False)
            entity_critic = self.memory_entity_critic[index, :]
            scalar_state = self.memory_scalar_state[index, :]
            r = self.memory_r[index, :]
            action = self.memory_action[index, :]
            my_id = self.memory_my_id[index, :]
            obj_id = self.memory_obj_id[index, :]
            move = self.memory_move[index, :]
            self.update(current_step, spatial_actor, spatial_critic, entity_actor, entity_critic, scalar_state, action,
                   my_id, obj_id, move, move_type, r)


    def get_log_act(self, logits, acts):
        try:
            log_act = 0
            for logit, act in zip(logits, acts):
                log_act += math.log(logit[int(act)] + self.eps)
            return log_act
        except Exception as e:
            print('error in class policy -> get_log_act() : {}'.format(str(e)))
            raise e

        ppo_actor_ratio = (new_log_acts - old_log_acts).exp()

        actor_loss = - (th.clamp(ppo_actor_ratio, self.eps, 1.0 + self.rl_hyper['ppo_actor_ratio']) * (
            advs.detach())).mean()

    def display_loss(self, adv_list, critic_list):
        # plt.plot(np.arange(len(adv_list)), adv_list, "--o", color="b")
        plt.plot(np.arange(len(critic_list)), critic_list, "-", color="r")
        # plt.bar(bar_index, pro[0], width=0.2, align='center')
        # plt.axis('off')
        plt.draw()
        # plt.pause(0.001)

    def get_loss(self, cur_step, s, a, r):
        with tf.device('/cpu:0'):
            adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
            loss = self.sess.run(self.merged, feed_dict={
                # self.tfaction_type: np.squeeze(a, axis=1),
                self.tfa: a,
                self.tfadv: adv,
                self.tfs: s,
                self.tfdc_r: r})
            self.writer.add_summary(loss, cur_step)

    # 保存网络参数
    def save_model(self, global_step, save_model_num):
        # save_file = game_num % 10
        save_model_num = save_model_num
        self.saver.save(self.sess, self.model_path + str(save_model_num) + '/',
                        global_step=global_step + 1)  # write_meta_graph=False

        # self.saver(self.sess, 'PredictAI_model', write_meta_graph=False, keep_checkpoint_every_n_hours=2, max_to_keep=10)

    # 读取网络参数
    def load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path + str(self.load_model_num) + '/')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        logger.info('加载神经网络' )

    # 监督学习
    def supervised_learning(self):
        pass
        # ## fc2 layer ##
        # W_fc2 = weight_variable([1024, 10])
        # b_fc2 = bias_variable([10])
        # prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        #
        # # the error between prediction and real data
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
        #                                               reduction_indices=[1]))  # loss
        # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
from ai_company.core import logger
import matplotlib.pyplot as plt
import gym
import keras.models

GAMMA = 0.9
A_LR = 0.001
C_LR = 0.002
training = True

A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM = 6

load_net = False
load_model_num = 3
is_trainable = True

METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

class PPO_MOVE(object):
    def __init__(self, step_num, load_net=load_net, load_model_num = load_model_num, is_trainable = is_trainable):
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            self.A_DIM = sum(range(step_num+1)) * 6
            self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
            self.move_space = 6
            self.display_loss_num = 0
            self.display_loss_interval = 50

            self.load_net = load_net
            self.load_model_num = load_model_num
            self.model_path_move = 'PredictAI_model_move/'

            self.fig = plt.figure()
            self.diff = self.fig.add_subplot(1, 1, 1)
            plt.ion()  # 本次运行请注释，全局运行不要注释
            plt.show()
            self.adv_list = []

            # critic
            with tf.variable_scope('critic'):
                w_init = tf.random_normal_initializer(0., .1)
                w_initializer = tf.contrib.layers.l2_regularizer(0.003)
                # init_w = tf.contrib.layers.xavier_initializer()
                # init_b = tf.constant_initializer(0.001)
                cnet_layer = [500, 500, 500]
                cnet = self.build_general_net(cnet_layer, self.tfs, "critic", True)
                lc = tf.layers.dense(cnet, 200, tf.nn.relu, kernel_initializer=w_init, name='lc')
                self.v = tf.layers.dense(lc, 1)
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))
                with tf.device('/cpu:0'):
                    tf.summary.scalar('closs', self.closs)  # tensorflow >= 0.12
                self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

            # actor
            with tf.variable_scope('actor'):
                self.tfa = tf.placeholder(tf.int32, [None, ], 'move_type')
                self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
                self.pi, pi_params = self._build_anet('pi', trainable=is_trainable)
                oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
                self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

                # a = tf.shape(self.tfa)[0]
                # b = [tf.range(a, dtype=tf.int32), self.tfa]
                # a_indices = tf.stack(b, axis=1)
                # pi_prob = tf.gather_nd(params=self.pi, indices=self.tfa)  # shape=(None, )
                # oldpi_prob = tf.gather_nd(params=oldpi, indices=self.tfa)  # shape=(None, )
                # ratio = pi_prob / oldpi_prob
                # surr = ratio * self.tfadv  # surrogate aloss
                a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
                pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)  # shape=(None, )
                oldpi_prob = tf.gather_nd(params=oldpi, indices=a_indices)  # shape=(None, )
                ratio = pi_prob / oldpi_prob
                surr = ratio * self.tfadv  # surrogate loss
                with tf.variable_scope('aloss'):
                    self.aloss = -tf.reduce_mean(tf.minimum(
                        surr,
                        tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * self.tfadv))

                    self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
                    # self.atrain_op = model.compile(loss='mse',
                    #               optimizer=TFOptimizer(tf.train.GradientDescentOptimizer(0.1)))

            self.merged = tf.summary.merge_all()
            self.writer_move = tf.summary.FileWriter("log/", self.sess.graph)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            # loading the network model
            if self.load_net is True:
                self.load_model()

    def update(self, begining_step, s_s_critic, move_type, r):
        with tf.device('/cpu:0'):
            self.sess.run(self.update_oldpi_op)
            adv = self.sess.run(self.advantage, {self.tfs: s_s_critic, self.tfdc_r: r})
            # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

            # update actor
            [self.sess.run(self.atrain_op, {self.tfs: s_s_critic, self.tfa: np.squeeze(move_type, axis=1), self.tfadv: adv}) for _ in
             range(A_UPDATE_STEPS)]

            # update critic
            [self.sess.run(self.ctrain_op, {self.tfs: s_s_critic, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

            # 显示损失函数
            lis = np.squeeze(np.square(adv), axis=1)
            self.adv_list += lis.tolist()
            if self.display_loss_num == self.display_loss_interval:
                self.display_loss(self.adv_list)
                self.display_loss_num = 0
            self.display_loss_num += 1

            # 记录tensorboard, 主ppo开启记录的这里要关掉
            # loss = self.sess.run(self.merged, feed_dict={
            #                                              # self.tfaction_type: np.squeeze(a, axis=1),
            #                                              self.tfa: move_type,
            #                                              self.tfadv: adv,
            #                                              self.tfs: s_s_critic,
            #                                              self.tfdc_r: r})
            #
            # self.writer_move.add_summary(loss, begining_step)

    def _build_anet(self, name, trainable):
        with tf.device('/cpu:0'):
            with tf.variable_scope(name):
                w_init = tf.random_normal_initializer(0., .1)
                w_initializer = tf.contrib.layers.l2_regularizer(0.003)
                anet_layer = [500, 500, 500]
                anet = self.build_general_net(anet_layer, self.tfs, "actor", True)
                l_a = tf.layers.dense(anet, 500, tf.nn.relu, trainable=trainable)
                a_prob = tf.layers.dense(l_a, self.A_DIM, tf.nn.softmax, trainable=trainable)
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            return a_prob, params

    def build_general_net(self, layer_number, input, name, is_train):
        for i in range(len(layer_number)):
            with tf.variable_scope(name, str(layer_number[i]), reuse=tf.AUTO_REUSE):
                g_net = tf.layers.dense(input, units=layer_number[i],
                                        activation=tf.nn.relu6, use_bias=True,
                                        bias_initializer=tf.zeros_initializer(),
                                        bias_regularizer=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                                        trainable=is_train
                                        )
                n_net = self.layer_normalize(g_net)
        return n_net

    def layer_normalize(self, net):
        g_net = tf.layers.batch_normalization(inputs=net, name='normalization')
        return g_net

    def choose_action(self, s_s_critic):  # run by a local
        with tf.device('/cpu:0'):
            prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s_s_critic})
            action = np.random.choice(range(prob_weights.shape[1]),
                                      p=prob_weights.ravel())  # select action w.r.t the actions prob
            print("移动概率分布：", prob_weights.ravel(), "移动类型：", action)
            return action

    def get_v(self, s_s_critic):
        with tf.device('/cpu:0'):
            if s_s_critic.ndim < 2: s = s_s_critic[np.newaxis, :]
            return self.sess.run(self.v, {self.tfs: s_s_critic})

    def display_loss(self, adv_list):
        plt.plot(np.arange(len(adv_list)), adv_list)
        # plt.bar(bar_index, pro[0], width=0.2, align='center')
        # plt.axis('off')
        plt.draw()
        plt.pause(0.001)

    # 保存网络参数
    def save_model(self, global_step, save_model_num):
        # save_file = game_num % 10
        self.saver.save(self.sess, self.model_path_move + str(save_model_num) + '/',
                        global_step=global_step + 1)  # write_meta_graph=False

        # self.saver(self.sess, 'PredictAI_model', write_meta_graph=False, keep_checkpoint_every_n_hours=2, max_to_keep=10)

    # 读取网络参数
    def load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path_move + str(self.load_model_num) + '/')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        logger.info('加载神经网络')


import tensorflow as tf

class ACNet(object):
    def __init__(self, img_high, img_width, Channel, lstm_batch_size, actions_space, action_type_size, my_id_size,
                 obj_id_size, move, A_LR, use_gpu):
        self.UNITS = {'resnet_v2_50': [3, 4, 6, 3], 'resnet_v2_101': [3, 4, 23, 3],
                 'resnet_v2_152': [3, 8, 36, 3], 'resnet_v2_18': [2, 2, 2, 2]}
        self.CHANNELS = [64, 128, 256, 512]
        self._device_str = '/gpu:0' if use_gpu else '/cpu:0'
        self.img_high = img_high
        self.img_width = img_width
        self.Channel = Channel
        self.actions_space = actions_space
        self.action_type_size = action_type_size
        self.my_id_size = my_id_size
        self.obj_id_size = obj_id_size
        self.move_space = move
        self.label = 10
        self.block_num = 4
        self.A_LR = A_LR

        #define the lstm
        self.lstm_n_hidden_units = 384
        self.lstm_batch_size = lstm_batch_size
        # self.LSTM = LSTMRNN(n_steps, input_size, output_size, cell_size, lstm_batch_size, A_LR)

    def build_entity_net(self, inputs_scale, is_train):
        # MLP处理标量信息
        with tf.variable_scope("entity_net", reuse=tf.AUTO_REUSE):
            with tf.variable_scope('entity_fc', reuse=tf.AUTO_REUSE):
                entity_net_layer = [128, 128]
                entity_net = self.build_general_net(entity_net_layer, inputs_scale, 'entity_net_general',
                                                    is_train)

            # with tf.variable_scope('entity_relu', reuse=tf.AUTO_REUSE):
            #     # 全连接层
            #     entity_net = tf.layers.dense(inputs=entity_net,
            #                              units=512,
            #                              bias_initializer=tf.zeros_initializer(),
            #                              activation=tf.nn.relu,
            #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
            #                              trainable=is_train)
        return entity_net

    def build_scalar_net(self, inputs_scale, is_train):
        # MLP处理标量信息
        with tf.variable_scope("scalar_net", reuse=tf.AUTO_REUSE):
            with tf.variable_scope('scalar_fc', reuse=tf.AUTO_REUSE):
                # 全连接层
                sclar_net_layer = [64, 64]
                sclar_net = self.build_general_net(sclar_net_layer, inputs_scale, 'scalar_net_general',
                                                   is_train)

            # with tf.variable_scope('scalar_relu', reuse=tf.AUTO_REUSE):
            #     # 全连接层
            #     sclar_net = tf.layers.dense(inputs=sclar_net,
            #                              units=64,
            #                              bias_initializer=tf.zeros_initializer(),
            #                              activation=tf.nn.relu,
            #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
            #                              trainable=is_train)
        return sclar_net

    def build_general_net(self, layer_number, input, name, is_train):
        with tf.variable_scope(name):
            input_ = input
            for i in range(len(layer_number)):
                # if layer_number[i] % 2 == 0 and layer_number[i] != 0:
                #     activation = tf.nn.relu
                # else:
                #     activation = None
                g_net = tf.layers.dense(input_, units=layer_number[i],
                                        activation=tf.nn.relu,
                                        use_bias=True,
                                        bias_initializer=tf.zeros_initializer(),
                                        bias_regularizer=None,
                                        kernel_regularizer=None,
                                        # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                                        trainable=is_train)
                n_net = self.layer_normalize(g_net)
                input_ = n_net
        return n_net

    def layer_normalize(self, net):
        # LN用于RNN效果比较明显，但是在CNN上，不如BN。
        # g_net = tf.layers.batch_normalization(inputs=net)
        g_net = tf.nn.l2_normalize(net, 1)
        return g_net

    def build_ACNet(self, inputs_entity, inputs_scalar, is_train):
        # with tf.device(self._device_str):
        ANet = {}
        old_ANet = {}
        with tf.variable_scope("old_actor"):
            # define initializer for weights and biases
            # w_initializer = tf.contrib.layers.xavier_initializer()
            w_initializer = tf.random_normal_initializer(0., .1)
            b_initializer = tf.zeros_initializer()
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            # 算子信息网络
            entity_mlp_net = self.build_entity_net(inputs_entity, False)
            # 标量信息网络
            scalar_mlp_net = self.build_scalar_net(inputs_scalar, False)

            # 张量拼接 
            # = tf.squeeze(spatial_con_net, axis=(1, 2))  # 首先对resnet两个没用的尺寸信息降维
            union_net = tf.concat([entity_mlp_net, scalar_mlp_net], 1)  # 和mlp的数据进行拼接
            # union_net = tf.expand_dims(union_net, axis=1)  # 升维成为lstm的格式
            # union_net = tf.expand_dims(union_net, axis=3)
            # union_net = tf.unstack(union_net, 1, 1)  # 按照时间步展开

            # 定义LSTM
            # with tf.device(self._device_str):
            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_n_hidden_units, forget_bias=1.0, state_is_tuple=True)
            # init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)  # 初始化全零 state

            # define the last convolution layer
            with tf.variable_scope('public'):
                # kernel = tf.get_variable(initializer=w_initializer,
                #                          shape=[1, 1, 1, 1], name='weights',
                #                          regularizer=regularizer,
                #                          # collections=['non_pretrain', key]
                #                          )
                # net = tf.nn.conv2d(input=union_net, filter=kernel,
                #                    strides=[1, 1, 1, 1], padding='VALID')
                # biases = tf.get_variable(initializer=b_initializer, shape=1,
                #                          name='biases', regularizer=regularizer,
                #                          # collections=['non_pretrain', key]
                #                          )
                # net = tf.nn.bias_add(net, biases)
                # net = tf.squeeze(net, axis=(1, 3))

                net_layer = [128, 128, 128]
                net = self.build_general_net(net_layer, union_net, 'public_general', False)

            # define the prob of the actions_type
            with tf.variable_scope('actions_type'):
                action_type_net_layer = [128, 128]
                action_type = self.build_general_net(action_type_net_layer, net, 'actions_type_general',
                                                     False)
                action_type = tf.layers.dense(
                    inputs=action_type,
                    units=self.action_type_size,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=False
                )

            # define the prob of the my_id
            with tf.variable_scope('my_id'):
                my_id_net_layer = [128, 128]
                my_id = self.build_general_net(my_id_net_layer, net, 'my_id_general', False)
                my_id = tf.layers.dense(
                    inputs=my_id,
                    units=self.my_id_size,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=False
                )

            # define the prob of the obj_id
            with tf.variable_scope('obj_id'):
                obj_id_net_layer = [128, 128]
                obj_id = self.build_general_net(obj_id_net_layer, net, 'obj_id_general', False)
                obj_id = tf.layers.dense(
                    inputs=obj_id,
                    units=self.obj_id_size,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=False
                )

            # define the prob of the move
            with tf.variable_scope('move'):
                move_net_layer = [128, 128]
                move = self.build_general_net(move_net_layer, net, 'move_general', False)
                move = tf.layers.dense(
                    inputs=move,
                    units=self.move_space,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=False
                )

            old_ANet['action_type_prob'] = tf.nn.softmax(action_type)
            old_ANet['my_id_prob'] = tf.nn.softmax(my_id)
            old_ANet['obj_id_prob'] = tf.nn.softmax(obj_id)
            old_ANet['move_prob'] = tf.nn.softmax(move)
            old_ANet['actor_params'] = tf.get_collection(tf.GraphKeys.VARIABLES, scope='actor')

        with tf.variable_scope('actor'):
            # define initializer for weights and biases
            # w_initializer = tf.contrib.layers.xavier_initializer()
            w_initializer = tf.random_normal_initializer(0., .1)
            b_initializer = tf.zeros_initializer()
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            # 算子信息网络
            entity_mlp_net = self.build_entity_net(inputs_entity, is_train)
            # 标量信息网络
            scalar_mlp_net = self.build_scalar_net(inputs_scalar, is_train)

            # 张量拼接
            # = tf.squeeze(spatial_con_net, axis=(1, 2))  # 首先对resnet两个没用的尺寸信息降维
            union_net = tf.concat([entity_mlp_net, scalar_mlp_net], 1)  # 和mlp的数据进行拼接
            # union_net = tf.expand_dims(union_net, axis=1)  # 升维成为lstm的格式
            # union_net = tf.expand_dims(union_net, axis=3)
            # union_net = tf.unstack(union_net, 1, 1)  # 按照时间步展开

            # 定义LSTM
            # with tf.device(self._device_str):
            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_n_hidden_units, forget_bias=1.0, state_is_tuple=True)
            # init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)  # 初始化全零 state

            # define the last convolution layer
            with tf.variable_scope('public'):
            #     kernel = tf.get_variable(initializer=w_initializer,
            #                              shape=[1, 1, 1, 1], name='weights',
            #                              regularizer=regularizer,
            #                              # collections=['non_pretrain', key]
            #                              )
            #     net = tf.nn.conv2d(input=union_net, filter=kernel,
            #                        strides=[1, 1, 1, 1], padding='VALID')
            #     biases = tf.get_variable(initializer=b_initializer, shape=1,
            #                              name='biases', regularizer=regularizer,
            #                              # collections=['non_pretrain', key]
            #                              )
            #     net = tf.nn.bias_add(net, biases)
            #     net = tf.squeeze(net, axis=(1, 3))

                net_layer = [128, 128, 128]
                net = self.build_general_net(net_layer, union_net, 'public_general', is_train)

            # define the prob of the actions_type
            with tf.variable_scope('actions_type'):
                action_type_net_layer = [128, 128]
                action_type = self.build_general_net(action_type_net_layer, net, 'actions_type_general',
                                                     is_train)
                action_type = tf.layers.dense(
                    inputs=action_type,
                    units=self.action_type_size,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=is_train
                )

            # define the prob of the my_id
            with tf.variable_scope('my_id'):
                my_id_net_layer = [128, 128]
                my_id = self.build_general_net(my_id_net_layer, net, 'my_id_general', is_train)
                my_id = tf.layers.dense(
                    inputs=my_id,
                    units=self.my_id_size,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=is_train
                )

            # define the prob of the obj_id
            with tf.variable_scope('obj_id'):
                obj_id_net_layer = [128, 128]
                obj_id = self.build_general_net(obj_id_net_layer, net, 'obj_id_general', is_train)
                obj_id = tf.layers.dense(
                    inputs=obj_id,
                    units=self.obj_id_size,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=is_train
                )

            # define the prob of the move
            with tf.variable_scope('move'):
                move_net_layer = [128, 128]
                move = self.build_general_net(move_net_layer, net, 'move_general', is_train)
                move = tf.layers.dense(
                    inputs=move,
                    units=self.move_space,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=is_train
                )

            ANet['action_type_prob'] = tf.nn.softmax(action_type)
            ANet['my_id_prob'] = tf.nn.softmax(my_id)
            ANet['obj_id_prob'] = tf.nn.softmax(obj_id)
            ANet['move_prob'] = tf.nn.softmax(move)
            ANet['actor_params'] = tf.get_collection(tf.GraphKeys.VARIABLES, scope='actor')

        with tf.variable_scope("critic"):
            # w_initializer = tf.contrib.layers.xavier_initializer()
            w_initializer = tf.random_normal_initializer(0., .1)
            b_initializer = tf.zeros_initializer()
            cnet_layer = [128, 128, 128]
            cnet = self.build_general_net(cnet_layer, net, 'critic_general', True)
            l_c = tf.layers.dense(cnet, 500, tf.nn.relu, kernel_initializer=w_initializer, name='lc')
            critic = tf.layers.dense(l_c, 1)
            critic_paramas = tf.get_collection(tf.GraphKeys.VARIABLES, scope='critic')

        return ANet, old_ANet, critic


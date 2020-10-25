# 当加载 Resnet_152的时候 会发生GPU内存溢出 所以就是用CPU进行训练
# 当使用 inception_V4 batch_sizei为8的时候 就会出现内存溢出的问题 说明这个网络还是比较复杂的
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
# 定义一些模型中所需要的参数
import tensorflow as tf

class Actor(object):
    def __init__(self, img_high, img_width, Channel, lstm_batch_size, actions_space, action_type_size, my_id_size,
                 obj_id_size, A_LR, use_gpu):
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
        self.label = 10
        self.block_num = 4
        self.A_LR = A_LR

        #define the lstm
        self.lstm_n_hidden_units = 384
        self.lstm_batch_size = lstm_batch_size
        # self.LSTM = LSTMRNN(n_steps, input_size, output_size, cell_size, lstm_batch_size, A_LR)

        # 使用ResNet_50_101_152 需要在最后加上batch normal 所以需要使用 is_train
        self.resnet_type ='resnet_v2_18'  # 'resnet_v2_18' 'resnet_v2_50'  'resnet_v2_101'  'resnet_v2_152'

        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.99  # 使用GPU显存的比率
        # config.gpu_options.allow_growth = True  # 按需求使用GPU
        # config.log_device_placement = False
        # # config.device_count = {'cpu': 0}
        # # config.allow_soft_placement = True
        # self.sess_actor = tf.Session(config=config)

    def bottleneck(self, net, channel, is_train, holes=1, c_name='pretrain', stride=1,
                   shortcut_conv=False, key=tf.GraphKeys.GLOBAL_VARIABLES):
        with tf.variable_scope('bottleneck_v2', reuse=tf.AUTO_REUSE):
            # define initializer for weights and biases
            w_initializer = tf.contrib.layers.xavier_initializer()
            b_initializer = tf.zeros_initializer()
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            # batch normalization
            net = tf.layers.batch_normalization(inputs=net, axis=-1,
                                                training=is_train, name='preact')
            net = tf.nn.relu(net)

            # shortcut
            if shortcut_conv:
                with tf.variable_scope('shortcut', reuse=tf.AUTO_REUSE):
                    kernel = tf.get_variable(initializer=w_initializer,
                                             shape=[1, 1, net.shape[-1],
                                                    channel * 4],
                                             name='weights',
                                             regularizer=regularizer,
                                             collections=['pretrain', key])
                    # convolution for shortcut in order to output size
                    shortcut = tf.nn.conv2d(input=net, filter=kernel,
                                            strides=[1, stride, stride, 1],
                                            padding='SAME')
                    biases = tf.get_variable(initializer=b_initializer,
                                             shape=channel * 4, name='biases',
                                             regularizer=regularizer,
                                             collections=['pretrain', key])
                    shortcut = tf.nn.bias_add(shortcut, biases)
            else:
                # shortcut
                shortcut = net

            # convolution 1
            with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
                kernel = tf.get_variable(initializer=w_initializer,
                                         shape=[1, 1, net.shape[-1], channel],
                                         name='weights', regularizer=regularizer,
                                         collections=['pretrain', key])
                net = tf.nn.atrous_conv2d(value=net, filters=kernel, rate=holes,
                                          padding='SAME')
                biases = tf.get_variable(initializer=b_initializer,
                                         shape=channel, name='biases',
                                         regularizer=regularizer,
                                         collections=['non_pretrain', key])
                net = tf.nn.bias_add(net, biases)
                # batch normalization
                net = tf.layers.batch_normalization(inputs=net, axis=-1,
                                                    training=is_train,
                                                    name='preact')
                net = tf.nn.relu(net)

            # convolution 2
            with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
                kernel = tf.get_variable(initializer=w_initializer,
                                         shape=[3, 3, channel, channel],
                                         name='weights', regularizer=regularizer,
                                         collections=['pretrain', key])
                net = tf.nn.conv2d(input=net, filter=kernel,
                                   strides=[1, stride, stride, 1], padding='SAME')
                biases = tf.get_variable(initializer=b_initializer,
                                         shape=channel, name='biases',
                                         regularizer=regularizer,
                                         collections=['non_pretrain', key])
                net = tf.nn.bias_add(net, biases)
                # batch normalization
                net = tf.layers.batch_normalization(inputs=net, axis=-1,
                                                    training=is_train,
                                                    name='preact')
                net = tf.nn.relu(net)

            # convolution 3
            with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
                kernel = tf.get_variable(initializer=w_initializer,
                                         shape=[1, 1, channel, channel * 4],
                                         name='weights', regularizer=regularizer,
                                         collections=['pretrain', key])
                net = tf.nn.atrous_conv2d(value=net, filters=kernel, rate=holes,
                                          padding='SAME')
                biases = tf.get_variable(initializer=b_initializer,
                                         shape=channel * 4, name='biases',
                                         regularizer=regularizer,
                                         collections=['pretrain', key])
                net = tf.nn.bias_add(net, biases)
        return net, shortcut

    def block(self, net, name, unit, channel, is_train):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for i in range(unit):
                with tf.variable_scope('unit_' + str(i + 1), reuse=tf.AUTO_REUSE):
                    # block1 i=0 stride=1
                    if i == 0:
                        if name != 'block1':
                            net, shortcut = self.bottleneck(net, channel, is_train,
                                                       stride=2,
                                                       shortcut_conv=True)
                        else:
                            net, shortcut = self.bottleneck(net, channel, is_train,
                                                       stride=1,
                                                       shortcut_conv=True)
                    else:
                        net, shortcut = self.bottleneck(net, channel, is_train)
                net = tf.add(net, shortcut)

        return net

    def build_entity_net(self, inputs_scale, is_train):
        # MLP处理标量信息
        with tf.variable_scope("entity_net", reuse=tf.AUTO_REUSE):
            # fc
            with tf.variable_scope('entity_fc', reuse=tf.AUTO_REUSE):
                entity_net_layer = [1024, 1024, 1024]
                entity_net = self.build_general_net(entity_net_layer, inputs_scale, is_train)

            with tf.variable_scope('entity_relu', reuse=tf.AUTO_REUSE):
                # 全连接层
                entity_net = tf.layers.dense(inputs=entity_net,
                                         units=1024,
                                         bias_initializer=tf.zeros_initializer(),
                                         activation=tf.nn.relu,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                                         trainable=is_train)

        return entity_net

    def build_scalar_net(self, inputs_scale, is_train):
        # MLP处理标量信息
        with tf.variable_scope("scalar_net", reuse=tf.AUTO_REUSE):
            # fc
            with tf.variable_scope('scalar_fc', reuse=tf.AUTO_REUSE):
                # 全连接层
                sclar_net_layer = [256, 256]
                sclar_net = self.build_general_net(sclar_net_layer, inputs_scale, is_train)

            with tf.variable_scope('scalar_relu', reuse=tf.AUTO_REUSE):
                # 全连接层
                sclar_net = tf.layers.dense(inputs=sclar_net,
                                         units=280,
                                         bias_initializer=tf.zeros_initializer(),
                                         activation=tf.nn.relu,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                                         trainable=is_train)
        return sclar_net

    def build_general_net(self, layer_number, input, is_train):
        for i in range(len(layer_number)):
            with tf.variable_scope(str(layer_number[i])):
                g_net = tf.layers.dense(input, units=layer_number[i], activation=None, use_bias=True,
                                        bias_initializer=tf.zeros_initializer(),
                                        bias_regularizer=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                                        trainable=is_train)
                n_net = self.layer_normalize(g_net)
        return n_net

    def layer_normalize(self, net):
        g_net = tf.layers.batch_normalization(inputs=net, name='normalization')
        return g_net

    def build_spatial_net(self, input_spatial, w_initializer, b_initializer, regularizer, key, is_train):
        # convolution 1
        with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(initializer=w_initializer,
                                     shape=[3, 3, self.Channel, 64],  # TODO建议改成 [3,3,
                                     name='weights', regularizer=regularizer,
                                     collections=['pretrain', key])
            net = tf.nn.conv2d(input=input_spatial, filter=kernel,
                               strides=[1, 2, 2, 1], padding='SAME')
            biases = tf.get_variable(initializer=b_initializer, shape=64,
                                     name='biases', regularizer=regularizer,
                                     collections=['pretrain', key])
            net = tf.nn.bias_add(net, biases)
            net = tf.nn.max_pool(value=net, ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')
        for i in range(self.block_num):
            net = self.block(net, 'block' + str(i + 1), self.UNITS[self.resnet_type][i],
                             self.CHANNELS[i], is_train)
        net = tf.layers.batch_normalization(inputs=net, axis=-1,
                                            training=is_train, name='postnorm')
        net = tf.nn.relu(net)
        h, w = net.shape[1:3]
        net = tf.nn.avg_pool(value=net, ksize=[1, h, w, 1],
                             strides=[1, 1, 1, 1], padding='VALID')
        # flatten = tf.contrib.layers.flatten(net)
        return net

    def build_Anet(self, input_spatial, inputs_entity, inputs_scalar, is_train, name):
        # with tf.device(self._device_str):
        with tf.variable_scope(name):
            key = tf.GraphKeys.GLOBAL_VARIABLES
            with tf.variable_scope(self.resnet_type, reuse=tf.AUTO_REUSE):
                # define initializer for weights and biases
                w_initializer = tf.contrib.layers.xavier_initializer()
                b_initializer = tf.zeros_initializer()
                regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            # 空间信息网络
            spatial_con_net = self.build_spatial_net(input_spatial, w_initializer, b_initializer, regularizer, key, is_train)
            # 算子信息网络
            entity_mlp_net = self.build_entity_net(inputs_entity, is_train)
            #标量信息网络
            scalar_mlp_net = self.build_scalar_net(inputs_scalar, is_train)

            # 张量拼接                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          = tf.squeeze(spatial_con_net, axis=(1, 2))  # 首先对resnet两个没用的尺寸信息降维
            union_net = tf.concat([spatial_con_net, entity_mlp_net, scalar_mlp_net], 1)  # 和mlp的数据进行拼接
            union_net = tf.expand_dims(union_net, axis=1)  # 升维成为lstm的格式
            union_net = tf.expand_dims(union_net, axis=3)
            # union_net = tf.unstack(union_net, 1, 1)  # 按照时间步展开

            # 定义LSTM
            # with tf.device(self._device_str):
            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_n_hidden_units, forget_bias=1.0, state_is_tuple=True)
            # init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)  # 初始化全零 state

            # define the last convolution layer
            with tf.variable_scope('con_out1', reuse=tf.AUTO_REUSE):
                kernel = tf.get_variable(initializer=w_initializer,
                                         shape=[1, 1, 1, 1], name='weights',
                                         regularizer=regularizer,
                                         collections=['non_pretrain', key])
                net = tf.nn.conv2d(input=union_net, filter=kernel,
                                   strides=[1, 1, 1, 1], padding='VALID')
                biases = tf.get_variable(initializer=b_initializer, shape=1,
                                         name='biases', regularizer=regularizer,
                                         collections=['non_pretrain', key])
                net = tf.nn.bias_add(net, biases)
                net = tf.squeeze(net, axis=(1, 3))

            # define the prob of the actions_type
            with tf.variable_scope('actions_type', reuse=tf.AUTO_REUSE):
                action_type_net_layer = [512, 512]
                action_type = self.build_general_net(action_type_net_layer, net, is_train)
                action_type = tf.layers.batch_normalization(inputs=action_type, name='normalization')
                action_type = tf.layers.dense(
                    inputs=action_type,
                    units=self.action_type_size,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=is_train
                )

            # define the prob of the my_id
            with tf.variable_scope('my_id', reuse=tf.AUTO_REUSE):
                my_id_net_layer = [512, 512]
                my_id = self.build_general_net(my_id_net_layer, net, is_train)
                my_id = tf.layers.batch_normalization(inputs=my_id, name='normalization')
                my_id = tf.layers.dense(
                    inputs=my_id,
                    units=self.my_id_size,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=is_train
                )

            # define the prob of the obj_id
            with tf.variable_scope('obj_id', reuse=tf.AUTO_REUSE):
                obj_id_net_layer = [512, 512]
                obj_id = self.build_general_net(obj_id_net_layer, net, is_train)
                obj_id = tf.layers.batch_normalization(inputs=obj_id, name='normalization')
                obj_id = tf.layers.dense(
                    inputs=obj_id,
                    units=self.obj_id_size,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=is_train
                )

            # define the prob of the move_x
            with tf.variable_scope('move_x', reuse=tf.AUTO_REUSE):
                move_x_net_layer = [512, 512]
                move_x = self.build_general_net(move_x_net_layer, net, is_train)
                move_x = tf.layers.batch_normalization(inputs=move_x, name='normalization')
                move_x = tf.layers.dense(
                    inputs=move_x,
                    units=self.img_width,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=is_train
                )
                # define the prob of the move_y
            with tf.variable_scope('move_y', reuse=tf.AUTO_REUSE):
                move_y_net_layer = [512, 512]
                move_y = self.build_general_net(move_y_net_layer, net, is_train)
                move_y = tf.layers.batch_normalization(inputs=move_y, name='normalization')
                move_y = tf.layers.dense(
                    inputs=move_y,
                    units=self.img_high,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='net',
                    trainable=is_train
                )

                # actions_score = tf.squeeze(actions_net, axis=(1, 2))
                # acts_prob = tf.nn.softmax(actions_score)

            action_type_prob = tf.nn.softmax(action_type)
            my_id_prob = tf.nn.softmax(my_id)
            obj_id_prob = tf.nn.softmax(obj_id)
            x_prob = tf.nn.softmax(move_x)
            y_prob = tf.nn.softmax(move_y)
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
            return action_type_prob, my_id_prob, obj_id_prob, x_prob, y_prob, params


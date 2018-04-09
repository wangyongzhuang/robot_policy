import tensorflow as tf

class cnn():
    def __init__(self, flag):
        self.flag = flag
        self.prepare_weights()
        self.get_feat()

    def prepare_weights(self, init=tf.contrib.layers.xavier_initializer()):
        # TBD,[3,3,3,32], relu, mp, [3,3,32,64], relu, mp, [3,3,64,128], relu, [20*15*128, 2048], relu, [2048, 256], relu
        self.w_conv_1 = tf.get_variable(name='w_conv_1', shape=[3, 3, 3, 32], initializer=init)
        self.b_conv_1 = tf.get_variable(name='b_conv_1', shape=[32])

        self.w_conv_2 = tf.get_variable(name='w_conv_2', shape=[3, 3, 32, 64], initializer=init)
        self.b_conv_2 = tf.get_variable(name='b_conv_2', shape=[64])

        self.w_conv_3 = tf.get_variable(name='w_conv_3', shape=[3, 3, 64, 128], initializer=init)
        self.b_conv_3 = tf.get_variable(name='b_conv_3', shape=[128])

        self.w_fc_1 = tf.get_variable(name='w_fc_1', shape=[25*40*128, 2048], initializer=init)
        self.b_fc_1 = tf.get_variable(name='b_fc_1', shape=[2048])

        self.w_fc_2 = tf.get_variable(name='w_fc_2', shape=[2048, 256], initializer=init)
        self.b_fc_2 = tf.get_variable(name='b_fc_2', shape=[256])

        self.w_q    = tf.get_variable(name='w_q_2', shape=[256, self.flag.mov_num * 4 + 2], initializer=init)

    def conv2d(self, d, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(d, w, strides=[1, 1, 1, 1], padding='SAME'), b))
    def mp2d(self, d, k_size=2):
        return tf.nn.max_pool(d, ksize=[1, k_size, k_size, 1], strides=[1, k_size, k_size, 1], padding='SAME')
    def fc2d(self, d, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(d, w), b))

    def get_feat(self):
        self.data = tf.placeholder(name='data', shape=[800, 500, 3], dtype=tf.float32)

        self.conv_1 = self.conv2d(tf.expand_dims(self.data, 0), self.w_conv_1, self.b_conv_1)
        self.mp_1   = self.mp2d(self.conv_1, k_size=5)

        self.conv_2 = self.conv2d(self.mp_1, self.w_conv_2, self.b_conv_2)
        self.mp_2   = self.mp2d(self.conv_2, k_size=2)

        self.conv_3 = self.conv2d(self.mp_2, self.w_conv_3, self.b_conv_3)
        self.mp_3   = self.mp2d(self.conv_3, k_size=2)
        
        self.mp_3   = tf.reshape(self.mp_3, [1, 25*40*128])
        self.fc_1   = self.fc2d(self.mp_3, self.w_fc_1, self.b_fc_1)

        self.fc_2   = self.fc2d(self.fc_1, self.w_fc_2, self.b_fc_2)

        self.q_val  = tf.matmul(self.fc_2, self.w_q)

        self.logits = tf.reshape(self.q_val, [2, self.flag.mov_num * 2 + 1])

        self.q_p    = tf.concat([tf.nn.softmax(self.logits[:,:self.flag.mov_num]), tf.nn.softmax(self.logits[:,self.flag.mov_num:self.flag.mov_num*2]), tf.sigmoid(self.logits[:,-1:])], -1)
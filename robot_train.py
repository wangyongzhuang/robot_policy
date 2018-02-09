import tensorflow as tf
import numpy as np

from robot_config import *
from robot_environ import *
from robot_model import cnn
from robot_client import *
from robot_agent import *

import pdb

pygame.init()
screen = pygame.display.set_mode((800, 500), 0, 32)
flag = config()
draw_init(screen) # Yelly deleted the parameter 'bars' as robot_client has this variable already

# model
with tf.variable_scope('cnn_1') as scope:
    cnn_1 = cnn(flag)
with tf.variable_scope('cnn_2') as scope:
    cnn_2 = cnn(flag)

# reward
r_1 = tf.placeholder(dtype=tf.float32, shape=[2, 2*flag.mov_num+1])
r_2 = tf.placeholder(dtype=tf.float32, shape=[2, 2*flag.mov_num+1])

# loss, policy-based
# without not shoot reward
loss = tf.reduce_mean((-1 * tf.log(cnn_1.q_p + 1e-8) * r_1) + (-1 * tf.log(cnn_2.q_p + 1e-8) * r_2))


# optimize
global_step   = tf.Variable(0, trainable=False)
learning_rate = tf.train.polynomial_decay(learning_rate=flag.learning_rate / flag.lr_decay, global_step=global_step, decay_steps=flag.decay_steps, power=2.0, end_learning_rate=flag.learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.8, beta2=0.999, epsilon=1e-8).minimize(loss, global_step=global_step)


# init
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())


# train
info_1, info_2, map_img = get_init()
for global_step in range(flag.steps):
    # Yelly modification:
    # round is over when two robots of one side both have 0 HP
    if global_step%300==0 or (info_1[0][2] == 0 and info_1[1][2] == 0) or (info_2[0][2] == 0 and info_2[1][2] == 0):
        info_1, info_2, map_img = get_init()
    act_1_p, act_2_p = sess.run([cnn_1.q_p, cnn_2.q_p], feed_dict={cnn_1.data:map_img, cnn_2.data:map_img})

    # environ and optimize
    if global_step%10==0:
        info_1, info_2, r1, r2, map_img_new, state_1, state_2 = agent(flag, info_1, info_2, act_1_p, act_2_p, raw_map_img, policy='MAX')
    else:
        info_1, info_2, r1, r2, map_img_new, state_1, state_2 = agent(flag, info_1, info_2, act_1_p, act_2_p, raw_map_img, policy='RANDOM')

    draw_state(screen, bars, state_1, state_2, info_1, info_2)

    _, l = sess.run([optimizer, loss], feed_dict={cnn_1.data:map_img, cnn_2.data:map_img, r_1:r1, r_2:r2})

    print 'Itr_%d loss: %0.3f'%(global_step, l)

    draw_info(screen, bars, info_1, info_2)
    pygame.time.delay(100)

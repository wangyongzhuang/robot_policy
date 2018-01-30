import tensorflow as tf
import numpy as np
import json as js
import scipy.io
import os
import random
import pdb
from nlgeval import compute_metrics
from collections import defaultdict

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4

from im2sents_config import config
from im2sents_model  import model
from im2sents_data_tf   import dataset
from im2sents_test   import *

# config
flag = config()

# data
data = dataset(flag)
flag.set_label_size(data.patch)
flag.set_data_size(data.patch)
print 'dataset init done.'


# model
model = model(flag, data)
print 'model init done.'

# init
sess = tf.Session(config=tf_config)
#sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=100)
ckpt  = tf.train.get_checkpoint_state(flag.ckpt_dir)
#ckpt  = tf.train.get_checkpoint_state('/home/wyz/log/ckpt_test_Bleu_4_best')

'''
ckpt_tmp = ckpt.all_model_checkpoint_paths[-3]
saver.restore(sess, ckpt_tmp)
test_single(model, data, flag, sess, select=''MAX')
print 'Model ckpt: ', ckpt_tmp
scores = compute_metrics(hypothesis='result/res.txt', references=['result/ref1.txt', 'result/ref2.txt', 'result/ref3.txt', 'result/ref4.txt', 'result/ref5.txt'], no_skipthoughts=True, no_glove=True)
'''

scores_best = defaultdict(int)
scores_cider = defaultdict(int)

for i in range(len(ckpt.all_model_checkpoint_paths)):
    ckpt_tmp = ckpt.all_model_checkpoint_paths[i]
    saver.restore[ckpt_tmp]
    print 'Test ckpt:', ckpt_tmp

    test_single(model, data, flag, sess, select='MAX')
    scores = compute_metrics(hypothesis='result/res.txt', references=['result/ref1.txt', 'result/ref2.txt', 'result/ref3.txt', 'result/ref4.txt', 'result/ref5.txt'], no_skipthoughts=True, no_glove=True)

    for k in scores.keys():
        if scores[k] > scores_best[k]:
            scores_best[k] = scores[k]
        scores_best['ckpt'] = ckpt_tmp
    if scores['CIDEr'] > scores_cider['CIDEr']:
        for k in scores.keys():
            scores_cider[k] = scores[k]
        scores_cider['ckpt'] = ckpt_tmp
        print 'new state-of-art:', ckpt_tmp, scores_cider

print 'Best: ', scores_best
print 'CIDEr:', scores_cider
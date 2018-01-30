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


# reward, v_tmp
if flag.training_approach == 'FLST':
    reward_sents       = tf.placeholder(dtype=tf.float32, shape=[flag.batch_size])
    reward_sents_tiled = tf.tile(tf.expand_dims(tf.expand_dims(reward_sents, 0), -1), [flag.sentence_length, 1, flag.label_0_size])

    reward_ratio_down  = tf.tile(tf.expand_dims(tf.expand_dims(tf.constant([flag.reward_ratio**_ for _ in range(flag.sentence_length)]), -1), -1), [1, flag.batch_size, flag.label_0_size])
    reward_ratio_up    = tf.tile(tf.expand_dims(tf.expand_dims(tf.constant([flag.reward_ratio**(flag.sentence_length - 1 - _) for _ in range(flag.sentence_length)]), -1), -1), [1, flag.batch_size, flag.label_0_size])
    mask_rl = tf.reverse(tf.cumsum(reward_ratio_up, axis=0, reverse=True), axis=[0]) * reward_sents_tiled
elif flag.training_approach == 'SCST':
    reward_sents       = tf.placeholder(dtype=tf.float32, shape=[flag.batch_size])
    reward_sents_tiled = tf.tile(tf.expand_dims(tf.expand_dims(reward_sents, 0), -1), [flag.sentence_length, 1, flag.label_0_size])
    mask_rl = reward_sents_tiled


# loss
loss_ori = tf.reduce_mean(-1.0 * model.label_0_masked * tf.log(tf.nn.softmax(model.logits_0_masked)+1e-8)) * flag.label_0_size + flag.weight_decay * model.l2_loss
loss = loss_ori
#loss_rl  = tf.reduce_mean(-1.0 * mask_rl * tf.log(tf.nn.softmax(model.logits_0_masked)+1e-8)) * flag.label_0_size + flag.weight_decay * model.l2_loss
if flag.training_approach == 'SCST' or flag.training_approach == 'FLST':
    loss_rl  = tf.reduce_mean(-1.0 * mask_rl * model.label_0_masked * tf.log(tf.nn.softmax(model.logits_0_masked)+1e-8)) * flag.label_0_size + flag.weight_decay * model.l2_loss
    loss = loss_rl
    tf.summary.scalar('loss_rl', flag.loss_scale * loss_rl)


# recall rate
recall_same_0 = tf.abs(tf.argmax(model.label_0, -1)+1 - (tf.argmax(model.logits_0, -1)+1) * tf.cast(model.sent_mask_0[:, :, 0], tf.int64)) < 1
recall_rate_0 = tf.reduce_sum(tf.cast(recall_same_0, tf.float32)) / tf.reduce_sum(tf.cast(model.sent_mask_0[:, :, 0], tf.float32))
tf.summary.scalar('recall_rate_0', recall_rate_0)

# train & optimize
global_step   = tf.Variable(0, trainable=False)
learning_rate = tf.train.polynomial_decay(learning_rate=flag.learning_rate / flag.lr_decay, global_step=global_step, decay_steps=flag.data_size * flag.epoch_decay/flag.batch_size, power=2.0, end_learning_rate=flag.learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.8, beta2=0.999, epsilon=1e-8).minimize(loss, global_step=global_step)

'''
learning_rate = tf.placeholder(shape=[1], dtype=tf.float32)
optimizer = tf.train.AdamOptimizer(learning_rate[0], beta1=0.8, beta2=0.999, epsilon=1e-8).minimize(loss, global_step=global_step)
tf.summary.scalar('learning_rate', learning_rate[0])
tf.summary.scalar('learning_rate', learning_rate[0])
'''


# init
sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())

# summary and saver
fs = os.listdir(flag.sum_dir)
for f in fs:
    os.remove(flag.sum_dir + '/' + f)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(flag.sum_dir, sess.graph)

saver = tf.train.Saver(max_to_keep=100)
ckpt  = tf.train.get_checkpoint_state(flag.ckpt_dir)
if flag.is_finetune:
    saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])



# score
scores_z = defaultdict(list)
score_Bleu_1 = 0.0
score_Bleu_4 = 0.0
score_METEOR = 0.0
score_ROUGLE_L = 0.0
score_CIDER = 0.0
score_epoch = 0
# coefficient
#lr = flag.learning_rate / flag.lr_decay
l_pre = 0
l_rl_pre = 0
s_max_pre = 0
s_scst_pre = 0
early_stop  = 0
global_step = -1

# train
while early_stop < 5/flag.epoch_val:
    global_step += 1
    fr = min(0.25 * global_step * flag.batch_size / (flag.epoch_decay * flag.data_size), 0.25)

    if flag.training_approach == 'SCST' or flag.training_approach == 'FLST':
        # load data and predict score
        feat_tmp, inds_tmp, word_refs_tmp = sess.run([data.samp_train_feat, data.samp_train_inds, data.samp_train_refs])
        scores_sub, scores_scst, scores_max = train_test_final(model, data, flag, sess, feat_tmp, inds_tmp, word_refs_tmp, fr, score_type='SUM')
    
        _, s, l_rl, l, r_0 = sess.run([optimizer, merged, loss, loss_ori, recall_rate_0], feed_dict={reward_sents:scores_sub, model.feat_train:feat_tmp,   model.inds_train:inds_tmp, model.refs_train:word_refs_tmp, model.fire_rate:[fr]})
    elif flag.training_approach == 'XE':
        _, s, l_rl, l, r_0 = sess.run([optimizer, merged, loss, loss_ori, recall_rate_0], feed_dict={model.fire_rate:[fr]})


    # summary
    epoch = 1.0 * global_step * flag.batch_size / flag.data_size
    l_pre = 0.99 * l_pre + 0.01 * l
    l_rl_pre = 0.99 * l_rl_pre + 0.01 * l_rl
    s_max_pre = 0.99 * s_max_pre + 0.01 * sum(scores_max) / flag.batch_size
    s_scst_pre = 0.99 * s_scst_pre + 0.01 * sum(scores_scst) / flag.batch_size

    if s_max_pre < 0.5 * sum(scores_max) / flag.batch_size:
        s_max_pre = sum(scores_max) / flag.batch_size
    if s_scst_pre < 0.5 * sum(scores_scst) / flag.batch_size:
        s_scst_pre = sum(scores_scst) / flag.batch_size
    if l_pre > 1.5*l or l_pre < 0.5 * l:
        l_pre = l
    if l_rl_pre > 1.5*l_rl or l_rl_pre < 0.5 * l_rl:
        l_rl_pre = l_rl

    # show the result
    if global_step% 10 == 0:
        print 'Epo_%0.2f_Itr_%d loss:%0.3f l_mean:%0.3f l_rl:%0.3f s_scst:%0.3f s_max:%0.3f recall:%0.3f'%(epoch, global_step, flag.loss_scale * l, flag.loss_scale * l_pre, flag.loss_scale * l_rl_pre, s_scst_pre, s_max_pre, r_0)
        #print 'score_mask:',scores_sub[:10]

    # val and save model
    if global_step% int(flag.epoch_val * flag.data_size / flag.batch_size) == 0:# and global_step>0:
        # val
        #score = test_final(model, data, flag, sess)
        score = test_final(model, data, flag, sess)

        if score['Bleu_1']>score_Bleu_1 or score['Bleu_4']>score_Bleu_4 or score['METEOR']>score_METEOR or score['ROUGE_L']>score_ROUGLE_L or score['CIDEr']>score_CIDER:
            early_stop = 0
            if score['Bleu_1'] > 0.7:
                saver.save(sess, flag.ckpt_dir + '/model_%s_epoch_%0.1f_%d.ckpt'%(flag.dataset, epoch, flag.loss_scale * l_pre))
                print 'save ckpt: model_%s_epoch_%0.1f_%d.ckpt'%(flag.dataset, epoch, flag.loss_scale * l_pre)
            score_epoch    = epoch
            score_Bleu_1   = max(score['Bleu_1'], score_Bleu_1)
            score_Bleu_4   = max(score['Bleu_4'], score_Bleu_4)
            score_METEOR   = max(score['METEOR'], score_METEOR)
            score_ROUGLE_L = max(score['ROUGE_L'], score_ROUGLE_L)
            score_CIDER    = max(score['CIDEr'], score_CIDER)
            scores_z['epoch'].append(epoch)
            scores_z['Bleu_1'].append(score['Bleu_1'])
            scores_z['Bleu_4'].append(score['Bleu_4'])
            scores_z['METEOR'].append(score['METEOR'])
            scores_z['ROUGE_L'].append(score['ROUGE_L'])
            scores_z['CIDEr'].append(score['CIDEr'])
            if len(scores_z['epoch']) > 10:
                for k in scores_z.keys():
                    scores_z[k].pop(0)

        elif score_Bleu_1 > 0.7:
            early_stop = early_stop + 1
        print 'Targ B-1:0.811 B-4:0.386 METEOR:0.277 ROUGE_L:0.587 CIDEr:1.254'
        print 'Best B-1:%0.3f B-4:%0.3f METEOR:%0.3f ROUGE_L:%0.3f CIDEr:%0.3f Epoch:%0.1f'%(score_Bleu_1, score_Bleu_4, score_METEOR, score_ROUGLE_L, score_CIDER, score_epoch)
        print 'Now  B-1:%0.3f B-4:%0.3f METEOR:%0.3f ROUGE_L:%0.3f CIDEr:%0.3f Epoch:%0.1f'%(score['Bleu_1'], score['Bleu_4'], score_METEOR, score['ROUGE_L'], score['CIDEr'], epoch)
    
    if early_stop > 5 / flag.epoch_val:
        if lr == flag.learning_rate:
            print 'early stop'
            break
        else:
            early_stop = -5 / flag.epoch_val
            saver.restore(sess, ckpt.model_checkpoint_path)
            lr = flag.learning_rate

for k in scores_z.keys():
    print scores_z[k]
js.dump(scores_z, open('scores_z.json','w'))
    

score = test(model, data, flag, sess)
print 'score_best B-1:%0.3f B-4:%0.3f METEOR:%0.3f ROUGE_L:%0.3f CIDEr:%0.3f Epoch:%0.1f'%(score_Bleu_1, score_Bleu_4, score_METEOR, score_ROUGLE_L, score_CIDER, score_epoch)
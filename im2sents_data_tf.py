import tensorflow as tf
import numpy as np
import scipy.io
import json as js
import random

class dataset():
    def __init__(self, flag):
        self.flag = flag
        self.patch = js.load(open(self.flag.data_dir + '/data_z_final/patch_coco.json', 'r'))
        self.label2word   = js.load(open(self.flag.data_dir + '/data_z_final/label2word_coco.json', 'r'))
        self.word2label   = js.load(open(self.flag.data_dir + '/data_z_final/word2label_coco.json', 'r'))

        self.load_split()
        self.get_samp()


    def gen_train(self):
        #feats = scipy.io.loadmat(self.flag.data_dir + '/data_z_final/feats/feats_train_00.mat')['feats']
        for i in range(len(self.patch['train'])):
            # feat
            '''
            if i%5000==0 and i>0:
                feats = scipy.io.loadmat(self.flag.data_dir + '/data_z_final/feats/feats_train_%02d.mat'%(i/5000))['feats']
            feat = feats[self.patch['train'][i]['feat_idx'][1]]
            '''
            feat = scipy.io.loadmat('/home/wyz/data/coco/data_z/feats/'+ self.patch['train'][i]['file_name'].split('.')[0]+'.mat')['feat']
            feat = feat.reshape([self.flag.feat_num-1, self.flag.feat_size])
            feat_ave = np.sum(feat, axis=0) / (self.flag.feat_num-1)
            
            feat = np.concatenate([feat_ave[np.newaxis, :], feat], axis=0)

            # refs
            refs = self.patch['train'][i]['references']
            if len(refs)>5:
                refs = refs[:5]
            elif len(refs)<5:
                refs = refs + refs[:5-len(refs)]

            # label
            for j in range(len(self.patch['train'][i]['sents_id'])):
                label_0 = self.patch['train'][i]['labels'][j]
                yield feat, label_0, refs


    def gen_val(self):
        #feats = scipy.io.loadmat(self.flag.data_dir + '/data_z_final/feats/feats_val_00.mat')['feats']
        for i in range(len(self.patch['val'])):
            # feat
            '''
            feat = feats[i]
            '''
            feat = scipy.io.loadmat('/home/wyz/data/coco/data_z/feats/'+ self.patch['train'][i]['file_name'].split('.')[0]+'.mat')['feat']
            feat = feat.reshape([self.flag.feat_num-1, self.flag.feat_size])
            feat_ave = np.sum(feat, axis=0) / (self.flag.feat_num-1)
            
            feat = np.concatenate([feat_ave[np.newaxis, :], feat], axis=0)

            # refs
            refs = self.patch['val'][i]['references']
            if len(refs)>5:
                refs = refs[:5]
            elif len(refs)<5:
                refs = refs + refs[:5-len(refs)]

            yield feat, refs


    def load_split(self):
        data_train = tf.data.Dataset.from_generator(self.gen_train, output_types=(tf.float32, tf.int64, tf.string), output_shapes=(tf.TensorShape([self.flag.feat_num, self.flag.feat_size]), tf.TensorShape([self.flag.sentence_length+1]), tf.TensorShape([5])))
        self.data_train = data_train.repeat().shuffle(buffer_size=3 * self.flag.batch_size).batch(self.flag.batch_size)
        self.iter_train = self.data_train.make_one_shot_iterator()

        data_val = tf.data.Dataset.from_generator(self.gen_val, output_types=(tf.float32, tf.string), output_shapes=(tf.TensorShape([self.flag.feat_num, self.flag.feat_size]), tf.TensorShape([5])))
        self.data_val = data_val.repeat().batch(self.flag.batch_size)
        self.iter_val = self.data_val.make_one_shot_iterator()

        data_val_single = tf.data.Dataset.from_generator(self.gen_val, output_types=(tf.float32, tf.string), output_shapes=(tf.TensorShape([self.flag.feat_num, self.flag.feat_size]), tf.TensorShape([5])))
        self.data_val_single = data_val_single.repeat().batch(1)
        self.iter_val_single = self.data_val_single.make_one_shot_iterator()

    def get_samp(self):
        self.samp_train_feat, self.samp_train_inds, self.samp_train_refs = self.iter_train.get_next()

        self.samp_val_feat, self.samp_val_refs = self.iter_val.get_next()

        self.samp_val_single_feat, self.samp_val_single_refs = self.iter_val_single.get_next()
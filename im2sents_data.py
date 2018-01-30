import tensorflow as tf
import json as js
import numpy as np
import random
import scipy.io
import h5py
from collections import defaultdict
import pdb
    
class dataset():
    def __init__(self, flag, dataset='flickr8k', patch_ready=False):
        self.flag = flag
        self.patch_ready = patch_ready
        self.load(dataset)


    # load data and split
    def load(self, dataset='flickr8k'):
        self.data = js.load(open(self.flag.data_dir+'/'+dataset+'.json', 'r'))['images']
        self.load_feats(dataset=dataset)
        self.patch = js.load(open(self.flag.data_dir+'/patch.json', 'r'))
        self.load_split()

        self.word2label_2  = js.load(open(self.flag.data_dir+'/word2label_2.json','r'))
        self.word2label    = js.load(open(self.flag.data_dir+'/word2label.json','r'))
        self.label2word    = js.load(open(self.flag.data_dir+'/label2word.json','r'))


    def load_feats(self, dataset='flickr8k'):
        feats = h5py.File(self.flag.data_dir + '/'+dataset+'_res_feats_hdf5.mat')['feats']
        feats_shape = feats.shape
        feats = np.transpose(np.reshape(feats, [feats_shape[0], feats_shape[1], feats_shape[2]*feats_shape[3]]), [0, 2, 1])
        feats_ave = np.sum(feats, axis=1)/(feats_shape[2]*feats_shape[3])
        self.feats = np.concatenate([feats_ave[:, np.newaxis, :], feats], axis=1)


    def load_split(self):
        data_test  = []
        data_val   = []
        data_train = []
        for d in self.data:
            if d['split']=='test':
                data_test.append(d)
            elif d['split']=='val':
                data_val.append(d)
            else:
                data_train.append(d)
        self.data_test  = data_test
        self.data_val   = data_val
        self.data_train = data_train


    def get_sample(self, size=-1):
        if size<0:
            sample = random.sample(self.patch['train'], self.flag.batch_size)
        else:
            sample = random.sample(self.patch['train'], size)

        sample_feat = [self.feats[_['imgid'],:,:].tolist() for _ in sample]
        sample_inds = []
        sample_label = []
        for samp in sample:
            if len(samp['label_2']) > self.flag.sentence_length:
                sample_inds.append([0] + samp['label'][:self.flag.sentence_length])
                sample_label.append([[0, 0]] + samp['label_2'][:self.flag.sentence_length])
            else:
                sample_inds.append([0] + samp['label'] + [0 for i in range(self.flag.sentence_length - len(samp['label']))])
                sample_label.append([[0, 0]] + samp['label_2'] + [[0, 0] for i in range(self.flag.sentence_length - len(samp['label']))])
        del sample

        return sample_feat, sample_inds, sample_label
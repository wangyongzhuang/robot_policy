import json as js
import numpy as np
import scipy.io
from collections import defaultdict
import pdb

data_name = 'coco'
data_dir  = '/home/wyz/data/'+data_name
target_dir = '/home/wyz/data/'+ data_name + '/data_z_final'


data_train = js.load(open(data_dir + '/annotations/captions_train2017.json', 'r'))
data_val   = js.load(open(data_dir + '/annotations/captions_val2017.json', 'r'))
data_test  = js.load(open(data_dir + '/annotations/image_info_test2017.json', 'r'))

# get tokens
def get_tokens(sent):
    chars_dict = {'0':'zero', '1':'one', '2':'two', '3':'three', '4':'four', '5':'five', '6':'six', '7':'seven', '8':'eight', '9':'nine', '"':' ', ',':' ', '.':' ', '-':' ', '_':' ', '#':' ', '&':' ', '/':' ', '!':' '}
    chars_dict_2 = {"'s":' is', "'":' '}
    chars = chars_dict.keys()
    chars_2 = chars_dict_2.keys()
    tokens = sent.lower().split('\n')[0].split(' ')
    res = []
    for token in tokens:
        # drop
        for char in chars:
            while char in token:
                idx = token.index(char)
                token = token[:idx] + chars_dict[char] + token[idx+len(char):]
        for char in chars_2:
            while char in token:
                idx = token.index(char)
                token = token[:idx] + chars_dict_2[char] + token[idx+len(char):]
        for t in token.split(' '):
            if len(t) > 1:
                res.append(t)
            elif t in ['a']:
                res.append(t)
    tokens = res
    return res
        


# create vocabulary
def create_vocab(data, word_thres=5):
    # vocab
    vocab = defaultdict(int)

    anns = data['annotations']
    for ann in anns:
        tokens = get_tokens(ann['caption'])
        tokens.append('.')
        for token in tokens:
            vocab[token] += 1

    # drop
    print len(vocab)
    for k in vocab.keys():
        if vocab[k] < word_thres:
            #print 'pop ', k, len(vocab)
            vocab.pop(k)
    voc_k = vocab.keys()
    voc_k.remove('.')
    
    word2label = {'.':0}
    label2word = {0:'.'}
    for i,k in enumerate(voc_k):
        word2label[k] = i+1
        label2word[i+1] = k


    print 'create vocab, word2label, label2word done. Size: %d'%len(vocab)
    return vocab, word2label, label2word

vocab, word2label, label2word = create_vocab(data_train)
js.dump(vocab,      open(target_dir + '/vocab_coco.json', 'w'))
js.dump(word2label, open(target_dir + '/word2label_coco.json', 'w'))
js.dump(label2word, open(target_dir + '/label2word_coco.json', 'w'))


# get synonyms
def get_synonym(sents, voc_k, cbow_size=5):
    cbow_content = defaultdict(list)
    for tokens in sents:
        for i in range(cbow_size, len(tokens)-cbow_size):
            cbow_content[' '.join([tokens[i-2], tokens[i-1], tokens[i+1], tokens[i+2]])].append(tokens[i])

    # find synonym
    for c in cbow_content.keys():
        tmp_c = list(set(cbow_content[c]))
        if len(tmp_c) == 1:
            cbow_content.pop(c)
        else:
            cbow_content[c] = tmp_c

    if len(cbow_content)>0:
        for c in cbow_content.keys():
            print c,' : ', cbow_content[c]

    return cbow_content




def create_tfidf(data, vocab, word2label):
    tfidf = np.zeros(len(vocab), dtype=np.float)
    idf   = np.zeros(len(vocab), dtype=np.float)

    tfs = defaultdict(float)
    idfs = defaultdict(float)

    for ann in data['annotations']:
        tokens = get_tokens(ann['caption'])
        tokens.append('.')
        for token in tokens:
            tfs[token] += 1
        tokens = list(set(tokens))
        for token in tokens:
            idfs[token] += 1

    tfs_max = max(tfs.values())
    idfs_max = max(idfs.values())

    for k in vocab.keys():
        tfidf[word2label[k]] = (tfs[k] / tfs_max) * np.log(idfs_max / idfs[k])
        idf[word2label[k]]   = 1 + np.log10(idfs_max / idfs[k])


    print 'create tfidf, idf done.'
    return tfidf, idf

tfidf, idf = create_tfidf(data_train, vocab, word2label)
scipy.io.savemat(target_dir + '/tfidf_coco.mat', {'tfidf':tfidf})
scipy.io.savemat(target_dir + '/idf_coco.mat', {'idf':idf})




def crate_samp(imgs, voc_k, anns, word2label, sentence_length=20, split='train'):
    samples = {}
    feats_idx = 0
    feats = np.zeros(shape=[5000, 7, 7, 2048])
    for i,img in enumerate(imgs):
        samp = {}
        samp['img_id'] = img['id']
        samp['file_name'] = img['file_name']
        samp['file_path'] = data_dir + '/'+ split +'_2017/' + img['file_name']
        samp['feat_path'] = '/home/wyz/data/coco/data_z/feats/' + img['file_name'].split('.')[0] + '.mat'
        '''
        samp['feat_path'] = target_dir + '/feats/' + 'feats_' + split + '_%02d.mat'%(i/5000)
        samp['feat_idx']  = [i/5000, i%5000]

        feat = scipy.io.loadmat(data_dir + '/data_z/feats/' + img['file_name'].split('.')[0] + '.mat')['feat']
        feats[i%5000] = feat
        if (i+1)%5000==0:
            scipy.io.savemat(target_dir + '/feats/' + 'feats_' + split + '_%02d.mat'%(i/5000), {'feats':feats})
            feats = np.zeros(shape=[5000, 7, 7, 2048])
            print 'save mat: ' + target_dir + '/feats/' + 'feats_' + split + '_%02d.mat'%(i/5000)
        elif i+1==len(imgs):
            scipy.io.savemat(target_dir + '/feats/' + 'feats_' + split + '_%02d.mat'%(i/5000), {'feats':feats[:(i+1)%5000]})
            print 'save mat: ' + target_dir + '/feats/' + 'feats_' + split + '_%02d.mat'%(i/5000)
        '''

        samp['sents_id']   = []
        samp['references'] = []
        samp['labels']     = []

        samples[img['id']] = samp

    if split=='test':
        return samples.values()

    for i,ann in enumerate(anns):
        idx = ann['image_id']
        samples[idx]['sents_id'].append(ann['id'])
        samples[idx]['references'].append(ann['caption'].split('\n')[0].lower())

        tokens = get_tokens(ann['caption'])
        labels = [0]
        for token in tokens:
            if token in voc_k:
                labels.append(word2label[token])
        if len(labels)>sentence_length:
            labels = labels[:sentence_length] + [0]
        else:
            while len(labels)<=sentence_length:
                labels.append(0)
        
        samples[idx]['labels'].append(labels)

        if i%100==0:
            print split + ' %d/%d'%(i, len(anns))

    samples = samples.values()
    return samples



def create_patch(data_train, data_val, data_test, vocab, word2label):
    patch = {}
    
    voc_k = vocab.keys()
    patch['label_0_size'] = len(word2label)
    patch['train_num']    = len(data_train['annotations'])
    patch['val_num']      = len(data_val['annotations'])

    # train
    patch['train'] = []
    imgs = data_train['images']
    anns = data_train['annotations']
    f_train = open(target_dir + '/file_path_train_coco.txt', 'w')
    for img in imgs:
        f_train.write(target_dir + '/train2017/' + img['file_name'] + '\n')
    f_train.close()
    patch['train'] = crate_samp(imgs, voc_k, anns, word2label, sentence_length=20, split='train')
    print 'create patch for train done.'

    # val
    patch['val'] = []
    imgs = data_val['images']
    anns = data_val['annotations']
    f_val = open(target_dir + '/file_path_val_coco.txt', 'w')
    for img in imgs:
        f_val.write(target_dir + '/val2017/' + img['file_name'] + '\n')
    f_val.close()
    patch['val'] = crate_samp(imgs, voc_k, anns, word2label, sentence_length=20, split='val')
    print 'create patch for val done.'

    # test
    imgs = data_test['images']
    f_test = open(target_dir + '/file_path_test_coco.txt', 'w')
    for i,img in enumerate(imgs):
        f_test.write(target_dir + '/test2017/' + img['file_name'] + '\n')
        if i%100==0:
            print 'sample for test_%d/%d done.'%(i, len(imgs))
    f_test.close()
    patch['test'] = crate_samp(imgs, voc_k, anns, word2label, sentence_length=20, split='test')
    print 'create patch for test done.'

    return patch

vocab        = js.load(open(target_dir + '/vocab_coco.json', 'r'))
word2label   = js.load(open(target_dir + '/word2label_coco.json', 'r'))
lable2word   = js.load(open(target_dir + '/label2word_coco.json', 'r'))
print 'load vocab, w2l, w2l2, l2w done.'

patch = create_patch(data_train, data_val, data_test, vocab, word2label)
js.dump(patch, open(target_dir+'/patch_coco.json', 'w'))

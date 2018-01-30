import json as js
import numpy as np
import scipy.io
import h5py
from collections import defaultdict

data_name = 'flickr8k'
data_dir  = '/home/wangyongzhuang/data/'+data_name


data = js.load(open(data_dir+'/'+data_name+'.json', 'r'))['images']

def create_vocab(data, word_thres=5):
    '''
    create vocab and vocab_thres, <UNK> will contain all the words which freq below word_thres.
    '''
    # vocab
    vocab = defaultdict(int)
    for d in data:
        sents = d['sentences']
        for sent in sents:
            tokens = [_.lower() for _ in sent['tokens']]# do not contain '.'
            if '.' not in tokens:
                tokens.append('.')
            #raw    = sent['raw'].lower()# the sent end with "A dog ... pool."
            #tokens = raw.split(' ')
            for token in tokens:
                vocab[token] += 1

    # vocab_thres
    vocab_thres = defaultdict(int)
    for k in vocab.keys():
        if vocab[k] >= word_thres:
            vocab_thres[k] = vocab[k]
        else:
            vocab_thres['<UNK>'] += vocab[k]

    print 'create vocab, vocab_thres done.'
    return vocab, vocab_thres


def create_word_idx(vocab):
    '''
    create index2word, word2index, especially {'.':0} (or {'#':0})
    '''
    index2word = {}
    word2index = {}
    voc_k      = vocab.keys()
    if '.' in voc_k:
        voc_k.remove('.')
    index2word['0'] = '.'
    word2index['.'] = 0

    idx = 1
    for k in voc_k:
        index2word[idx] = k
        word2index[k]        = idx
        idx += 1

    print 'create word2index, index2word done.'
    return word2index, index2word


def create_tfidf(data, vocab):
    '''
    create tfidf of the word
    '''
    tfidf = {}
    for w in vocab.keys():
        print w, vocab[w]
        freq = 0.
        txt_num = 0
        for d in data:
            sents = d['sentences']
            for sent in sents:
                if w in sent['tokens']:
                    freq += 1.0 * sent['tokens'].count(w) / len(sent['tokens'])
                    txt_num += 1
        tfidf[w] = (freq/txt_num) * np.log(len(data)*5./txt_num + 1)

    print 'create tfidf done.'
    return tfidf



vocab, vocab_thres     = create_vocab(data)
word2index, index2word = create_word_idx(vocab_thres)

js.dump(vocab, open(data_dir+'/vocab.json', 'w'))
js.dump(vocab_thres, open(data_dir+'/vocab_thres.json', 'w'))
js.dump(word2index, open(data_dir+'/word2index.json', 'w'))
js.dump(index2word, open(data_dir+'/index2word.json', 'w'))


# cbow
def get_synonym(sents, vocab):
    # TBD using CBOW
    cbow_num = 3
    cbow_content = defaultdict(list)
    vocab_list = vocab.keys()
    for sent in sents:
        token = sent['tokens']
        for i,t in enumerate(token):
            if t not in vocab_list:
                token[i] = '<UNK>'
        for i in range(cbow_num, len(token)-cbow_num):
            if vocab[token[i]] >= 5:
                cbow_content[''.join([token[i-2], token[i-1], token[i+1], token[i+2]])].append(token[i])

    for c in cbow_content.keys():
        if len(list(set(cbow_content[c]))) == 1:
            cbow_content.pop(c)
        else:
            cbow_content[c] = list(set(cbow_content[c]))

    if len(cbow_content)>0:
        for c in cbow_content.keys():
            print c,' : ', cbow_content[c]

    return cbow_content



def create_cbow_idx(data, vocab, cbow_l=2):#8):
    cbow2index = {}
    index2cbow = defaultdict(list)

    # count
    cbow_num   = defaultdict(int)
    for d in data:#[:30]:
        synonym = get_synonym(d['sentences'], vocab)
        for k in synonym.keys():
            words = synonym[k]
            # num
            cbow_num[' '.join(words)] += 1
            words.reverse()
            cbow_num[' '.join(words)] += 1
    print 'len of cbow_num', len(cbow_num)

    # sort
    num_cbow  = defaultdict(list)
    for k in cbow_num.keys():
        print 'k: ', k, ' num:', cbow_num[k]
        num_cbow[cbow_num[k]].append(k)
        print cbow_num[k]

    print num_cbow[1]

    num_k = [int(_) for _ in num_cbow.keys()]
    num_k.sort(reverse=True)
    print 'len of num_k', len(num_k)

    # create index
    index = 1
    for k in num_k:
        print 'num_K: ', k
        l = len(num_cbow[k])/2
        tmp = num_cbow[k]
        print 'num2cbow: ', tmp
        for i in range(l):
            words = tmp[-1].split(' ')
            tmp.remove(' '.join(words))
            words.reverse()
            tmp.remove(' '.join(words))

            # add to cbow_idx
            if words[0] not in cbow2index.keys() and words[1] not in cbow2index.keys():
                print 'add ', index, words
                index2cbow[index].append(words[0])
                index2cbow[index].append(words[1])
                cbow2index[words[0]] = index
                cbow2index[words[1]] = index
                index += 1
            elif words[0] in cbow2index.keys() and words[1] not in cbow2index.keys():
                idx = cbow2index[words[0]]
                if len(index2cbow[idx]) < cbow_l:
                    cbow2index[words[1]] = idx
                    index2cbow[idx].append(words[1])
                    print 'append ', idx, words
                else:
                    print 'add ', index, words[1]
                    index2cbow[index].append(words[1])
                    cbow2index[words[1]] = index
                    index += 1
            elif words[0] not in cbow2index.keys() and words[1] in cbow2index.keys():
                idx = cbow2index[words[1]]
                if len(index2cbow[idx]) < cbow_l:
                    cbow2index[words[0]] = idx
                    index2cbow[idx].append(words[0])
                    print 'append ', idx, words
                else:
                    print 'add ', index, words[0]
                    index2cbow[index].append(words[0])
                    cbow2index[words[0]] = index
                    index += 1
            elif words[0] in cbow2index.keys() and words[1] in cbow2index.keys():
                # since before all are more frequent than now, so drop now.
                print 'drop ', words
    # new
    # fill cbow with vocab
    voc_k = vocab.keys()
    ic_k = index2cbow.keys()
    if 0 in ic_k:
        ic_k.remove(0)
    if '.' in voc_k:
        voc_k.remove('.')
    ic_f = defaultdict(int)
    for k in ic_k:
        for t in index2cbow[k]:
            voc_k.remove(t)
        if len(index2cbow[k]) < cbow_l:
            ic_f[k] = sum([vocab[_] for _ in index2cbow[k]]) / (1.0 * len(index2cbow[k]))
    
    f2i = defaultdict(list)
    for k in ic_f.keys():
        f2i[ic_f[k]].append(k)
    items = f2i.items()
    items.sort()

    # find the word not in the cbow
    f2v = defaultdict(list)
    for k in voc_k:
        f2v[vocab[k]].append(k)
    f2v_k = f2v.keys()
    # small to large
    f2v_k.sort()
    voc_list = []
    for k in f2v_k:
        for v in f2v[k]:
            voc_list.append(v)
    
    # fill the cbow
    for k in f2i.keys():
        for idx in f2i[k]:
            l_tmp = len(index2cbow[idx])
            print 'index: ',idx
            print 'f: ',k
            for i in range(cbow_l - l_tmp):
                v_tmp = voc_list.pop()
                index2cbow[idx].append(v_tmp)
                print 'add f: ',vocab[v_tmp], 'len: ', len(index2cbow[idx])
                cbow2index[v_tmp] = idx
    
    cbow2index['.'] = 0
    index2cbow[0] = ['.']
    for i in range(cbow_l-1):
        v_tmp = voc_list[0]
        voc_list.remove(voc_list[0])
        index2cbow[0].append(v_tmp)
        cbow2index[v_tmp] = 0

    
    rest_l = len(voc_list)
    p_num = rest_l / cbow_l
    voc_list = voc_list[rest_l % cbow_l:]
    if rest_l % cbow_l > 0:
        for d in voc_list[:rest_l % cbow_l]:
            cbow2index[d] = cbow2index['<UNK>']
            print 'change ', d, vocab[d], 'as <UNK>'

    for i in range(p_num):
        for j in range(cbow_l):
            index2cbow[index].append(voc_list[j * p_num + i])
            cbow2index[voc_list[j * p_num + i]] = index
        index += 1

    print 'cbow_num, num_cbow, cbow2index, index2cbow is done.'
    return cbow_num, num_cbow, cbow2index, index2cbow


#cbow2index, index2cbow = create_cbow_idx(data, vocab_thres)
cbow_num, num_cbow, cbow2index, index2cbow = create_cbow_idx(data, vocab_thres)
js.dump(cbow2index, open(data_dir+'/cbow2index.json', 'w'))
js.dump(index2cbow, open(data_dir+'/index2cbow.json', 'w'))
print 'cbow num:',len(index2cbow)
print 'words in cbow num:',len(cbow2index)
print 'words num:',len(word2index)
print index2cbow[1]
print index2cbow[0]


def create_word2label(vocab, cbow2index, index2cbow, cbow_l=2):#8):
    word2label   = {}
    word2label_2 = {}
    label2word   = {}

    cbow_k = cbow2index.keys()
    for idx in index2cbow.keys():
        cbows = index2cbow[idx]
        if idx%2==1:
            cbows.reverse()
        for j,v in enumerate(cbows):
            word2label_2[v] = [idx, j]
            word2label[v]   = idx*cbow_l + j
            label2word[idx*cbow_l + j] = v

    print '"." in word2label_2: ',word2label_2['.']
    print '"a" in word2label_2: ',word2label_2['a']
    print '"." in label2word: ',label2word[word2label_2['.'][0] * cbow_l + word2label_2['.'][1]]
    print '"a" in label2word: ',label2word[word2label['a']]

    print 'word2label, word2label_2, label2word is done.'

    return word2label, word2label_2, label2word
            
word2label, word2label_2, label2word = create_word2label(vocab, cbow2index, index2cbow)
js.dump(word2label,   open(data_dir + '/word2label.json', 'w'))
js.dump(word2label_2, open(data_dir + '/word2label_2.json', 'w'))
js.dump(label2word,   open(data_dir + '/label2word.json', 'w'))

def create_word_mask(vocab):
    word_mask = {}
    upper = max([vocab[_] for _ in vocab.keys()]) * 1.0
    for k in vocab.keys():
        word_mask[k] = 1 + np.log10(upper / vocab[k])
    return word_mask

word_mask = create_word_mask(vocab_thres)
js.dump(word_mask, open(data_dir + '/word_mask.json', 'w'))
word_mask_np = np.zeros(shape=len(word_mask), dtype=np.float)
for k in word_mask.keys():
    word_mask_np[int(word2label[k])] = word_mask[k]
scipy.io.savemat(data_dir + '/word_mask.mat', word_mask_np)


def create_patch(data, vocab, word2index, cbow2index, word2label_2, word2label):
    '''
    create patch, the keys contain ['train','val', 'test', 'data_num', 'vocab_num', 'layer_num']
    '''
    patch = {'train':[], 'val':[], 'test':[]}
    voc_k = vocab.keys()
    cbow_k = cbow2index.keys()
    label_k = word2label.keys()
    patch['vocab_num'] = len(voc_k)
    patch['layer_num'] = 1

    for i,d in enumerate(data):
        sents = d['sentences']
        #feat  = feats[i,:].tolist()
        for sent in sents:
            sample = {}
            sample['filename'] = d['filename']
            sample['sentences']  = d['sentences']
            sample['imgid'] = d['imgid']
            sample['label'] = []
            sample['label_2'] = []
            sample['cbow']  = []
            for token in sent['raw'].lower().split(' '):
                if token in label_k:
                    sample['label'].append(word2label[token])
                else:
                    sample['label'].append(word2label['<UNK>'])

                if token in label_k:
                    sample['label_2'].append(word2label_2[token])
                else:
                    sample['label_2'].append(word2label_2['<UNK>'])

                if token in cbow_k:
                    sample['cbow'].append(cbow2index[token])
                else:
                    sample['cbow'].append(cbow2index['<UNK>'])
            
            patch[d['split']].append(sample)

    print 'create patch done.'
    return patch

patch                  = create_patch(data, vocab_thres, word2index, cbow2index, word2label_2, word2label)
js.dump(patch, open(data_dir+'/patch.json', 'w'))
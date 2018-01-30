import json as js
import numpy as np
from nlgeval import *
from collections import defaultdict
from time import sleep
import pdb

def save_res(res, refs, data, flag, f_res, f_ref1, f_ref2, f_ref3, f_ref4, f_ref5):
    for i in range(flag.batch_size):
        img_blob = {}
        # candidate
        tmp = []
        for _ in res[i,1:].tolist():
            if _==0:
                break
            tmp.append(_)
        tmp.append(0)
        candidate = ' '.join([data.label2word[str(ix)] for ix in tmp])
        img_blob['candidate'] = candidate

        # reference
        img_blob['references'] = []
        for j in range(5):
            ref = []
            for _ in refs[i,j,1:].tolist():
                if _==0:
                    break
                ref.append(_)
            ref.append(0)
            txt = ' '.join([data.label2word[str(ix)] for ix in ref])
            img_blob['references'].append(txt)

        # write to file
        f_res.write(img_blob['candidate'] + '\n')
        f_ref1.write(img_blob['references'][0] + '\n')
        f_ref2.write(img_blob['references'][1] + '\n')
        f_ref3.write(img_blob['references'][2] + '\n')
        f_ref4.write(img_blob['references'][3] + '\n')
        f_ref5.write(img_blob['references'][4] + '\n')


def save_res_final(res, refs, data, flag, f_res, f_ref1, f_ref2, f_ref3, f_ref4, f_ref5):
    for i in range(flag.batch_size):
        img_blob = {}
        # candidate
        tmp = []
        for _ in res[i,1:].tolist():
            if _==0:
                break
            tmp.append(_)
        tmp.append(0)
        candidate = ' '.join([data.label2word[str(ix)] for ix in tmp])
        img_blob['candidate'] = candidate

        # reference
        img_blob['references'] = []
        for j in range(5):
            img_blob['references'].append(refs[i, j].split('\n')[0])

        # write to file
        f_res.write(img_blob['candidate'] + '\n')
        f_ref1.write(img_blob['references'][0] + '\n')
        f_ref2.write(img_blob['references'][1] + '\n')
        f_ref3.write(img_blob['references'][2] + '\n')
        f_ref4.write(img_blob['references'][3] + '\n')
        f_ref5.write(img_blob['references'][4] + '\n')

def get_each_score(res, refs, data, size, score_type='SUM'):
    scores = []
    for i in range(size):
        # res
        res_tmp = []
        for _ in res[i,1:].tolist():
            if _==0:
                break
            res_tmp.append(_)
        res_tmp.append(0)
        candidate = ' '.join([data.label2word[str(ix)] for ix in res_tmp]).encode('unicode-escape')

        score = compute_cider(candidate, refs[i])
        #print score
        scores.append(score['CIDEr'])

        '''
        # refs
        txts = '||<|>||'.join(refs[i]).encode('unicode-escape')

        # score
        score_tmp = compute_individual_score(ref=txts, hyp=candidate)
        if score_type=='SUM':
            scores.append(sum(score_tmp.values()) / len(score_tmp))
        else:
            scores.append(score_tmp[score_type])
        '''

    return scores


def train_test_final(model, data, flag, sess, feat_tmp, inds_tmp, refs, fr, score_type='SUM'):
    res_max, res_scst = sess.run([model.res_max_train, model.res_scst_train], feed_dict={model.feat_train:feat_tmp, model.inds_train:inds_tmp, model.refs_train:refs, model.fire_rate:[fr]})

    scores_scst = get_each_score(res_scst, refs, data, size=flag.batch_size, score_type=score_type)
    scores_max  = get_each_score(res_max, refs, data, size=flag.batch_size, score_type=score_type)
    scores_sub  = [scores_scst[_] - scores_max[_] for _ in range(flag.batch_size)]
    #print scores_sub
    #pdb.set_trace()

    return scores_sub, scores_scst, scores_max
    

def train_test(model, data, flag, sess, feat_tmp, inds_tmp, word_label_2_tmp, refs_tmp, score_type='SUM'):
    #scores = defaultdict(list)
    scores = []
    res, refs = sess.run([model.res_train, model.word_refs], feed_dict={model.feat:feat_tmp, model.inds:inds_tmp, model.word_label_2:word_label_2_tmp, model.word_refs:refs_tmp})

    for i in range(flag.batch_size):
        res_tmp = []
        for _ in res[i,1:].tolist():
            if _==0:
                break
            res_tmp.append(_)
        res_tmp.append(0)
        candidate = ' '.join([data.label2word[str(ix)] for ix in res_tmp]).encode('unicode-escape')

        txts = []
        for j in range(5):
            ref_tmp = []
            for _ in refs[i,j,1:].tolist():
                ref_tmp.append(_)
                if _==0:
                    break
            ref_tmp.append(0)
            txt = ' '.join([data.label2word[str(ix)] for ix in ref_tmp])
            txts.append(txt)
        
        #print txts
        txts = '||<|>||'.join(txts).encode('unicode-escape')
        #print txts
        #print type(txts)
        score_tmp = compute_individual_score(ref=txts, hyp=candidate)
        scores.append(sum(score_tmp.values()) / len(score_tmp))

    return scores



    def predict_val(self, data):
        self.feat_val, self.refs_val = data.samp_val_feat, data.samp_val_refs

        self.embed_feat_val = tf.reshape(tf.nn.bias_add(tf.matmul(tf.reshape(feat, [-1, self.flag.feat_size]), self.w_feat), self.b_feat), [self.flag.batch_size, self.flag.feat_num, self.flag.embedding_size])

        # result_val
        self.res_max_val  = self.predict_batch(self.embed_feat_val, inds=None, predict_type='MAX')


def test_final(model, data, flag, sess):
    f_res  = open('result/res.txt','w')
    f_ref1 = open('result/ref1.txt','w')
    f_ref2 = open('result/ref2.txt','w')
    f_ref3 = open('result/ref3.txt','w')
    f_ref4 = open('result/ref4.txt','w')
    f_ref5 = open('result/ref5.txt','w')

    blob = []
    for itr in range(len(data.patch['val']) / flag.batch_size):
        #feat_tmp, refs_tmp = sess.run([data.samp_val_feat, data.samp_val_refs])
        res, refs = sess.run([model.res_max_val, model.refs_val])
        
        save_res_final(res, refs, data, flag, f_res, f_ref1, f_ref2, f_ref3, f_ref4, f_ref5)

    f_res.close()
    f_ref1.close()
    f_ref2.close()
    f_ref3.close()
    f_ref4.close()
    f_ref5.close()

    scores = compute_metrics(hypothesis='result/res.txt', references=['result/ref1.txt', 'result/ref2.txt', 'result/ref3.txt', 'result/ref4.txt', 'result/ref5.txt'], no_skipthoughts=True, no_glove=True)

    return scores

'''
def test(model, data, flag, sess):
    f_res  = open('result/res.txt','w')
    f_ref1 = open('result/ref1.txt','w')
    f_ref2 = open('result/ref2.txt','w')
    f_ref3 = open('result/ref3.txt','w')
    f_ref4 = open('result/ref4.txt','w')
    f_ref5 = open('result/ref5.txt','w')

    blob = []
    for itr in range(len(data.patch['val']) / flag.batch_size):
        res, refs = sess.run([model.res_max_val, model.inds_val])
        
        save_res(res, refs, data, flag, f_res, f_ref1, f_ref2, f_ref3, f_ref4, f_ref5)

    f_res.close()
    f_ref1.close()
    f_ref2.close()
    f_ref3.close()
    f_ref4.close()
    f_ref5.close()

    scores = compute_metrics(hypothesis='result/res.txt', references=['result/ref1.txt', 'result/ref2.txt', 'result/ref3.txt', 'result/ref4.txt', 'result/ref5.txt'], no_skipthoughts=True, no_glove=True)

    return scores
'''


def test_single(model, data, flag, sess, select='MAX'):
    f_res  = open('result/res.txt','w')
    f_ref1 = open('result/ref1.txt','w')
    f_ref2 = open('result/ref2.txt','w')
    f_ref3 = open('result/ref3.txt','w')
    f_ref4 = open('result/ref4.txt','w')
    f_ref5 = open('result/ref5.txt','w')

    blob = []
    for itr in range(len(data.patch['val'])):
        feat_single, refs_single =  sess.run([data.samp_val_single_feat, data.samp_val_single_refs])#sess.run(data.get_sample(split='val_single'))

        cell_cs_0 = [np.zeros([1,flag.hidden_state_size])] * flag.beam_size
        cell_hs_0 = [np.zeros([1,flag.hidden_state_size])] * flag.beam_size
        cell_cs_1 = [np.zeros([1,flag.hidden_state_size])] * flag.beam_size
        cell_hs_1 = [np.zeros([1,flag.hidden_state_size])] * flag.beam_size

        res = [[0]] * flag.beam_size
        log_p = [0.0] * flag.beam_size

        is_all = [False] * flag.beam_size
        for t in range(flag.sentence_length):
            res_tmp = []
            logp_tmp = []
            _is      = []
            cell_cs_0_tmp = []
            cell_hs_0_tmp = []
            cell_cs_1_tmp = []
            cell_hs_1_tmp = []

            for i in range(flag.beam_size):
                if not is_all[i]:
                    #print res[i][-1]
                    p_top, idx_top, (cell_c_0, cell_h_0), (cell_c_1, cell_h_1) = sess.run([model.p_top, model.idx_top, model.cell_state_0_single, model.cell_state_1_single], feed_dict={model.feat_single:feat_single, model.inds_single: res[i][-1:], model.cell_c_0:cell_cs_0[i], model.cell_h_0:cell_hs_0[i], model.cell_c_1:cell_cs_1[i], model.cell_h_1:cell_hs_1[i]})

                    _is = _is + [i] * flag.beam_size
                    res_tmp = res_tmp + idx_top.tolist()
                    logp_tmp = logp_tmp + [log_p[i]-np.log(_) for _ in p_top.tolist()]
                else:
                    _is = _is + [i] * flag.beam_size
                    res_tmp = res_tmp + (0 * idx_top).tolist()
                    logp_tmp = logp_tmp + [log_p[i] for _ in p_top.tolist()]

                cell_cs_0_tmp.append(cell_c_0)
                cell_hs_0_tmp.append(cell_h_0)
                cell_cs_1_tmp.append(cell_c_1)
                cell_hs_1_tmp.append(cell_h_1)

            log_p = []
            res_new = []
            cell_cs_0 = []
            cell_hs_0 = []
            cell_cs_1 = []
            cell_hs_1 = []
            for i in range(flag.beam_size):
                tmp = min(logp_tmp)
                tmp_idx = logp_tmp.index(tmp)
                _i = _is[tmp_idx]
                _ii = res_tmp[tmp_idx]

                log_p.append(tmp)
                res_new.append(res[_i] + [res_tmp[tmp_idx]])
                cell_cs_0.append(cell_cs_0_tmp[_i])
                cell_hs_0.append(cell_hs_0_tmp[_i])
                cell_cs_1.append(cell_cs_1_tmp[_i])
                cell_hs_1.append(cell_hs_1_tmp[_i])
                if res_tmp[tmp_idx]==0:
                    is_all[i]=True

                if is_all[i]:
                    while _i in _is:
                        ii = _is.index(_i)
                        logp_tmp.pop(ii)
                        res_tmp.pop(ii)
                        _is.pop(ii)
                elif t==0:
                    while _ii in res_tmp:
                        ii = res_tmp.index(_ii)
                        logp_tmp.pop(ii)
                        res_tmp.pop(ii)
                        _is.pop(ii)
                else:
                    logp_tmp.pop(tmp_idx)
                    res_tmp.pop(tmp_idx)
                    _is.pop(tmp_idx)
            #pdb.set_trace()
            res = res_new
            #print res, '\n'
        for i in range(flag.beam_size):
            r = res[i]
            r.append(0)
            #print 'logp: ', log_p[i], '\t',' '.join([data.label2word[str(ix)] for ix in r[1:r[1:].index(0)+1]])
        res = res[log_p.index(min(log_p))]
        res.append(0)
        candidate = ' '.join([data.label2word[str(ix)] for ix in res[1:res[1:].index(0)+2]])
        #print itr, ': ', candidate, '\n'
        f_res.write(candidate + '\n')
        txts = []
        for i in range(5):
            ref = refs_single[0][i].tolist()
            ref.append(0)
            txt = ' '.join([data.label2word[str(ix)] for ix in ref[1:ref[1:].index(0)+2]])
            txts.append(txt)
        f_ref1.write(txts[0]+'\n')
        f_ref2.write(txts[1]+'\n')
        f_ref3.write(txts[2]+'\n')
        f_ref4.write(txts[3]+'\n')
        f_ref5.write(txts[4]+'\n')
    f_ref1.close()
    f_ref2.close()
    f_ref3.close()
    f_ref4.close()
    f_ref5.close()


    
    #scores = compute_metrics(hypothesis='result/res.txt', references=['result/ref1.txt', 'result/ref2.txt', 'result/ref3.txt', 'result/ref4.txt', 'result/ref5.txt'], no_skipthoughts=True, no_glove=True)

    #return 0#scores
import tensorflow  as tf
import numpy as np
import random

def ln(inp, bias=0.0, gain=1.0):
    '''
    layer norm
    '''
    inp_shape = inp.get_shape()

    mean, var = tf.nn.moments(inp, -1)
    var += 1e-20
    if len(inp_shape)==1:
        return gain * (inp - mean) / tf.sqrt(var) + bias
    elif len(inp_shape)==2:
        mean = tf.tile(tf.expand_dims(mean, -1), [1, int(inp_shape[-1])])
        var  = tf.tile(tf.expand_dims(var,  -1), [1, int(inp_shape[-1])])
        return gain * (inp - mean) / tf.sqrt(var) + bias    
    elif len(inp_shape)==3:
        mean = tf.tile(tf.expand_dims(mean, -1), [1, 1, int(inp_shape[-1])])
        var  = tf.tile(tf.expand_dims(var,  -1), [1, 1, int(inp_shape[-1])])
        return gain * (inp - mean) / tf.sqrt(var) + bias    

    return inp

def softmax_monte_carlo_sample(logits, shape, ground=None, sample_rate=0.25):
    p = tf.nn.softmax(logits)
    if ground is None:
        ground = tf.argmax(p, -1)
    else:
        ground = ground

    mask_tmp = tf.sign(tf.random_uniform(shape=[shape[0]], minval=0-sample_rate, maxval=1.0-sample_rate))
    samp_mask_p = tf.cast((tf.abs(mask_tmp) - mask_tmp) / 2, tf.int64)

    samp_tmp = tf.tile(tf.random_uniform(shape=[shape[0], 1], minval=0.0, maxval=1.0), [1, shape[1]])
    
    # sample
    res = samp_mask_p * tf.cast(tf.argmin(tf.abs(tf.cumsum(p, -1) - samp_tmp), -1), tf.int64) + (1 - samp_mask_p) * ground
    
    return res

class model():
    def __init__(self, flag, data):
        self.flag = flag

        self.train(data)
        self.predict_train()
        self.predict_val(data)
        self.predict_single()


    def prepare_weights(self, init=tf.contrib.layers.xavier_initializer()):
        # encode
        with tf.variable_scope('encode'):
            self.w_feat = tf.get_variable('w_feat', shape=[self.flag.feat_size, self.flag.embedding_size], dtype=tf.float32, initializer=init)
            self.b_feat = tf.get_variable('b_feat', shape=[self.flag.embedding_size], dtype=tf.float32, initializer=init)
        
            self.word_embedding = tf.get_variable('word_embedding', shape=[self.flag.label_0_size, self.flag.embedding_size], dtype=tf.float32, initializer=init)

            self.encode_loss = tf.nn.l2_loss(self.w_feat) + tf.nn.l2_loss(self.word_embedding)

        # lstm
        with tf.variable_scope('lstm_0'):
            self.lstm_cell_0 = tf.nn.rnn_cell.LSTMCell(self.flag.hidden_state_size, forget_bias=0.0, state_is_tuple=True, initializer=init)
            self.lstm_cell_0 = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_0, output_keep_prob=self.flag.keep_prob)

        with tf.variable_scope('lstm_1', reuse=tf.AUTO_REUSE):
            self.lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(self.flag.hidden_state_size, forget_bias=0.0, state_is_tuple=True,  initializer=init)
            self.lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_1, output_keep_prob=self.flag.keep_prob)


        # decode
        with tf.variable_scope('decode'):
            self.w_softmax_0 = tf.get_variable('w_softmax_0', shape=[2 * self.flag.hidden_state_size, self.flag.label_0_size], dtype=tf.float32, initializer=init)
            self.b_softmax_0 = tf.get_variable('b_softmax_0', shape=[self.flag.label_0_size], dtype=tf.float32, initializer=init)

            self.decode_loss = tf.nn.l2_loss(self.w_softmax_0)


        # adap
        with tf.variable_scope('adap'):
            
            self.w_v = tf.get_variable('w_v', shape=[self.flag.embedding_size, self.flag.adap_size], dtype=tf.float32, initializer=init)
            self.b_v = tf.get_variable('b_v', shape=[self.flag.adap_size], dtype=tf.float32, initializer=init)

            self.w_g = tf.get_variable('w_g', shape=[self.flag.hidden_state_size, self.flag.adap_size], dtype=tf.float32, initializer=init)
            self.b_g = tf.get_variable('b_g', shape=[self.flag.adap_size], dtype=tf.float32, initializer=init)

            self.w_x = tf.get_variable('w_x', shape=[self.flag.hidden_state_size, self.flag.hidden_state_size], dtype=tf.float32, initializer=init)
            self.w_sh = tf.get_variable('w_sh', shape=[self.flag.hidden_state_size, self.flag.hidden_state_size], dtype=tf.float32, initializer=init)

            self.w_s = tf.get_variable('w_s', shape=[self.flag.hidden_state_size, self.flag.adap_size], dtype=tf.float32, initializer=init)

            self.w_h = tf.get_variable('w_h', shape=[self.flag.adap_size, 1], dtype=tf.float32, initializer=init)

            self.adap_loss = tf.nn.l2_loss(self.w_g) + tf.nn.l2_loss(self.w_h) + tf.nn.l2_loss(self.w_v) + tf.nn.l2_loss(self.w_x) + tf.nn.l2_loss(self.w_sh) + tf.nn.l2_loss(self.w_s)

        # l2_loss
        self.l2_loss = self.encode_loss + self.decode_loss + self.adap_loss


    def forward(self, cell_state_0, cell_state_1, embed_feat, embed_word, batch_size=1, is_training=True):
        # pre
        if self.flag.layer_norm:
            embed_feat = ln(embed_feat)
            embed_word = ln(embed_word)

            cell_tmp = tf.concat([cell_state[0], cell_state[1]], 0)
            cell_tmp = ln(cell_tmp)
            cell_state = tuple(tf.split(cell_tmp, num_or_size_splits=2, axis=0))


        if self.flag.encode_relu:
            embed_feat = tf.nn.relu(embed_feat)
        if self.flag.drop_encoder:
            embed_feat = tf.nn.dropout(embed_feat, keep_prob=self.flag.keep_prob)
            embed_word = tf.nn.dropout(embed_word, keep_prob=self.flag.keep_prob)

        # language
        cell_input_0 = embed_word# + embed_feat[:,0,:]

        (cell_output_0, cell_state_0_new) = self.lstm_cell_0(cell_input_0, cell_state_0)
        if not is_training or not self.flag.drop_decoder:
            cell_output_0 = cell_state_0_new[1]
        # cell_state:[c_bew, h_new]

        # sent adap
        with tf.variable_scope('adap') as scope:
            scope.reuse_variables()
            # img
            tmp_v = tf.reshape(tf.matmul(tf.reshape(embed_feat, [-1, self.flag.embedding_size]), self.w_v), [batch_size, self.flag.feat_num, self.flag.adap_size])
            tmp_h = tf.tile(tf.expand_dims(tf.matmul(cell_output_0, self.w_g), 1), [1, self.flag.feat_num, 1])

            z_t = tf.matmul(tf.nn.tanh(tmp_v + tmp_h), tf.tile(tf.expand_dims(self.w_h, 0), [batch_size, 1, 1]))
            a_t = tf.nn.softmax(z_t, 1)
            c_t = tf.reduce_sum(tf.tile(a_t, [1, 1, self.flag.embedding_size]) * embed_feat, 1)            

            # sent
            g_t = tf.nn.sigmoid(tf.matmul(cell_input_0, self.w_x) + tf.matmul(cell_output_0, self.w_sh))
            s_t = g_t * tf.nn.tanh(cell_state_0_new[0])
        cell_input_1 = tf.concat([c_t, cell_output_0], -1)

        with tf.variable_scope('lstm_1') as scope:
            (cell_output_1, cell_state_1_new) = self.lstm_cell_1(cell_input_1, cell_state_1)
        if not is_training or not self.flag.drop_decoder:
            cell_output_1 = cell_state_1_new[1]

        adap_out = tf.concat([cell_output_1, s_t], -1)

        # decode
        with tf.variable_scope('decode') as scope:
            logits_0 = tf.nn.bias_add(tf.matmul(adap_out, self.w_softmax_0), self.b_softmax_0)
            logits = logits_0
        
        return logits, cell_state_0_new , cell_state_1_new
        


    def train(self, data):
        self.prepare_weights()

        # batch_data
        self.fire_rate   = tf.placeholder(dtype=tf.float32, shape=[1])

        self.feat_train, self.inds_train, self.refs_train = data.samp_train_feat, data.samp_train_inds, data.samp_train_refs

        # embed_feat
        self.embed_feat_train = tf.reshape(tf.nn.bias_add(tf.matmul(tf.reshape(self.feat_train, [self.flag.batch_size * self.flag.feat_num, self.flag.feat_size]), self.w_feat), self.b_feat), [self.flag.batch_size, self.flag.feat_num, self.flag.embedding_size])

        # embed_word
        self.embed_word_train = tf.transpose(tf.nn.embedding_lookup(self.word_embedding, self.inds_train[:,:-1]), perm=[1, 0, 2])

        #  loop
        output = []
        cell_state_0  = self.lstm_cell_0.zero_state(self.flag.batch_size, tf.float32)
        cell_state_1  = self.lstm_cell_1.zero_state(self.flag.batch_size, tf.float32)
        logits = tf.constant(np.zeros([self.flag.batch_size, self.flag.label_0_size]), dtype=tf.float32)
        for t in range(self.flag.sentence_length):
            # annealing
            input_inds = softmax_monte_carlo_sample(logits=logits, shape=[self.flag.batch_size, self.flag.label_0_size], ground=self.inds_train[:, t], sample_rate=self.fire_rate[0])
            embed_word_tmp = tf.nn.embedding_lookup(self.word_embedding, input_inds)
            if t==0:
                embde_word_tmp = tf.nn.embedding_lookup(self.word_embedding, self.inds_train[:,0])
            # forward
            logits, cell_state_0, cell_state_1 = self.forward(cell_state_0, cell_state_1, self.embed_feat_train, embed_word_tmp, batch_size=self.flag.batch_size, is_training=True)
            output.append(logits)

        # logits and labels
        self.logits_0 = tf.concat([tf.expand_dims(_, 0) for _ in output], 0)

        self.label_0 = tf.one_hot(tf.transpose(self.inds_train[:,1:]), self.flag.label_0_size)

        self.sent_mask_0 = tf.tile(tf.expand_dims(tf.transpose(tf.cast(tf.concat([self.inds_train[:,1:2], self.inds_train[:,1:-1]], -1)>0, tf.float32)), -1), [1, 1, self.flag.label_0_size])

        self.label_0_masked  = self.label_0  * self.sent_mask_0
        self.logits_0_masked = self.logits_0 * self.sent_mask_0
        

    def predict_batch(self, embed_feat, inds=None, predict_type='MAX'):
        # load input

        # LSTM
        cell_state_0  = self.lstm_cell_0.zero_state(self.flag.batch_size, tf.float32)
        cell_state_1  = self.lstm_cell_1.zero_state(self.flag.batch_size, tf.float32)

        # result
        res = []
        res.append(tf.constant(np.zeros(self.flag.batch_size), dtype=tf.int64))

        # predict
        for t in range(self.flag.sentence_length):
            # pre word
            embed_word = tf.nn.embedding_lookup(self.word_embedding, tf.cast(res[-1], tf.int64))

            # lstm
            logits, cell_state_0, cell_state_1 = self.forward(cell_state_0, cell_state_1, embed_feat, embed_word, batch_size=self.flag.batch_size, is_training=False)

            # res
            if predict_type=='MAX' or (predict_type=='SCST' and inds is None):
                res.append(tf.argmax(logits, -1))
            elif predict_type=='SCST' and inds is not None:
                res.append(softmax_monte_carlo_sample(logits=logits, shape=[self.flag.batch_size, self.flag.label_0_size], ground=inds[:, t+1],  sample_rate=self.fire_rate[0]))

        return tf.concat([tf.expand_dims(_, -1) for _ in res], -1)

    def predict_val(self, data):
        self.feat_val, self.refs_val = data.samp_val_feat, data.samp_val_refs

        self.embed_feat_val = tf.reshape(tf.nn.bias_add(tf.matmul(tf.reshape(self.feat_val, [-1, self.flag.feat_size]), self.w_feat), self.b_feat), [self.flag.batch_size, self.flag.feat_num, self.flag.embedding_size])

        # result_val
        self.res_max_val  = self.predict_batch(self.embed_feat_val, inds=None, predict_type='MAX')

    def predict_train(self):
        # result_train
        self.res_max_train  = self.predict_batch(self.embed_feat_train, self.inds_train, predict_type='MAX')
        self.res_scst_train = self.predict_batch(self.embed_feat_train, self.inds_train, predict_type='SCST')

    def predict_single(self):
        self.feat_single = tf.placeholder(dtype=tf.float32, shape=[1, self.flag.feat_num, self.flag.feat_size])
        self.inds_single = tf.placeholder(dtype=tf.int64, shape=[1])

        self.embed_feat_single = tf.reshape(tf.nn.bias_add(tf.matmul(tf.reshape(self.feat_single, [-1, self.flag.feat_size]), self.w_feat), self.b_feat), [1, self.flag.feat_num, self.flag.embedding_size])
        self.embed_word_single = tf.nn.embedding_lookup(self.word_embedding, self.inds_single)

        self.cell_h_0 = tf.placeholder(dtype=tf.float32, shape=[1,self.flag.hidden_state_size])
        self.cell_h_1 = tf.placeholder(dtype=tf.float32, shape=[1,self.flag.hidden_state_size])
        self.cell_c_0 = tf.placeholder(dtype=tf.float32, shape=[1,self.flag.hidden_state_size])
        self.cell_c_1 = tf.placeholder(dtype=tf.float32, shape=[1,self.flag.hidden_state_size])

        logits, self.cell_state_0_single, self.cell_state_1_single = self.forward((self.cell_c_0, self.cell_h_0), (self.cell_c_1, self.cell_h_1), self.embed_feat_single, self.embed_word_single, batch_size=1, is_training=False)

        self.p_top, self.idx_top = tf.nn.top_k(tf.nn.softmax(logits)[0], k=self.flag.beam_size)
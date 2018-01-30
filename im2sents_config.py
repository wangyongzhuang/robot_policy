class config():
    def __init__(self):
        # data
        self.dataset   = 'coco'
        self.data_dir  = '/home/wyz/data/' + self.dataset
        self.feats_path  = '/home/wyz/data/coco/data_z_final/feats'
        self.data_size = 0
        self.feat_size = 2048
        self.feat_num  = 50

        self.sentence_length = 20
        self.label_0_size    = 0

        # lstm
        self.hidden_state_size = 512
        self.embedding_size    = 512
        self.adap_size         = 512

        self.feed_once    = False
        self.layer_norm   = False
        self.encode_relu  = False
        self.drop_encoder = True
        self.drop_decoder = True
        self.keep_prob    = 0.5

        # train
        self.training_approach  = 'SCST' #'FLST' # 'XE', 'SCST'
        self.reward_ratio    = 0.2 #0.8**10=0.1
        self.beam_size       = 3
        #self.label_1_weight  = 0.99
        self.is_finetune     = True#False

        self.loss_scale    = 1000
        self.batch_size    = 100
        self.learning_rate = 5e-5
        self.lr_decay      = 0.1
        self.weight_decay  = 1e-5

        self.epoch_num   = 50
        self.epoch_decay = 20
        self.epoch_val   = 0.1

        # save
        self.ckpt_dir = '/home/wyz/log/ckpt_final_'+self.training_approach
        self.sum_dir  = '/home/wyz/log/sum_final_'+self.training_approach


    def set_data_size(self, patch):
        self.data_size = patch['train_num']

    def set_label_size(self, patch):
        self.label_0_size = patch['label_0_size']

        print 'label_size: %d'%(self.label_0_size)

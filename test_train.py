import tensorflow as tf
from Recognizer import MathFormulaRecognizer

from util import *

from data_iterator import dataIterator

import sys
class eval_train_code:
    def __init__(self, ind):
        self.config_initialize()
        self.load_data()
        self.initialize_model(ind)


    def config_initialize(self):
        self.home_path = os.getcwd()
        self.checkpoint_path = os.path.join(self.home_path, 'save', 'model.ckpt')
        self.checkpoint_dir = os.path.join(self.home_path, 'save')
        self.max_iters = 100000
        self.batch_size = 16
        self.valid_batch_size = 8
        # Evaluation Checkpoint
        self.nEvaImages = 300
        self.EvaEach = 2500
        self.SummaryEach = 1000
        self.device = "/gpu:0"
        self.batch_Imagesize = 500000
        self.valid_batch_Imagesize = 500000
        self.maxImagesize = 500000
        self.maxlen = 200
        self.n_epoch = 10000

    def initialize_model(self, ind):
        checkpoint_dir = self.checkpoint_dir
        self.sess = tf.Session()
        self.model = MathFormulaRecognizer(num_label=112, dim_hidden=128)

        train = np.squeeze(self.train)
        x, x_mask, y, y_mask = prepare_data(train[ind, 0], train[ind, 1])
        y_mask = np.transpose(y_mask)
        self.start_step = 0

        self.total_correct, self.alphas, self.betas, self.corrects = self.model.eval_train(y_mask.shape[1])
        self.saver = tf.train.Saver(max_to_keep=10)
        saved_path = tf.train.latest_checkpoint(checkpoint_dir)

        if (saved_path):
            # tf.reset_default_graph()
            self.saver.restore(self.sess, saved_path)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            self.start_step = int(step)
            print('model restored', self.start_step)
        else:
            self.sess.run(tf.global_variables_initializer())

    def load_data(self):
        datasets = ['./data/offline-train.pkl',
                    './data/train_caption.txt']
        dictionaries = ['./data/dictionary.txt']

        worddicts = load_dict(dictionaries[0])
        worddicts_r = [None] * len(worddicts)

        for kk, vv in worddicts.items():
            worddicts_r[vv] = kk

        self.train, self.train_uid_list = dataIterator(datasets[0], datasets[1],
                                             worddicts,
                                             batch_size=self.batch_size, batch_Imagesize=self.batch_Imagesize,
                                             maxlen=self.maxlen,maxImagesize=self.maxImagesize)
    def run(self, ind):

        model = self.model
        sess = self.sess
        train = np.squeeze(self.train)
        n_train_img = train.shape[0]


        x, x_mask, y, y_mask = prepare_data(train[ind, 0], train[ind, 1])
        y = np.transpose(y)
        y_mask = np.transpose(y_mask)
        x = x[0:1, :, :, :]
        x_mask = x_mask[0:1, :, :]
        y = y[0:1, :]
        y_mask = y_mask[0:1, :]
        total_correct, corrects = sess.run([self.total_correct, self.corrects], feed_dict={model.x: x, model.x_mask: x_mask, model.y: y, \
                                                   model.y_mask: y_mask, model.is_train: True})

        print(total_correct/np.sum(y_mask))
        print(corrects)

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print('please give one arg to specify image batch')
    # batch_selected = int(sys.argv[1])
    batch_selected = 2
    test_obj = eval_train_code(batch_selected)
    test_obj.run(batch_selected)


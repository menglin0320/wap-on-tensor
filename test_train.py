import sys

import cv2
import tensorflow as tf

from Recognizer import MathFormulaRecognizer
from data_iterator import dataIterator
from util import *


def attention_on_origin(attention, im):
    height, width = im.shape
    aug_attention = cv2.resize(attention, (width, height))
    ret = np.zeros((height, width))
    ret = cv2.normalize(im + aug_attention, ret, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return ret


class eval_train_code:
    def __init__(self, ind, chosen_set):
        self.config_initialize()
        self.load_data()
        self.initialize_model(ind, chosen_set)

    def config_initialize(self):
        self.home_path = os.getcwd()
        self.checkpoint_path = os.path.join(self.home_path, 'save', 'model.ckpt')
        self.checkpoint_dir = os.path.join(self.home_path, 'save')
        self.max_iters = 100000
        self.batch_size = 16
        self.valid_batch_size = 16
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

    def initialize_model(self, ind, chosen_set):
        checkpoint_dir = self.checkpoint_dir
        self.sess = tf.Session()
        self.model = MathFormulaRecognizer(num_label=112, dim_hidden=128)

        train = np.squeeze(self.train)
        valid = np.squeeze(self.valid)
        if chosen_set == 'train':
            x, x_mask, y, y_mask = prepare_data(train[ind, 0], train[ind, 1])
        else:
            x, x_mask, y, y_mask = prepare_data(valid[ind, 0], valid[ind, 1])
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
        valid_datasets = ['./data/offline-test.pkl',
                          './data/test_caption.txt']
        dictionaries = ['./data/dictionary.txt']

        worddicts = load_dict(dictionaries[0])
        worddicts_r = [None] * len(worddicts)

        for kk, vv in worddicts.items():
            worddicts_r[vv] = kk

        self.train, self.train_uid_list = dataIterator(datasets[0], datasets[1],
                                                       worddicts,
                                                       batch_size=self.batch_size, batch_Imagesize=self.batch_Imagesize,
                                                       maxlen=self.maxlen, maxImagesize=self.maxImagesize)
        self.valid, self.valid_uid_list = dataIterator(valid_datasets[0], valid_datasets[1],
                                                       worddicts,
                                                       batch_size=self.batch_size, batch_Imagesize=self.batch_Imagesize,
                                                       maxlen=self.maxlen, maxImagesize=self.maxImagesize)

    def run(self, ind, chosen_set):

        model = self.model
        sess = self.sess
        train = np.squeeze(self.train)
        valid = np.squeeze(self.valid)
        n_train_img = train.shape[0]

        if chosen_set == 'train':
            x, x_mask, y, y_mask = prepare_data(train[ind, 0], train[ind, 1])
        else:
            x, x_mask, y, y_mask = prepare_data(valid[ind, 0], valid[ind, 1])
        y = np.transpose(y)
        y_mask = np.transpose(y_mask)
        x = x[0:1, :, :, :]
        x_mask = x_mask[0:1, :, :]
        y = y[0:1, :]
        y_mask = y_mask[0:1, :]
        total_correct, corrects, betas, height, width = sess.run(
            [self.total_correct, self.corrects, self.betas, model.feature_height, model.feature_width], \
            feed_dict={model.x: x, model.x_mask: x_mask, model.y: y, \
                       model.y_mask: y_mask, model.is_train: False})
        for i in range(0, y.shape[1]):
            with_att = attention_on_origin(np.reshape(betas[i][0], (height, width, 1)), np.squeeze(x[0]))
            cv2.imwrite('with_att' + str(i) + '.png', with_att * 255)
        print(total_correct / np.sum(y_mask))
        print(corrects)
        print(y)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError('please give one arg to specify image batch')
    batch_selected = int(sys.argv[1])
    chosen_set = sys.argv[2]
    # batch_selected = 30
    # chosen_set = 'valid'
    test_obj = eval_train_code(batch_selected, chosen_set)
    test_obj.run(batch_selected, chosen_set)

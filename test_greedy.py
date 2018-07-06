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


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# TODO save and check attention
class test_code:
    def __init__(self):
        self.init_config()
        self.build_dict()
        self.load_model()

    def init_config(self):
        # not sure if this one is safe on windows, warning
        self.home_path = os.getcwd()

        self.checkpoint_path = os.path.join(self.home_path, 'save')
        self.max_iters = 100000
        self.batch_size = 16
        self.valid_batch_size = 2
        self.finetune_encoder_after = -1
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
        self.datasets = ['./data/offline-train.pkl',
                         './data/train_caption.txt']
        self.valid_datasets = ['./data/offline-test.pkl',
                               './data/test_caption.txt']

    def build_dict(self):
        dictionaries = ['./data/dictionary.txt']

        self.worddicts = load_dict(dictionaries[0])
        self.worddicts_r = [None] * len(self.worddicts)
        for kk, vv in self.worddicts.items():
            self.worddicts_r[vv] = kk

    def load_model(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.model = MathFormulaRecognizer(num_label=112, dim_hidden=128)
        self.logits, self.alpha_t, self.beta_t = self.model.build_greedy_eval()
        saver = tf.train.Saver(max_to_keep=10)
        saved_path = tf.train.latest_checkpoint(self.checkpoint_path)
        saver.restore(self.sess, saved_path)
        print('restored model from: ' + saved_path)
        # self.writer = tf.summary.FileWriter("./log", self.sess.graph)

    def get_data(self, set_chosen):
        if set_chosen == 'train':
            return dataIterator(self.datasets[0], self.datasets[1],
                                self.worddicts,
                                batch_size=self.batch_size, batch_Imagesize=self.batch_Imagesize,
                                maxlen=self.maxlen, maxImagesize=self.maxImagesize)

        else:
            return dataIterator(self.valid_datasets[0], self.valid_datasets[1],
                                self.worddicts,
                                batch_size=self.batch_size, batch_Imagesize=self.batch_Imagesize,
                                maxlen=self.maxlen, maxImagesize=self.maxImagesize)

    def run(self, batch_picked):
        # This code assumes that at least one character in the list
        # is recognized
        train, train_uid_list = self.get_data('train')
        valid, valid_uid_list = self.get_data('test')
        valid = np.squeeze(valid)
        train = np.squeeze(train)
        sess = self.sess
        model = self.model
        if chosen_set == 'train':
            x, x_mask, y, y_mask = prepare_data(train[batch_picked, 0], train[batch_picked, 1])
        else:
            x, x_mask, y, y_mask = prepare_data(valid[batch_picked, 0], valid[batch_picked, 1])
        # for simplicity only test first image on the batch
        x = x[0:1, :, :, :]
        x_mask = x_mask[0:1, :, :]

        Words, Alphas, height, width, Beta = sess.run(
            [self.logits, self.alpha_t, model.feature_height, model.feature_width, self.beta_t],
            feed_dict={model.x: x, model.x_mask: x_mask,
                       model.is_train: False})

        Words = [w[0] for w in Words]
        str_list = []
        self.worddicts_r.append('sof')
        for c in Words:
            if c == 0:
                break
            str_list.append(self.worddicts_r[c])
        str = ''.join(str_list)
        print(str)
        print(height)
        print(width)
        with_atts = []
        for i in range(0, 10):
            with_att = attention_on_origin(np.reshape(Alphas[i], (height, width, 1)), np.squeeze(x[0]))
            with_atts.append(with_att)

        return Words, np.squeeze(x[0]), with_atts


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError('please give one arg to specify image batch')
    batch_selected = int(sys.argv[1])
    chosen_set = sys.argv[2]
    test_obj = test_code()
    latex_ret, im, with_atts = test_obj.run(batch_selected, chosen_set)
    cv2.imwrite('test_out.png', im * 255)
    for i in range(0, 10):
        cv2.imwrite('with_att' + str(i) + '.png', with_atts[i] * 255)
    print(latex_ret)

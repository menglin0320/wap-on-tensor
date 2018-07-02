import cv2
import tensorflow as tf

from Recognizer import MathFormulaRecognizer
from data_iterator import dataIterator
from util import *


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
        self.batch_size = 8
        self.valid_batch_size = 2
        self.finetune_encoder_after = -1
        # Evaluation Checkpoint
        self.nEvaImages = 300
        self.EvaEach = 2500
        self.SummaryEach = 1000
        self.device = "/gpu:0"
        self.batch_Imagesize = 250000
        self.valid_batch_Imagesize = 250000
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
        self.model = MathFormulaRecognizer(num_label=112, dim_hidden=128)
        self.logits, self.alpha_t, self.beta_t = self.model.build_greedy_eval()
        saver = tf.train.Saver(max_to_keep=10)
        self.sess = tf.Session()
        tf.reset_default_graph()
        saved_path = tf.train.latest_checkpoint(self.checkpoint_path)
        saver.restore(self.sess, saved_path)
        self.writer = tf.summary.FileWriter("/log", self.sess.graph)

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
        x, x_mask, y, y_mask = prepare_data(train[batch_picked, 0], train[batch_picked, 1])
        # for simplicity only test first image on the batch
        x = x[0:1, :, :, :]
        x_mask = x_mask[0:1, :, :]

        x = x[0:1, :, :, :]
        x_mask = x_mask[0:1, :, :]
        Words, Alphas, height, width, Beta = sess.run(
            [self.logits, self.alpha_t, model.feature_height, model.feature_width, self.beta_t],
            feed_dict={model.x: x, model.x_mask: x_mask,
                       model.is_train: True})

        Words = [w[0] for w in Words]
        for c in Words:
            print(self.worddicts_r[c])
        for i in range(0,10):
            im = np.reshape(Alphas[i],(height,width,1))
            norm_image = np.zeros((height,width))
            norm_image = cv2.normalize(im, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            print(norm_image)
            cv2.imwrite('attention' + str(i) + '.png', norm_image*255)
        self.writer.close()
        return Words, np.squeeze(x[0])



if __name__ == "__main__":
    test_obj = test_code()
    latex_ret, im = test_obj.run(0)
    cv2.imwrite('test_out.png', im * 255)
    print(latex_ret)

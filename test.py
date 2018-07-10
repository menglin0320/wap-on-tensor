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
        self.batch_Imagesize = 250000
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
        self.model = MathFormulaRecognizer(num_label=112, dim_hidden=128)
        self.alpha_t, self.beta_t, self.state, self.out, self.logit = self.model.build_eval()
        saver = tf.train.Saver(max_to_keep=10)
        self.sess = tf.Session()
        saved_path = tf.train.latest_checkpoint(self.checkpoint_path)
        tf.reset_default_graph()
        saver.restore(self.sess, saved_path)

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

    def get_all_candidates(self, model, sess, x, x_mask, n_cand = 5, max_len = 100):
        full_beam = []
        x = x[0:1, :, :, :]
        x_mask = x_mask[0:1, :, :]
        im = np.squeeze(x[0])

        cur_beam = []
        ignore = []
        # TODO this is just for initialization, there should be a
        # more elegant walk around
        vec_mask = 0
        information_tensor = 0
        height = 0
        width = 0
        for i in range(0, max_len):
            temp_structure = []
            if i == 0:
                Alpha, Beta, State, Logit, information_tensor, vec_mask, height, width = sess.run(
                [self.alpha_t, self.beta_t, self.state, self.logit, model.information_tensor, model.vec_mask,
                 model.feature_height, model.feature_width],
                feed_dict={model.x: x, model.x_mask: x_mask, model.is_train: False})
                orders = np.argsort(Logit[0])[::-1]
                probs = softmax(Logit[0])
                inds = orders[0:5]
                ordered_probs = np.log(probs[inds])

                for z in range(0, n_cand):
                    temp_structure.append([np.copy(Alpha), np.copy(Beta), np.copy(State), [inds[z]],
                                           ordered_probs[z]])
            else:
                for j in range(0, n_cand):
                    if not j in ignore:
                        previous_word = np.expand_dims(np.asarray(cur_beam[j][3][i - 1]), 0)
                        Beta = cur_beam[j][1]
                        State = cur_beam[j][2]
                        Alpha, Beta, State, Logit = sess.run([self.alpha_t, self.beta_t, self.state, self.logit],
                                                                 feed_dict=
                                                                 {model.information_tensor: information_tensor,
                                                                  model.vec_mask: vec_mask,
                                                                  model.in_beta_t: Beta, model.c: State[0],
                                                                  model.out: State[1],
                                                                  model.in_previous_word: previous_word,
                                                                  model.is_train: False})
                        orders = np.argsort(Logit[0])[::-1]
                        probs = softmax(Logit[0])
                        inds = orders[0:5]
                        ordered_probs = np.log(probs[inds])
                        with_att = attention_on_origin(np.reshape(Alpha, (height, width, 1)), im)
                        cv2.imwrite('with_att' + str(i) + '_' + str(j) + '.png', with_att * 255)
                        for z in range(0, n_cand):
                            temp_structure.append([np.copy(Alpha), np.copy(Beta), np.copy(State), cur_beam[j][3] + [inds[z]], ordered_probs[z] + cur_beam[j][4]])
            sorted_beams = sorted(temp_structure, key=lambda x: x[4], reverse=True)
            cur_beam = sorted_beams[0:5]
            ignore = []
            if i == 0:
                ignore = np.arange(5)[1:]
            else:
                for j in range(0, n_cand):
                    if (cur_beam[j][3][i] == 0):
                        full_beam.append([cur_beam[j][3], cur_beam[j][4] / len(cur_beam[j][3])])
                        ignore.append(i)
                        break
            if len(ignore) == n_cand:
                return full_beam
        return full_beam


    def run(self, batch_picked, chosen_set):
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
        all_cands = self.get_all_candidates(model, sess, x, x_mask)
        sorted_cands = sorted(all_cands, key=lambda x: x[1], reverse=True)
        chosen_beam = sorted_cands[0][0]
        str_list = []
        for i in range(0, len(chosen_beam)):
            str_list.append(self.worddicts_r[chosen_beam[i]])
        print(str(str_list))
        return chosen_beam, np.squeeze(x[0])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(sys.argv)
        raise ValueError('please give two args to specify image batch and set')
    batch_selected = int(sys.argv[1])
    chosen_set = sys.argv[2]
    test_obj = test_code()
    latex_ret, im = test_obj.run(batch_selected, chosen_set)
    cv2.imwrite('test_out.png', im * 255)
    print(latex_ret)

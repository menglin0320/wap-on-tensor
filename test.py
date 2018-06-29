import tensorflow as tf

from Recognizer import MathFormulaRecognizer
from data_iterator import dataIterator
from util import *

import cv2

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def show_image(im):
    plt.imshow(np.squeeze(im), cmap='gray')

#TODO save and check attention
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

    def run(self, batch_picked):
        #This code assumes that at least one character in the list
        #is recognized
        alpha_t, beta_t, state, logit = self.model.build_eval()
        train, train_uid_list = self.get_data('train')
        valid, valid_uid_list = self.get_data('test')
        valid = np.squeeze(valid)
        train = np.squeeze(train)
        sess = self.sess
        model = self.model
        x, x_mask, y, y_mask = prepare_data(valid[batch_picked, 0], valid[batch_picked, 1])
        # for simplicity only test first image on the batch
        x = x[0:1, :, :, :]
        x_mask = x_mask[0:1, :, :]

        n_cand = 5

        # on first iteration just pick the top n_cand candidates
        Alpha, Beta, State, Logit, information_tensor, vec_mask = sess.run(
            [alpha_t, beta_t, state, logit, model.information_tensor, model.vec_mask],
            feed_dict={model.x: x, model.x_mask: x_mask, model.is_train: False})
        orders = np.argsort(Logit[0])[::-1]
        probs = softmax(Logit[0])
        inds = orders[0:n_cand]
        probs = np.log(probs[inds])
        result = [[ind] for ind in inds]
        # second iteration
        temp_structure = []
        for i in range(0, 5):
            previous_word = np.expand_dims(np.asarray(result[i][0]), 0)

            tAlpha, tBeta, tState, tLogit = sess.run([alpha_t, beta_t, state, logit], feed_dict=
            {model.information_tensor: information_tensor, model.vec_mask: vec_mask,
             model.in_beta_t: Beta, model.in_state: State,
             model.in_previous_word: previous_word, model.is_train: False})

            orders = np.argsort(tLogit[0])[::-1]
            tprobs = softmax(tLogit[0])
            inds = orders[0:5]
            tprobs = np.log(tprobs[inds])
            #     print(tprobs)
            for j in range(0, 5):
                temp_structure.append(
                    [np.copy(tAlpha), np.copy(tBeta), np.copy(tState), result[i] + [inds[j]], tprobs[j] + probs[i]])
        a = sorted(temp_structure, key=lambda x: x[4], reverse=True)
        cur_beam = a[0:5]
        finish_flag = False
        latex_array = ''
        for j in range(0, 5):
            if (cur_beam[j][3][1] == 0):
                latex_array = cur_beam[j][3]
                finish_flag = True
                break
        iter_num = 2
        # following iterations
        while (True):
            temp_structure = []
            iter_num += 1
            if finish_flag:
                break
            for i in range(0, 5):
                previous_word = np.expand_dims(np.asarray(cur_beam[i][3][iter_num - 2]), 0)
                Alpha = cur_beam[i][0]
                Beta = cur_beam[i][1]
                State = cur_beam[i][2]
                tAlpha, tBeta, tState, tLogit = sess.run([alpha_t, beta_t, state, logit], feed_dict=
                {model.information_tensor: information_tensor, model.vec_mask: vec_mask,
                 model.in_beta_t: Beta, model.in_state: State,
                 model.in_previous_word: previous_word, model.is_train: False})
                orders = np.argsort(tLogit[0])[::-1]
                tprobs = softmax(tLogit[0])
                inds = orders[0:5]
                tprobs = np.log(tprobs[inds])
                #     print(tprobs)
                for j in range(0, 5):
                    temp_structure.append([np.copy(tAlpha), np.copy(tBeta), np.copy(tState), cur_beam[i][3] + [inds[j]],
                                           tprobs[j] + cur_beam[i][4]])
            a = sorted(temp_structure, key=lambda x: x[4], reverse=True)
            cur_beam = a[0:5]
            for j in range(0, 5):
                print(cur_beam[j][3])
                if (cur_beam[j][3][iter_num - 1] == 0):
                    latex_array = cur_beam[j][3]
                    finish_flag = True
                    break
        return latex_array, np.squeeze(x[0])

if __name__ == "__main__":
    test_obj = test_code()
    latex_ret,im = test_obj.run(4)
    cv2.imwrite('test_out.png', im*255)
    print(latex_ret)

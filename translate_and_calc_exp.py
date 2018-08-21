import sys
import cv2
import tensorflow as tf

from Recognizer import MathFormulaRecognizer
from data_iterator import dataIterator
from util import *


# A Dynamic Programming based Python program for edit
# distance problem
def editDistDP(str1, str2):
    # Create a table to store results of subproblems
    m = len(str1)
    n = len(str2)
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):

            # If first string is empty, only option is to
            # isnert all characters of second string
            if i == 0:
                dp[i][j] = j  # Min. operations = j

            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i  # Min. operations = i

            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                   dp[i - 1][j],  # Remove
                                   dp[i - 1][j - 1])  # Replace

    return dp[m][n]

def attention_on_origin(attention, im):
    height, width = im.shape
    aug_attention = cv2.resize(attention, (width, height))
    ret = np.zeros((height, width))
    ret = cv2.normalize(im + aug_attention, ret, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return ret

class translate():
    def __init__(self, mode):
        self.init_config()
        self.build_dict()
        self.load_model(mode)

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


    def load_model(self, mode):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.model = MathFormulaRecognizer(num_label=112, dim_hidden=128)
        if mode == 'greedy':
            self.logits, self.alpha_t, self.beta_t = self.model.build_greedy_eval()
        if mode == 'beam_search':
            self.alpha_t, self.beta_t, self.state, self.out, self.logit = self.model.build_eval()
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

    def run(self, chosen_set, mode):
        # This code assumes that at least one character in the list
        # is recognized
        captions = []
        train, train_uid_list = self.get_data('train')
        valid, valid_uid_list = self.get_data('test')
        valid = np.squeeze(valid)
        train = np.squeeze(train)
        sess = self.sess
        model = self.model
        n_train_batch = train.shape[0]
        n_valid_batch = valid.shape[0]
        batches_seen = 0
        if chosen_set == 'train':
            n_batch = n_train_batch
            using_set = train
        else:
            n_batch = n_valid_batch
            using_set = valid
        correct = 0
        total = 0
        ignore_label = 191
        if mode == 'greedy':
            for i in range(0, n_batch):
                x, x_mask, y, y_mask = prepare_data(using_set[i, 0], using_set[i, 1])
                y = np.transpose(y)
                y_mask = np.transpose(y_mask)
                Words, Alphas, height, width, Beta = sess.run(
                    [self.logits, self.alpha_t, model.feature_height, model.feature_width, self.beta_t],
                    feed_dict={model.x: x, model.x_mask: x_mask,
                               model.is_train: False})

                for j in range(y.shape[0]):
                    line = [w[j] for w in Words]
                    str_list = []
                    label_list = []
                    for c in line:
                        if c == 0:
                            break
                        str_list.append(self.worddicts_r[c])
                        label_list.append(c)
                    label_list = label_list[1:]
                    label_list.append(0)
                    str = ' '.join(str_list[1:])
                    GT = y[j]
                    sample_mask = y_mask[j]
                    y_inds = np.where(sample_mask == 1)[0]
                    GT = GT[y_inds]
                    GT = [label for label in GT if not label == ignore_label]
                    label_list = [label for label in label_list if not label == ignore_label]
                    dist = editDistDP(GT, label_list)
                    GT_str = ' '.join([self.worddicts_r[label] for label in GT])
                    #TODO change this
                    if dist != 0:
                        captions.append('[{}\t{}]'.format(valid_uid_list[batches_seen], str))
                        captions.append('[{}\t{}]'.format(valid_uid_list[batches_seen], GT_str))
                    else:
                        correct += 1
                        captions.append('{}\t{}'.format(valid_uid_list[batches_seen], str))
                        captions.append('{}\t{}'.format(valid_uid_list[batches_seen], GT_str))

                    batches_seen += 1

                    total += 1

            print(float(correct)/ float(total))
            return captions
        else:
            return [], []


def write_out(captions, out_file_name):
    with open(out_file_name, 'w') as f:
        for line in captions:
            f.write(line + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError('please give two args to specify image batch and set')
    chosen_set = sys.argv[1]
    mode = sys.argv[2]
    translate_obj = translate(mode)
    captions = translate_obj.run(chosen_set, mode)
    write_out(captions, 'incorrect_trans.txt')
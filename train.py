import tensorflow as tf
from Recognizer import MathFormulaRecognizer

from util import *

from data_iterator import dataIterator

class train_code:
    def __init__(self):
        self.config_initialize()
        self.initialize_model()
        self.load_data()

    def config_initialize(self):
        self.home_path = os.getcwd()
        self.checkpoint_path = os.path.join(self.home_path, 'save', 'model.ckpt')
        self.checkpoint_dir = os.path.join(self.home_path, 'save')
        self.max_iters = 100000
        self.batch_size = 16
        self.valid_batch_size = 8
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

    def initialize_model(self):
        checkpoint_dir = self.checkpoint_dir
        self.model = MathFormulaRecognizer(num_label=112, dim_hidden=128)
        self.loss, self.opt = self.model.build_train()
        self.saver = tf.train.Saver(max_to_keep=10)
        self.sess = tf.Session()
        saved_path = tf.train.latest_checkpoint(checkpoint_dir)
        self.start_step = 0
        if (saved_path):
            tf.reset_default_graph()
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
    def run(self):
        start_step = self.start_step
        saver = self.saver
        model = self.model
        sess = self.sess
        train = np.squeeze(self.train)
        n_train_img = train.shape[0]
        for i in range(start_step // n_train_img, self.n_epoch):
            rand_permute = np.arange(n_train_img)
            np.random.shuffle(rand_permute)
            saver.save(sess, self.checkpoint_path, global_step=i * rand_permute.shape[0])
            avg_loss = 0
            count = 0
            print('epoch: ', i)
            for j in range(0, rand_permute.shape[0]):
                x, x_mask, y, y_mask = prepare_data(train[rand_permute[j], 0], train[rand_permute[j], 1])
                y = np.transpose(y)
                y_mask = np.transpose(y_mask)
                _, Loss = sess.run([self.opt, self.loss], feed_dict={model.x: x, model.x_mask: x_mask, model.y: y, \
                                                           model.y_mask: y_mask, model.is_train: True, model.lr: 0.002})
                avg_loss = avg_loss + Loss
                count = count + 1
                if (not (j % 100)):
                    avg_loss = avg_loss / count
                    print(j, avg_loss)
                    count = 0
                    avg_loss = 0

if __name__ == "__main__":
    train = train_code()
    train.run()

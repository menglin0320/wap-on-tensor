import numpy as np
import os


def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])

    print ('total words/phones',len(lexicon))
    return lexicon

def prepare_data( images_x, seqs_y, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences

    heights_x = [s.shape[1] for s in images_x]
    widths_x = [s.shape[2] for s in images_x]
    lengths_y = [len(s) for s in seqs_y]

    n_samples = len(heights_x)
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_y = np.max(lengths_y) + 1

    x = np.zeros((n_samples, max_height_x, max_width_x, 1)).astype('float32')
    y = np.zeros((maxlen_y, n_samples)).astype('int64') # the <eol> must be 0 in the dict !!!
    x_mask = np.zeros((n_samples, max_height_x, max_width_x)).astype('float32')
    y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(images_x, seqs_y)):
        x[idx, :heights_x[idx], :widths_x[idx], :] = s_x.transpose((1,2,0)) / 255.
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask

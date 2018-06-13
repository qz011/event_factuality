from __future__ import print_function
from six.moves import xrange
import six.moves.cPickle as pickle

import gzip
import os

import numpy
import theano

import StringIO


def read_data_xy(readfilename):
    f = open(readfilename,"r")#.txt file

    x0 = []#list of list
    x1 = []#list of list
    x2 = []#list of list
    x3 = []#list of list
    y0 = []#list
    y1 = []#list

    allLines = f.readlines()

    '''for tmp_line in f:
        oneList = map(int,tmp_line.split(' '))
        x.append(oneList[:-1])
        y.append(oneList[-1])'''

    f.close()

    i = 0
    while i < len(allLines):
        x0.append(map(int ,allLines[i].split(' ')))
        i += 1

        x1.append(map(int ,allLines[i].split(' ')))
        i += 1

        x2.append(map(int ,allLines[i].split(' ')))
        i += 1

        x3.append(map(int ,allLines[i].split(' ')))
        i += 1

        labels = map(int ,allLines[i].split(' '))
        y0.append(labels[0])
        y1.append(labels[1])
        i += 1

    return x0, x1, x2, x3, y0, y1


def produce_data(readfilenames,savefilename):
    '''.txt files'''
    train_x0, train_x1, train_x2, train_x3, train_y0, train_y1 = read_data_xy(readfilename[0])
    valid_x0, valid_x1, valid_x2, valid_x3, valid_y0, valid_y1 = read_data_xy(readfilename[1])
    test_x0, test_x1, test_x2, test_x3, test_y0, test_y1  = read_data_xy(readfilename[2])


    data = ((train_x0, train_x1, train_x2, train_x3, train_y0, train_y1),
            (valid_x0, valid_x1, valid_x2, valid_x3, valid_y0, valid_y1),
            (test_x0, test_x1, test_x2, test_x3, test_y0, test_y1))#tuple

    f = open(savefilename,'wb')
    pickle.dump(data,f)
    f.close()  


    

def prepare_data(seqs, addIdxNum=0, maxlen=None, win_size=1):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    '''if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None'''

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    '''
    n_samples : numbers of sentences
    '''

    x = numpy.zeros((maxlen, n_samples)).astype('int32')
    x_mask = numpy.zeros(((maxlen - addIdxNum) / win_size, n_samples)).astype(theano.config.floatX)

    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:((lengths[idx] - addIdxNum) / win_size), idx] = 1.

    #labels = numpy.asarray(labels).astype('int32')

    return x, x_mask#, labels


def load_data(path, n_words=5000, valid_portion=0.2, maxlen=None,
              sort_by_len=False):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset 
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

   

    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    '''if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            #if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y'''


    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x0, test_set_x1, test_set_x2, test_set_x3, test_set_y0, test_set_y1 = test_set
    valid_set_x0, valid_set_x1, valid_set_x2, valid_set_x3, valid_set_y0, valid_set_y1 = valid_set
    train_set_x0, train_set_x1, train_set_x2, train_set_x3, train_set_y0, train_set_y1 = train_set

    '''train_set_x1 = remove_unk(train_set_x1)
    train_set_x2 = remove_unk(train_set_x2)
    train_set_x3 = remove_unk(train_set_x3)
    
    valid_set_x1 = remove_unk(valid_set_x1)
    valid_set_x2 = remove_unk(valid_set_x2)
    valid_set_x3 = remove_unk(valid_set_x3)
    
    test_set_x1 = remove_unk(test_set_x1)
    test_set_x2 = remove_unk(test_set_x2)
    test_set_x3 = remove_unk(test_set_x3)'''

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x0 = [test_set_x0[i] for i in sorted_index]
        test_set_x1 = [test_set_x1[i] for i in sorted_index]
        test_set_x2 = [test_set_x2[i] for i in sorted_index]
        test_set_x3 = [test_set_x3[i] for i in sorted_index]
        test_set_y0 = [test_set_y0[i] for i in sorted_index]
        test_set_y1 = [test_set_y1[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x0 = [valid_set_x0[i] for i in sorted_index]
        valid_set_x1 = [valid_set_x1[i] for i in sorted_index]
        valid_set_x2 = [valid_set_x2[i] for i in sorted_index]
        valid_set_x3 = [valid_set_x3[i] for i in sorted_index]
        valid_set_y0 = [valid_set_y0[i] for i in sorted_index]
        valid_set_y1 = [valid_set_y1[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x0 = [train_set_x0[i] for i in sorted_index]
        train_set_x1 = [train_set_x1[i] for i in sorted_index]
        train_set_x2 = [train_set_x2[i] for i in sorted_index]
        train_set_x3 = [train_set_x3[i] for i in sorted_index]
        train_set_y0 = [train_set_y0[i] for i in sorted_index]
        train_set_y1 = [train_set_y1[i] for i in sorted_index]

    train = (train_set_x0, train_set_x1, train_set_x2, train_set_x3, train_set_y0, train_set_y1)
    valid = (valid_set_x0, valid_set_x1, valid_set_x2, valid_set_x3, valid_set_y0, valid_set_y1)
    test  = (test_set_x0, test_set_x1, test_set_x2, test_set_x3, test_set_y0, test_set_y1)

    return train, valid, test

def read_embedding_file_to_get_matrix(filename, savefilename):
    file_obj = open(filename,"r")
    embeddings = []
    
    for tmp_line in file_obj:
         one_embedding = numpy.loadtxt(StringIO.StringIO(tmp_line))#matrix
         embeddings.append(one_embedding)

    matrix = numpy.asarray(embeddings)

    file_obj.close()

    f = open(savefilename,'wb')
    pickle.dump(matrix,f)
    f.close()

    return matrix

def read_gz_file(filename):
    f = gzip.open(filename,'rb')
    data = pickle.load(f)
    f.close()

    return data


if __name__ == '__main__':
    
    #d = read_data_xy("../train_idx.txt")
    #print(type(d[3]))

    ##############################################################

    '''readfilename = ["../train_idx.txt",
                    "../valid_idx.txt",
                    "../test_idx.txt"]
    savefilename = '../mydata.pkl'
    produce_data(readfilename,savefilename)'''


    '''m_arr = read_embedding_file_to_get_matrix("../word_embed.txt",
                                              "../matrix.pkl")
    print(m_arr.shape)'''

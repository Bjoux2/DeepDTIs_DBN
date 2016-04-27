__author__ = 'wenming'

import random
import pickle
import numpy
import numpy as np
import theano
import theano.tensor as T
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler

def load_labeled_data(data_path):
    # base_path = '/home/wenming/my_task_ubuntu/DTIs/data2016/'
    # output = open(base_path + 'DATA FOR MODELING/TRAINING_DATA.pkl')
    output = open(data_path)
    TRAINING_DATA = pickle.load(output)
    output.close()
    print 'Training_data load finished'
    # X = normalize(TRAINING_DATA['X'], axis=0)  ## normalize
    min_max_scaler = MinMaxScaler()  ## min max scaler
    X = min_max_scaler.fit_transform(TRAINING_DATA['X'])  ## min max scaler
    Y = TRAINING_DATA['Y']
    print X.shape
    print np.max(X)
    print Y.shape
    tv_set_x, test_set_x, tv_set_y, test_set_y = train_test_split(X, Y, test_size=0.25, random_state=123)
    train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(tv_set_x, tv_set_y, test_size=0.2, random_state=123)

    DTI = (
        (
            train_set_x,
            train_set_y
        ),
        (
            valid_set_x,
            valid_set_y
        ),
        (
            test_set_x,
            test_set_y
        )
    )

    train_set, valid_set, test_set = DTI

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def load_labeled_data_rand(data_path, random_state=123):
    # base_path = '/home/wenming/my_task_ubuntu/DTIs/data2016/'
    # output = open(base_path + 'DATA FOR MODELING/TRAINING_DATA.pkl')
    output = open(data_path)
    TRAINING_DATA = pickle.load(output)
    output.close()
    print 'Training_data load finished'
    # X = normalize(TRAINING_DATA['X'], axis=0)  ## normalize
    min_max_scaler = MinMaxScaler()  ## min max scaler
    X = min_max_scaler.fit_transform(TRAINING_DATA['X'])  ## min max scaler
    Y = TRAINING_DATA['Y']
    print X.shape
    print np.max(X)
    print Y.shape
    tv_set_x, test_set_x, tv_set_y, test_set_y = train_test_split(X, Y, test_size=0.25, random_state=random_state)
    train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(tv_set_x, tv_set_y, test_size=0.2, random_state=123)

    DTI = (
        (
            train_set_x,
            train_set_y
        ),
        (
            valid_set_x,
            valid_set_y
        ),
        (
            test_set_x,
            test_set_y
        )
    )

    train_set, valid_set, test_set = DTI

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def load_labeled_data_rand_generate(data_path, rand=1):
    # base_path = '/home/wenming/my_task_ubuntu/DTIs/data2016/'
    # output = open(base_path + 'DATA FOR MODELING/TRAINING_DATA%s.pkl' % rand)
    output = open(data_path)
    TRAINING_DATA = pickle.load(output)
    output.close()
    print 'Training_data load finished'
    # X = normalize(TRAINING_DATA['X'], axis=0)  ## normalize
    min_max_scaler = MinMaxScaler()  ## min max scaler
    X = min_max_scaler.fit_transform(TRAINING_DATA['X'])  ## min max scaler
    Y = TRAINING_DATA['Y']
    print X.shape
    print np.max(X)
    print Y.shape
    tv_set_x, test_set_x, tv_set_y, test_set_y = train_test_split(X, Y, test_size=0.25, random_state=1)
    train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(tv_set_x, tv_set_y, test_size=0.2, random_state=1)

    DTI = (
        (
            train_set_x,
            train_set_y
        ),
        (
            valid_set_x,
            valid_set_y
        ),
        (
            test_set_x,
            test_set_y
        )
    )

    train_set, valid_set, test_set = DTI

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval



if __name__=="__main__":
    load_labeled_data()


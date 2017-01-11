__author__ = 'wenming'
import numpy
import numpy as np
import timeit
from load_data import load_labeled_data
from DBN_for_DTIs import DBN
import sys, os
import theano
import theano.tensor as T
import pickle
from sklearn import metrics

def dbn_training(
    finetune_lr=0.1,
    pretrain_lr=0.1,
    batch_size=128,
    hidden_layers_sizes=[2000, 2000, 2000, 2000]
):
    ##  Parameters of DBN CLASS

    pretraining_epochs = 100
    k = 1
    training_epochs = 1000
    n_ins = 14564
    # n_ins = 456
    datasets = load_labeled_data()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)



    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=n_ins,
              hidden_layers_sizes=hidden_layers_sizes,
              n_outs=2)


    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))  ## output: cost
            # print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            # print numpy.mean(c)
    end_time = timeit.default_timer()
    # end-snippet-2
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.)) ## The pretraining code for file DBN_adapt.py ran for 272.98m
    pretrain_time = (end_time - start_time) / 60.

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model, train_predprob_model, train_predclass_model, valid_predprob_model, valid_predclass_model, test_predprob_model, test_predclass_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf  # infinite
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)  # the iteration is trainng dataset.
                # print(
                #     'epoch %i, minibatch %i/%i, validation error %f %%'
                #     % (
                #         epoch,
                #         minibatch_index + 1,
                #         n_train_batches,
                #         this_validation_loss * 100.
                #     )
                # )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    # save the best model
                    # with open('best_model_dbn.pkl', 'wb') as f:
                    #     pickle.dump(dbn, f)

                    # test predprab on the test set
                    train_predprob = train_predprob_model()
                    train_predclass = train_predclass_model()
                    valid_predprob = valid_predprob_model()
                    valid_predclass = valid_predclass_model()
                    test_predprob = test_predprob_model()
                    test_predclass = test_predclass_model()

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'obtained at iteration %i, '
            'with test performance %f %%'
        ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))

    finetune_time = (end_time - start_time) / 60.
    y_train_true = np.array(train_set_y.eval())
    y_valid_true = np.array(valid_set_y.eval())
    y_test_true = np.array(test_set_y.eval())


    auc_train = metrics.roc_auc_score(y_train_true, train_predprob[:, 1])
    acc_train = metrics.accuracy_score(y_train_true, train_predclass)
    tpr_train = metrics.recall_score(y_train_true, train_predclass)
    tnr_train = (acc_train * y_train_true.shape[0] - list(y_train_true).count(1) * tpr_train)/list(y_train_true).count(0)

    auc_valid = metrics.roc_auc_score(y_valid_true, valid_predprob[:, 1])
    acc_valid = metrics.accuracy_score(y_valid_true, valid_predclass)
    tpr_valid = metrics.recall_score(y_valid_true, valid_predclass)
    tnr_valid = (acc_valid * y_valid_true.shape[0] - list(y_valid_true).count(1) * tpr_valid)/list(y_valid_true).count(0)

    auc_test = metrics.roc_auc_score(y_test_true, test_predprob[:, 1])
    acc_test = metrics.accuracy_score(y_test_true, test_predclass)
    tpr_test = metrics.recall_score(y_test_true, test_predclass)
    tnr_test = (acc_test * y_test_true.shape[0] - list(y_test_true).count(1) * tpr_test)/list(y_test_true).count(0)
    # print 'auc_train, acc_train, tpr_train, tnr_train : ', auc_train, acc_train, tpr_train, tnr_train
    # print 'auc_valid, acc_valid, tpr_valid, tnr_valid : ', auc_valid, acc_valid, tpr_valid, tnr_valid
    # print 'auc_test, acc_test, tpr_test, tnr_test : ', auc_test, acc_test, tpr_test, tnr_test


    return auc_train, acc_train, tpr_train, tnr_train,auc_valid, acc_valid, tpr_valid, tnr_valid,auc_test, acc_test, tpr_test, tnr_test, pretrain_time, finetune_time

if __name__ == "__main__":
    auc_train, acc_train, tpr_train, tnr_train,auc_valid, acc_valid, tpr_valid, tnr_valid,auc_test, acc_test, tpr_test, tnr_test, pretrain_time, finetune_time = dbn_prediction()
    print auc_train, acc_train, tpr_train, tnr_train,auc_valid, acc_valid, tpr_valid, tnr_valid,auc_test, acc_test, tpr_test, tnr_test, pretrain_time, finetune_time

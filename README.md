# DeepDTIs
Please cite 'Deep learning-based drug-target interaction prediction'.

The Deep belief net (DBN) code was rewritten from www.deeplearning.net 

The code in 'code_sklearn-like' is recommended, the usage of the DBN here is similar to sklean:

## pseudo-example:

from DBN_wm import DBN

dbn_classifier = DBN()

dbn_classifier.pretraining(train_x) 

dbn_classifier.finetuning(train_x, train_y, valid_x, valid_y)    # the valid set is used to optimize the parameters

y_pred = dbn_classifier.predict(test_y)

>>>More detaild example, see test_DBN.py

## Dependencies:

1), Python 2.7, latest version

2), Theano, latest version

## Further reading: 

1), Deep-Learning-in-Bioinformatics-Papers-Reading-Roadmap

https://github.com/Bjoux2/Deep-Learning-in-Bioinformatics-Papers-Reading-Roadmap

from DBN_wm import DBN
from load_data import load_data
import pickle
from sklearn.preprocessing import MinMaxScaler
from utilis import shared_dataset_x, shared_dataset_y
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


XX, YY = load_data()
min_max_scaler = MinMaxScaler()  ## min max scaler
min_max_scaler.fit(XX)
XX = min_max_scaler.transform(XX)
print XX.shape
print YY.shape

f = open('/home/thl/my_task/for-data3/data3/ml_results/DBN_20_times.txt', 'w')
f.write('random_seed' + ',' + 'AUC' + ',' + 'ACC' + ',' + 'TPR' + ',' + 'TNR' + '\n')
for rand_seed in range(1234, 1254):
    np.random.seed(rand_seed)
    p = np.random.permutation(XX.shape[0])
    X = XX[p].astype(np.float32)
    Y = YY[p]

    train_set_x = shared_dataset_x(X[:5000])
    valid_set_x = shared_dataset_x(X[5000:6500])
    test_set_x = X[6500:]

    train_set_y = shared_dataset_y(Y[:5000])
    valid_set_y = shared_dataset_y(Y[5000:6500])
    test_set_y = Y[6500:]

    # dbn = SdA(hidden_layers_sizes=[1500, 2000, 2000, 2000, 1500], pretrain_epochs=120, finetune_epochs=800)
    # dbn = SdA(hidden_layers_sizes=[800, 600], pretrain_epochs=12, finetune_epochs=80)
    dbn = DBN()
    dbn.pretraining(train_set_x)
    dbn.finetuning(train_set_x, train_set_y, valid_set_x, valid_set_y)

    with open('/home/thl/my_task/for-data3/data3/DBN_models_pkl/DBN_best_model_%s.pkl' % str(rand_seed), 'wb') \
            as f_dbn:
        pickle.dump(dbn, f_dbn)

    test_y_pred_proba = dbn.predict_proba(test_set_x)[:, 1]
    test_y_pred = dbn.predict(test_set_x)

    acc = accuracy_score(test_set_y, test_y_pred)
    auc = roc_auc_score(test_set_y, test_y_pred_proba)

    tpr = list(test_y_pred[test_set_y == 1]).count(1) / float(list(test_set_y).count(1))
    tnr = list(test_y_pred[test_set_y == 0]).count(0) / float(list(test_set_y).count(0))
    print auc, acc, tpr, tnr
    f.write('random_seed' + ',' + str(auc) + ',' + str(acc) + ',' + str(tpr) + ',' + str(tnr) + '\n')

f.close()

#  this script is used for grid search

from DBN_training import dbn_training
f = open('/home/wenming/my_task_ubuntu/DTIs/data2016/dbn_superviased_grid_search_results20160326.csv', 'w')
for finetune_lr in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
    for pretrain_lr in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        for batch_size in [10, 32, 64, 128]:
            for hidden_layers_sizes in [[3000, 500], [3000, 3000, 500], [4000, 4000, 4000, 1000], [5000, 5000, 1000]]:
                auc_train, acc_train, tpr_train, tnr_train,auc_valid, acc_valid, tpr_valid, tnr_valid, auc_test, acc_test, \
                tpr_test, tnr_test, pretrain_time, finetune_time = dbn_training(finetune_lr, pretrain_lr, batch_size,
                                                                                hidden_layers_sizes)
                f.write(str(finetune_lr) + ',' + str(pretrain_lr) + ',' + str(batch_size) + ',' + str(auc_test) + ',' +
                        str(acc_test) + ',' + str(tpr_test) + ',' + str(tnr_test) + ',' + str(pretrain_time) + ',' +
                        str(finetune_time) + ',' + str(hidden_layers_sizes) + '\n')

                print finetune_lr, pretrain_lr, batch_size, hidden_layers_sizes
                print 'auc_train, acc_train, tpr_train, tnr_train : ', auc_train, acc_train, tpr_train, tnr_train
                print 'auc_valid, acc_valid, tpr_valid, tnr_valid : ', auc_valid, acc_valid, tpr_valid, tnr_valid
                print 'auc_test, acc_test, tpr_test, tnr_test : ', auc_test, acc_test, tpr_test, tnr_test
f.close()

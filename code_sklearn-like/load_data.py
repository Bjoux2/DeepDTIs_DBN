import cPickle as pk
import numpy as np
base_path = '/home/thl/my_task/for-data3/'


def load_data():
    f = open(base_path + 'data3/pkl/CLASH_N_features_dic_all.pkl')
    CLASH_N_features_dic = pk.load(f)
    f.close()

    f = open(base_path + 'data3/pkl/CLASH_features_dic_all.pkl')
    CLASH_features_dic = pk.load(f)
    f.close()

    f = open(base_path + 'data3/pkl/Mark_features_dic_all.pkl')
    Mark_features_dic = pk.load(f)
    f.close()

    f = open(base_path + 'data3/pkl/Mark_N_features_dic_all.pkl')
    Mark_N_features_dic = pk.load(f)
    f.close()

    X_p = []
    for key in CLASH_features_dic.keys():
        X_p.append(CLASH_features_dic[key].values())
    for key in Mark_features_dic.keys():
        X_p.append(Mark_features_dic[key].values())

    X_n = []
    for key in CLASH_N_features_dic.keys():
        X_n.append(CLASH_N_features_dic[key].values())
    for key in Mark_N_features_dic.keys():
        X_n.append(Mark_N_features_dic[key].values())

    X = np.array(X_p + X_n)
    Y = np.array([1]*len(X_p) + [0]*len(X_n))
    return X, Y


if __name__ == '__main__':
    load_data()
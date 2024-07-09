import heapq
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import joblib

data = []
target = []


def train(data, target, save_path):
    num = 0
    maxx = 0
    kfold = KFold(n_splits=9)
    while True:
        if num > 20:
            break
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=0.2)

        gbm = lgb.LGBMClassifier(objective='multiclass', num_leaves=31, learning_rate=0.05, n_estimators=20)
        # gbm = lgb.LGBMClassifier(objective='binary', num_leaves=31, learning_rate=0.05, n_estimators=20)
        gbm.fit(Xtrain, Ytrain)
        score = gbm.score(Xtest, Ytest)
        result = cross_val_score(gbm, data, target, cv=kfold)
        num += 1

        # -----------
        y_pred_prob = gbm.predict_proba(Xtest)
        y_class = gbm.classes_
        y_pred_top3_lists = []
        for pred in y_pred_prob:
            tmp_list = zip(y_class, pred)
            zip_list = sorted(tmp_list, key=lambda x: x[1])[::-1]
            y_pred_top3 = list(zip(*zip_list))[0][:3]
            y_pred_top3_lists.append(y_pred_top3)

        hit_num = 0
        for i in range(len(Ytest)):
            if Ytest[i] in y_pred_top3_lists[i]:
                hit_num += 1
        print('top-3 acc = ', hit_num / len(Ytest) * 100)
        if score > maxx:
            # print("update!")
            maxx = score
            joblib.dump(gbm, save_path)
    print('max score : ', maxx)


def def_some(dir):
    list = os.listdir(dir)
    for file in list:
        path = os.path.join(dir, file)
        with open(path, 'r', encoding='gbk') as f:
            lines = f.read().split('\n')
        tree_start = lines.index('tree') + 1
        pandas_end = lines.index('pandas_categorical:null')
        new_file_path = os.path.join(dir, 'new_' + file)
        new_lines = lines[tree_start:pandas_end]
        with open(new_file_path, 'w') as f:
            f.write(new_lines)


if __name__ == '__main__':
    edge_path = r'\SplitDataset\train_X'
    label_path = r'\SplitDataset\train_Y'

    vector_list = os.listdir(edge_path)
    label_list = os.listdir(label_path)

    for i in range(len(vector_list)):
        vector_path_i = os.path.join(edge_path, vector_list[i])
        label_path_i = os.path.join(label_path, label_list[i])
        train_vectors = np.load(vector_path_i, allow_pickle=True)
        train_labels = np.load(label_path_i, allow_pickle=True)
        wxh = vector_list[i].split('_')[1]
        w = wxh.split('x')[0]
        h = wxh.split('x')[1]
        hxw = h + 'x' + w
        save_path = os.path.join(r'\SplitDataset\model', hxw + '.txt')
        if not os.path.exists(save_path):
            print('------' + wxh + '-----------')
            try:
                train(train_vectors, train_labels, save_path)
            except:
                continue

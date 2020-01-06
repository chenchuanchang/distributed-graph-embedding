import numpy as np
import random
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from walks import random_walk, aggregate

def emb_value_cos(x, y):
    a_b = np.dot(x, y)
    a = np.fabs(sum(x ** 2) ** 0.5)
    b = np.fabs(sum(y ** 2) ** 0.5)
    a_dev_b = a_b / (a * b)
    return a_dev_b.reshape(1)

def emb_value_weight_2(x, y):
    a_b = np.dot(x, y)
    return a_b.reshape(1)

def node_classification(x, x_label, y, y_lable, emb):
    x = emb[x]
    y = emb[y]
    cls = OneVsRestClassifier(svm.SVC(kernel='linear', random_state=np.random.RandomState(0)))
    cls.fit(x, x_label)
    y_pre = cls.predict(y)
    macro_f1 = f1_score(y_lable, y_pre, average='macro')
    micro_f1 = f1_score(y_lable, y_pre, average='micro')
    return np.array([macro_f1, micro_f1])

def link_prediction(data, emb):
    pre = []
    label = []
    for i in range(data.shape[1]):
        pre.append(emb_value_weight_2(emb[data[0][i][0]], emb[data[0][i][1]]))
        pre.append(emb_value_weight_2(emb[data[1][i][0]], emb[data[1][i][1]]))
        label.append(1)
        label.append(0)
    cls = LogisticRegression(C=10000)
    cls.fit(pre, label)
    rel_pre = cls.predict(pre)

    AUC = roc_auc_score(label, rel_pre)
    return AUC

def negative_sampling(unigram, ns):
    neg_set = []
    while len(neg_set) < ns:
        neg = random.random()
        L = 0
        R = len(unigram) - 1
        while(L < R):
            mid = (L+R)//2
            if unigram[mid] < neg:
                L = mid + 1
            else:
                R = mid
        neg_set.append(L)
    return neg_set

def traning_data(hp, G, step):
    if hp.method == 'deepwalk':
        split = hp.node_num//hp.worker
        node_list = list(G.keys())[:-1][split*hp.task : min(hp.node_num, split*(hp.task+1))]
        idx = hp.batch_size * step
        node_len = len(node_list)
        node_list = [node_list[(idx+i)%node_len] for i in range(hp.batch_size)]
        walk = random_walk(hp, G, node_list)
        xc_0 = []
        xc_1 = []
        xuc_0 = []
        xuc_1 = []
        neg_list = G['negative']
        for w in walk:
            for i in range(hp.cs, len(w)-hp.cs):
                for k in range(i-hp.cs, i+hp.cs+1):
                    if k == i:
                        continue
                    xc_0.append(w[k])
                    xc_1.append(w[i])
                    neg = negative_sampling(neg_list, hp.ns)
                    for q in neg:
                        xuc_0.append(w[k])
                        xuc_1.append(q)
        return np.array(xc_0), np.array(xc_1), np.array(xuc_0), np.array(xuc_1)
    else:
        split = hp.node_num // hp.worker
        node_list = list(G.keys())[:-1][split * hp.task: min(hp.node_num, split * (hp.task + 1))]
        while len(node_list)<split:
            node_list.append(random.sample(node_list, 1)[0])
        xa = aggregate(hp, G, node_list)
        xs = np.array([int(node_list[i])-hp.task*split for i in range(int(split * hp.semi_size))])
        ys = np.zeros((int(split*hp.semi_size), hp.labels))
        for i in range(int(split*hp.semi_size)):
            ys[i][int(G[node_list[i]]["label"])] = 1
        return xa, xs, ys

def testing_data(hp, G):
    node_list = list(G.keys())[:-1]
    if hp.method == 'deepwalk':
        val = np.zeros((2, hp.test_size, 2))
        cnt_1, cnt_2 = 0, 0
        while cnt_1<hp.test_size or cnt_2<hp.test_size:
            x, y = random.sample(node_list, 2)
            if x == y:
                continue
            if y not in G[x]["edge"]:
                if cnt_2<hp.test_size:
                    val[1][cnt_2][0] = x
                    val[1][cnt_2][1] = y
                    cnt_2 += 1
            elif cnt_1<hp.test_size:
                val[0][cnt_1][0] = x
                val[0][cnt_1][1] = y
                cnt_1 += 1
        return val
    else:
        node_list = node_list + node_list[:hp.worker+1]
        split = hp.node_num // hp.worker
        span = int(split * hp.semi_size)
        x = np.zeros((span*hp.worker))
        x_label = np.zeros((span*hp.worker))
        y = np.zeros((hp.worker*(split - span)))
        y_label = np.zeros((hp.worker*(split - span)))
        x_cnt = 0
        y_cnt = 0
        for i in range(hp.worker*split):
            tem = i%split
            if tem>=0 and tem<span:
                x[x_cnt] = int(node_list[i])
                x_label[x_cnt] = int(G[node_list[i]]["label"])
                x_cnt+=1
            else:
                y[y_cnt] = int(node_list[i])
                y_label[y_cnt] = int(G[node_list[i]]["label"])
                y_cnt+=1
        return x, x_label, y, y_label


import pandas as pd
import math
from collections import Counter
import os
import matplotlib
import sys
import numpy as np

def summary(path):
    data = pd.read_csv(path)
    text = list(data['text'])
    label = data['label']
    sample_num = len(text)
    # 
    words_len = list(map(len_def, text))
    max_len = max(words_len)
    min_len = min(words_len)
    avg_len = sum(words_len) / sample_num
    words_len_cut = list(map(len_def1, text))
    avg_len_cut = sum(words_len_cut) / sample_num

    clusters = Counter(label)
    clusters_num = len(clusters)
    cluster_r = max(clusters.values()) / min(clusters.values())
    print('cluster_distribution:', sorted(np.array(list(clusters.values())) / sample_num))
    print('cluster_num:', clusters_num)
    #print('clusters:', clusters)
    print('sample_num:', sample_num)
    #print('max len:', max_len)
    #print('min len:', min_len)
    #print('cluster max/min:', cluster_r)
    #print('avg len:', avg_len)
    #print('avg_len_cut:', avg_len_cut)
    
def len_def(text):
    return len(text.split())

def len_def1(text):
    tmp = len(text.split())
    return tmp if tmp <= 32 else 32

def get_text(dataname, datapath='/home/hml/text_cluster/sccl/AugData/augmented-datasets', target='./agnews.csv'):
    source = os.path.join(datapath, dataname+".csv")
    train_data = pd.read_csv(source)
    data1 = train_data[train_data['label'] == 3][:2000]
    data2 = train_data[train_data['label'] == 2][:1200]
    data3 = train_data[train_data['label'] == 1][:800]
    new_data = pd.concat([data1, data2, data3])
    new_data.to_csv(target, index=False)
    return

if __name__ == '__main__':
    if len(sys.argv) == 1:
        data_name = 'searchsnippets_trans_subst_10'
        get_text(dataname=data_name, target='./search.csv')
    elif sys.argv[1] == 'sum':
        root_path = '/home/hml/text_cluster/sccl/AugData/augmented-datasets'
        data_names = ['agnewsdataraw-8000_trans_subst_10.csv', 'stackoverflow_trans_subst_10.csv', 'biomedical_trans_subst_10.csv', 'searchsnippets_trans_subst_10.csv', 'TS_trans_subst_10.csv', 'T_trans_subst_10.csv', 'S_trans_subst_10.csv', 'tweet-original-order_trans_subst_10.csv']
        #root_path = '/home/hml/text_cluster'
        #data_names = ['agnews.csv']
        for data_name in data_names:
            print(data_name)
            path = os.path.join(root_path, data_name)
            summary(path)
    elif sys.argv[1] == 'data':
        #data_name = 'agnewsdataraw-8000_trans_subst_10'
        data_name = 'searchsnippets_trans_subst_10'
        get_text(dataname=data_name, target='./search.csv')
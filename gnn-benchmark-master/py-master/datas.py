import pickle
import random

import torch
import numpy as np
from utils import normalize_adj
from scipy import sparse as sp
import sklearn.preprocessing as preprocess


def load_data_by_name(name):
    DATASET = name
    # print(f'load data by name: {name}')
    if name != 'wiki':
        with open(f"data/ind.{DATASET}.trueTotalX", "rb+") as f:
            trueTotalX = pickle.load(f, encoding='latin1')
        with open(f"data/ind.{DATASET}.trueTotalY", "rb+") as f:
            trueTotalY = pickle.load(f, encoding='latin1')
        with open(f"data/ind.{DATASET}.adj", "rb+") as f:
            adj = pickle.load(f, encoding='latin1')
        adj = torch.FloatTensor(adj.toarray())
        features = torch.FloatTensor(trueTotalX)
        labels = torch.LongTensor(np.where(trueTotalY)[1])

    else:
        adj, features, labels = load_wiki()
        adj = torch.FloatTensor(adj.toarray())
        labels = torch.LongTensor(labels)

    if DATASET == 'cora':
        idx_train = torch.LongTensor(range(140))
        idx_val = torch.LongTensor(range(200, 700))
        idx_test = torch.LongTensor(range(700, 1700))
    elif DATASET == 'citeseer':
        idxList=list(range(0,1700))
        random.shuffle(idxList)
        idx_train = torch.LongTensor(idxList[0:120])
        idx_val = torch.LongTensor(idxList[200:700])
        idx_test = torch.LongTensor(idxList[700:1700])
        # idx_train = torch.LongTensor(range(120,240))
        # idx_val = torch.LongTensor(range(240, 340))
        # idx_test = torch.LongTensor(range(700, 1700))
    elif DATASET == 'pubmed':
        idxList = list(range(0, 1700))
        random.shuffle(idxList)
        idx_train = torch.LongTensor(idxList[0:60])
        idx_val = torch.LongTensor(idxList[200:700])
        idx_test = torch.LongTensor(idxList[700:1700])
        # idx_train = torch.LongTensor(range(60))
        # idx_val = torch.LongTensor(range(200, 700))
        # idx_test = torch.LongTensor(range(700, 1700))
    elif DATASET == 'wiki':
        idx_train = torch.LongTensor(range(150))
        idx_val = torch.LongTensor(range(200, 700))
        idx_test = torch.LongTensor(range(700, 1700))
    #
    # if DATASET!='wiki':
    #     adj = sp.csc_matrix(adj)
    #     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #     adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    #     adj = torch.FloatTensor(np.array(adj.todense()))

    one_vec = torch.ones_like(adj)
    zero_vec = torch.zeros_like(adj)
    adj_noNormal = torch.where(adj > 0, one_vec, zero_vec)
    B = torch.mul(adj_noNormal, adj_noNormal)
    M = torch.mm(B, B)
    M = torch.mul(M, B)

    # M = (1 - beta) * adj_noNormal + beta * M
    # M = np.array(M)
    # M = sp.csc_matrix(M)
    # M = normalize_adj(M + sp.eye(M.shape[0]))
    # M = torch.FloatTensor(np.array(M.todense()))

    return M, adj, features, labels, idx_train, idx_val, idx_test


def load_wiki():
    f = open('data/graph.txt', 'r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()

        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()
    ##print(len(adj))

    f = open('data/group.txt', 'r')
    label = []
    for line in f.readlines():
        line = line.split()
        label.append(int(line[1]))
    f.close()

    f = open('data/tfidf.txt', 'r')
    fea_idx = []
    fea = []
    adj = np.array(adj)
    adj = np.vstack((adj, adj[:, [1, 0]]))
    adj = np.unique(adj, axis=0)

    labelset = np.unique(label)
    labeldict = dict(zip(labelset, range(len(labelset))))
    label = np.array([labeldict[x] for x in label])
    adj = sp.csr_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(len(label), len(label)))

    for line in f.readlines():
        line = line.split()
        fea_idx.append([int(line[0]), int(line[1])])
        fea.append(float(line[2]))
    f.close()

    fea_idx = np.array(fea_idx)
    features = sp.csr_matrix((fea, (fea_idx[:, 0], fea_idx[:, 1])), shape=(len(label), 4973)).toarray()
    scaler = preprocess.MinMaxScaler()
    # features = preprocess.normalize(features, norm='l2')
    features = scaler.fit_transform(features)
    features = torch.FloatTensor(features)

    return adj, features, label

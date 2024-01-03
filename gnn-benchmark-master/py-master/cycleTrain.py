from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt

import conf
from utils import load_data, accuracy
from datas import load_data_by_name
from models import GAT, SpGAT


def autoTrain(dataset, beta, modeltype):
    torch.cuda.empty_cache()
    conf.model = modeltype
    conf.beta = beta
    conf.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default='sparse', help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=55555, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    files = glob.glob('*.pkl')
    for file in files:
        os.remove(file)

    args = parser.parse_args()
    # print(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    # M, adj, features, labels, idx_train, idx_val, idx_test = load_data(path="./data/cora/", dataset="cora")
    M, adj, features, labels, idx_train, idx_val, idx_test = load_data_by_name(dataset)

    # Model and optimizer
    if args.sparse == 'sparse':
        model = SpGAT(nfeat=features.shape[1],
                      nhid=args.hidden,
                      nclass=int(labels.max()) + 1,
                      dropout=args.dropout,
                      nheads=args.nb_heads,
                      alpha=args.alpha)
    elif args.sparse == 'dense':
        model = GAT(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha)
    elif args.sparse == 'MLPGAT':
        pass
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        M = M.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    M, features, adj, labels = Variable(M), Variable(features), Variable(adj), Variable(labels)

    # list_beta=[]
    # x=[]
    # list_loss=[]
    # list_acc=[]

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj, M)

        # output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        # list_beta.append(model.attentions[2].beta.item())
        # x.append(epoch)
        # list_acc.append(acc_train.item())
        # list_loss.append(loss_train.item())

        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj, M)
            # output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        # print('Epoch: {:04d}'.format(epoch+1),
        #       'loss_train: {:.4f}'.format(loss_train.data.item()),
        #       'acc_train: {:.4f}'.format(acc_train.data.item()),
        #       'loss_val: {:.4f}'.format(loss_val.data.item()),
        #       'acc_val: {:.4f}'.format(acc_val.data.item()),
        #       'time: {:.4f}s'.format(time.time() - t))

        return loss_val.data.item()

    def compute_test():
        model.eval()
        output = model(features, adj, M)
        # output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "model=",
              conf.model,
              "dataset=",
              dataset,
              "beta={:.4f}".format(conf.beta),
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()))

    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_values.append(train(epoch))

        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    #
    # # Restore best model
    # print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    compute_test()

    # plt.figure(figsize=(5, 5))
    # plt.plot(x, list_beta)
    # plt.plot(x, list_beta,'r',x,list_loss,'g',x,list_acc,'b')
    # plt.legend(labels=['beta','loss','acc'])
    # plt.show()


for dataset in ['pubmed', 'citeseer']:
    for model in ['gat', 'gatv2']:
        for beta in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            autoTrain(dataset=dataset, beta=beta, modeltype=model)

import os
import numpy as np
import torch
from utils.ot_functions import sim2real, reverse_val
from utils.model import Model
from utils.eval_functions import test_acc
import pandas as pd

def get_hyperparams(d):
    out = dict()
    for k in d.keys():
        if k not in d['keys_to_store']:
            out[k] = d[k]
    return out

def get_existing(path):
    """
    This function returns a list of existing experiments in a folder given by path.
    Parameters:
    - path, path to the folder containing the experiments
    Returns:
    - list of Experiment objects that exist in that folder, with run and model parameters
    """
    models = []
    for model in sorted([int(x) for x in os.listdir(path)]):
        p = path + '/' + str(model) + '/nn_features/'
        if os.path.isdir(p):
            x = os.listdir(p)
            assert len(x) == 1
            p += x[0]
            m = Model(None)
            m = m.load_model(p + '/model.pkl')
            m['idx'] = model
            models.append(m)
        else:
            print("Skipping: {}, not a directory!".format(p))
    return models

"""
m = Model(None)
m = m.load_model('./models/nn_features/svhn_smallmnist_svhn_mnist_smallmnist_svhn_a_0.0001_b_1e-05_z1_128_z2_128_lr_0.0005_hd1_256_hd2_512_bs1_128_bs2_128_seed1_0_seed2_0/model.pkl')
print(m)
exit()
"""

load_data = False

if not load_data:
    path = '/scratch/elec/t40106-everyone/laot/results/laot_sweep/sweep_6/results/'
    #path = '/scratch/elec/t40106-everyone/laot/results/vae_laot_sweep/sweep_6/results/'

    models = get_existing(path)
    print("Found {} models".format(len(models)))

    accuracies = []
    params = []

    seeds = []
    methods = []
    alphas = []
    betas = []
    bsize = []
    zdims = []
    lrs = []
    ids = []
    best_epochs = []
    hiddens1 = []
    hiddens2 = []

    for m in models:
        if len(m['acc_knn']) > 0:
            accuracies.append(max(m['acc_knn']))
            best_epochs.append(np.argmax(np.array(m['acc_knn'])))
            params.append(m)
            try:
                methods.append(m['model_1_method'])
                betas.append(m['model_1_beta'])
            except Exception:
                methods.append('unknown')
                betas.append(0.0)
                print("warning")
            hiddens1.append(m['hidden1'])
            hiddens2.append(m['hidden2'])
            seeds.append(m['torch_seed'])
            alphas.append(m['alpha'])
            lrs.append(m['learning_rate'])
            bsize.append(m['batch_size1'])
            zdims.append(m['latent_dim'])
            ids.append(m['idx'])
        else:
            ids.append(m['idx'])
            print("Warning, this ID has no accuracy data: ", m['idx'])
    accuracies = np.array(accuracies)
    idx = np.argmax(accuracies)
    print("idx: {}, best_epoch: {}, acc: {}, seed: {}, lr: {}, alpha: {}, beta: {}, bsize: {}, zdim: {}, h1: {}, h2: {}".format(\
            idx, best_epochs[idx], accuracies.max(), seeds[idx], lrs[idx], alphas[idx], betas[idx], \
            bsize[idx], zdims[idx], hiddens1[idx], hiddens2[idx]))

    data = []
    for i in ids:
        #print(i, len(best_epochs), len(accuracies), len(seeds), len(lrs), len(alphas), len(betas), len(bsize), len(zdims), len(hiddens1), len(hiddens2))
        data.append([i, best_epochs[i-1], accuracies[i-1], seeds[i-1], lrs[i-1], alphas[i-1], betas[i-1], bsize[i-1], zdims[i-1], \
                hiddens1[i-1], hiddens2[i-1]])
    d = pd.DataFrame(data, columns = ['ID', 'best_epoch', 'acc', 'seed', 'lr', 'alpha', 'beta', 'bsize', 'zdim', 'hidden1', 'hidden2'])
    d.to_pickle('d.pkl')
else:
    d = pd.read_pickle('d.pkl')


import seaborn as sns
import matplotlib.pyplot as plt


sns.lineplot(d, x='best_epoch', y='acc')
plt.savefig(f'best_epoch.pdf')
plt.clf()

for a in ['alpha', 'bsize', 'lr', 'hidden1', 'hidden2']:
    sns.scatterplot(d, x=a, y='acc')
    plt.savefig(f'scat_{a}.pdf')
    plt.clf()


for a in ['alpha', 'bsize', 'lr', 'hidden1', 'hidden2']:
    plot = sns.lineplot(x=a, y='acc', data=d)
    plt.xscale('log')
    plt.savefig(f'eval_{a}.pdf')
    plt.clf()

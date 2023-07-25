import matplotlib.pyplot as plt
import torch

# OT imports for affinity loss
import ot
from ot.utils import dist
from utils.ot_functions import sim2real, compute_cost_matrix, Affine_map, RMTWassDist

mse_loss = torch.nn.MSELoss()

def standard_scaler(z):
    m = z.mean(0, keepdim=True)
    s = z.std(0, unbiased=False, keepdim=True)
    return (z - m) / s

def plot_batch(training_params, o_in, o_pred):
        n = min(o_in.size(0), 8)
        comp = torch.cat([o_in.view(o_in.shape[0], 3, \
                training_params['data_dim'][1], training_params['data_dim'][0])[:n], \
                o_pred.view(o_in.shape[0], 3, \
                training_params['data_dim'][1], training_params['data_dim'][0])[:n]])
        return comp

def mse(o, o_pred):
    return mse_loss(o, o_pred)

def rec_loss(o, o_pred):
    """
    rec = F.binary_cross_entropy(\
            o_pred.flatten(), o.flatten(), reduction='mean')
    return rec
    """
    return mse_loss(o, o_pred.view(o.shape))

def kld_loss(beta, logvar, mu):
    kld = (-beta * (1 + logvar - mu.pow(2) - \
            logvar.exp())).sum(1).mean(0, True)
    return kld

def affinity_loss(epoch, z_1, z_2, l1=None, l2=None, idxToUse=None, origC=None, map=None, alpha=0, scale=False, train=False, device='cpu'):

    if torch.isnan(torch.sum(z_1)) or torch.isnan(torch.sum(z_2)):
        print("Error, z_1 or z_2 contain nans. sim2real in network/train_functions_global.py@41 will segfault!")
        exit()

    if scale:
        z_1_norm = standard_scaler(z_1)
        z_2_norm = standard_scaler(z_2)
    else:
        z_1_norm = z_1
        z_2_norm = z_2

    if map is None:
        T_z_1, _, A, b = sim2real(z_1_norm, z_2_norm, return_map=True, do_procrustes=False, device=device)
    else:
        A = map[0]
        b = map[1]
        mean = z_1_norm.mean(0).reshape(-1, 1)
        T_z_1 = Affine_map(z_1_norm.T, A, b, mean).T

    if torch.isnan(torch.sum(T_z_1)):
        print("Error, T_z_1 contains nan (but z_1 and z_2 didnt). Try larger batch sizes or lower latent dims!" + \
                " You could also increase the noise in utils/ot_functions.py@46")

    C = dist(T_z_1, z_2_norm, metric='sqeuclidean')

    if train:
        if l1.ndim>1:
            l1flat = torch.argmax(l1, -1)
            l2flat = torch.argmax(l2, -1)
        else:
            l1flat = l1
            l2flat = l2

        Ty_ss = -1 * torch.ones_like(l2flat)

        for c in torch.unique(l2flat):
            idxClass = torch.argwhere(l2flat == c)
            idx = [idx for idx in idxToUse if idx in idxClass]
            Ty_ss[idx] = c

        M_lin = torch.from_numpy(compute_cost_matrix(l1flat, Ty_ss, v=1e6)).to(device).type(torch.float32)

    else:
        M_lin = 0

    mu_1 = torch.ones(z_1.shape[0], device=device)/z_1.shape[0]
    mu_2 = torch.ones(z_2.shape[0], device=device)/z_2.shape[0]

    if origC is None:
        origC = 0
    else:
        origC = origC#/origC.mean()

    # Leaving this untouched yields best results
    C = C#/C.mean()
    origC = origC
    #print(C.detach().cpu().numpy().mean(), C.detach().cpu().numpy().max(), C.detach().cpu().numpy().min()
    #print(origC.detach().cpu().numpy().mean(), origC.detach().cpu().numpy().max(), origC.detach().cpu().numpy().min()

    ot_cost = ot.emd2(mu_1, mu_2, C+origC, numItermax=100000)

    return ot_cost, A, b

def bures_loss(z_1, z_2, l1=None, l2=None, idxToUse=None, origC=None, map=None, alpha=0, scale=False, train=False, device='cpu'):

    if torch.isnan(torch.sum(z_1)) or torch.isnan(torch.sum(z_2)):
        print("Error, z_1 or z_2 contain nans. sim2real in network/train_functions_global.py@41 will segfault!")
        exit()

    if scale:
        z_1_norm = standard_scaler(z_1)
        z_2_norm = standard_scaler(z_2)
    else:
        z_1_norm = z_1
        z_2_norm = z_2

    if map is None:
        T_z_1, _, A, b = sim2real(z_1_norm, z_2_norm, return_map=True, do_procrustes=False, device=device)
    else:
        A = map[0]
        b = map[1]
        mean = z_1_norm.mean(0).reshape(-1, 1)
        T_z_1 = Affine_map(z_1_norm.T, A, b, mean).T

    bures_cost = RMTWassDist(T_z_1, z_2_norm)

    return bures_cost, 0, 0


def temporal_loss(z_1, z_2, scale=False, device='cpu'):

    if torch.isnan(torch.sum(z_1)) or torch.isnan(torch.sum(z_2)):
        print("Error, z_1 or z_2 contain nans. sim2real in network/train_functions_global.py@41 will segfault!")
        exit()

    if scale:
        z_1_norm = standard_scaler(z_1)
        z_2_norm = standard_scaler(z_2)
    else:
        z_1_norm = z_1
        z_2_norm = z_2

    temp_cost = 0

    for z1, z2 in zip(z_1[1:,:], z_2[:-1,:]):
        temp_cost += torch.linalg.norm(z1-z2)

    return temp_cost

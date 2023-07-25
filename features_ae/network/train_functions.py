from network.train_functions_global import rec_loss, affinity_loss, kld_loss
import torch

#1. Preprocess data from the dataset def get_data(data, training_params):
def get_data(data):
    # Return features and labels
    if len(data)>2:
        return data[0],data[1], data[2]
    else:
        return data[0], None, data[1]
#2. Compute z and other stuff by passing the observation through the encoder/decoder
def get_z(data, ae, ae_id):
    o = data
    tmp = ae(o, ae_id)
    # When using AE
    if len(list(tmp)) == 2:
        o_pred = tmp[0]
        z = tmp[1]
        return o, o_pred, z
    # When using VAE
    if len(list(tmp)) == 4:
        o_pred = tmp[0]
        z = tmp[1]
        mu = tmp[2]
        logvar = tmp[3]
        return o, o_pred, z, mu, logvar

#2. Return the label for the downstream task
def get_label(data):
    return data['label']

def get_l_probs(z, classifier):
    l_probs = classifier(z)
    return l_probs

#4. Loss function
def loss_function_ae(epoch, model, z1, z2, l1, l2, idxToUse, C=0, map=None, alpha=0, scale=False, use_target=False, device='cpu'):
    if model.model_params_1['method'] == 'ae':
        o_1, o_pred_1, z_1 = z1
        o_2, o_pred_2, z_2 = z2
    else:
        o_1, o_pred_1, z_1, mu_1, logvar_1 = z1
        o_2, o_pred_2, z_2, mu_2, logvar_2 = z2

    ret = []

    # Reconstruction loss
    rec_1 = rec_loss(o_1, o_pred_1)
    rec_2 = rec_loss(o_2, o_pred_2)
    loss = 0.5 * (rec_1 + rec_2)
    ret.append(rec_1.detach().cpu().numpy())
    ret.append(rec_2.detach().cpu().numpy())

    # Affinity loss
    if model.training_params['alpha'] >= 0:
        affinity = affinity_loss(epoch, z_1, z_2, l1, l2, idxToUse, origC=C, map=map, 
                alpha=alpha, scale=scale, train=use_target, device=device)
        loss += model.training_params['alpha'] * affinity[0]
        ret.append(affinity[0].detach().cpu().numpy())
    else:
        ret.append(0.0)

    # KLD loss (VAE only)
    if model.model_params_1['method'] == 'vae':
        kld_1 = kld_loss(model.model_params_1['beta'], logvar_1, mu_1)[0]
        kld_2 = kld_loss(model.model_params_2['beta'], logvar_2, mu_2)[0]
        kld = 0.5 * (kld_1 + kld_2)
        loss += kld
        ret.append(kld_1.detach().cpu().numpy())
        ret.append(kld_2.detach().cpu().numpy())
    else:
        ret.append(0.0)
        ret.append(0.0)

    return loss, ret, affinity[1], affinity[2]

#5. Housekeeping
def setup_housekeeping():
    hk_loss_rec = []
    hk_loss_slow = []
    return [hk_loss_rec, hk_loss_slow]

def housekeeping(hk_data, hk_lists):
    rec_loss, kld_loss, slow_loss = hk_data
    hk_lists[0].append(rec_loss)
    hk_lists[2].append(slow_loss)
    return hk_lists
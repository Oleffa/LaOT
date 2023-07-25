import torch
from utils.ot_functions import sim2real
from network.train_functions_global import standard_scaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def test_acc(z_main_store, z_secondary_store, l_main_store, l_secondary_store, \
        experiment, scale_latents, invariant, device, verbose, benchmark=False):
    if benchmark:
        proc_time, perf_time, sourceAligned, _, A, b = sim2real(
            torch.from_numpy(z_main_store).to(device),
            torch.from_numpy(z_secondary_store).to(device),
            return_map=True, do_procrustes=False, device=device, benchmark=True)
        return proc_time, perf_time
    else:
        sourceAligned, _, A, b = sim2real(
            torch.from_numpy(z_main_store).to(device),
            torch.from_numpy(z_secondary_store).to(device),
            return_map=True, do_procrustes=False, device=device)

    if scale_latents:
        z1T = standard_scaler(torch.from_numpy(z_main_store).to(device))
        z2T = standard_scaler(torch.from_numpy(z_secondary_store).to(device))
    else:
        z1T = torch.from_numpy(z_main_store).to(device)
        z2T = torch.from_numpy(z_secondary_store).to(device)

    if invariant:
        sourceAligned = z1T
    else:
        sourceAligned, _, A, b = sim2real(
            z1T, z2T, return_map=True,
            do_procrustes=False, device=device)

    if experiment == 'nn_features':
        clf_1 = KNeighborsClassifier(n_neighbors=1)

        param_grid = {'kernel': ['linear'], 'C': [.001, .01, .1]}

        clf_SVC = GridSearchCV(
            SVC(), param_grid, scoring='accuracy', n_jobs=-1)

        clf_1.fit(sourceAligned.detach().cpu().numpy(), l_main_store)
        acc = clf_1.score(
            z2T.cpu(), l_secondary_store)

        if verbose:
            print("Transfer score (1NN): {:.2f}".format(acc * 100))

        return acc*100

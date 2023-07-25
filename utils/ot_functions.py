import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
plt.style.use('seaborn')
from sklearn.model_selection import train_test_split, StratifiedKFold
import scipy
import time
import ot

def duplicates(L):
    seen = set()
    seen2 = set()
    seen_add = seen.add
    seen2_add = seen2.add
    for item in L:
        if item in seen:
            seen2_add(item)
        else:
            seen_add(item)
    return list(seen2)


def shuffle(a, b, c, d):
    assert a.shape[0] == b.shape[0]
    p1 = np.random.permutation(a.shape[0])
    p2 = np.random.permutation(c.shape[0])
    return a[p1], b[p1], c[p2], d[p2]

def procrustes(X,Y):
    M = Y@(X.T)
    U, Sigma, V = torch.linalg.svd(M)
    return U@V

def sqrtm(A, device):
    A_ = A.to(torch.float64)
    e_vals, e_vecs = torch.linalg.eigh(A_)
    e_vecs = torch.real(e_vecs)
    e_vals = torch.real(e_vals)

    ### negative eigs
    idx = torch.argwhere(e_vals>0)
    if len(idx)!=len(e_vals):
        #print('Caution! Numerical errors in sqrtm with {}/{} negative eigs.'.format(len(e_vals)-len(idx), len(e_vals)))
        consts = torch.linspace(1e-7, 1e-6, len(e_vals)-len(idx), device=device)
        eval_pos = e_vals[idx].ravel()
        e_vals_new = torch.cat((consts, eval_pos)).ravel()
        if len(torch.unique(e_vals_new)) != len(e_vals):
            e_vals_new_unique = [e_vals_new[0]]
            for i in range(1, len(e_vals_new)):
                if e_vals_new[i] == e_vals_new[i-1]:
                    print(e_vals_new[i])
                    e_vals_new_unique.append((e_vals_new[i+1] + e_vals_new[i-1])/2)
                else:
                    e_vals_new_unique.append(e_vals_new[i])
            e_vals_new = torch.FloatTensor(e_vals_new_unique)
            assert len(torch.unique(e_vals_new)) == len(e_vals)
    else:
        e_vals_new = e_vals
    Sigma = torch.sqrt(e_vals_new)

    ### fix for testing with random
    """
    e_vals = torch.real(e_vals)
    e_vecs = torch.real(e_vecs)
    e_vals_new = torch.linspace(1e-6, torch.max(e_vals).item(), len(e_vals))
    e_vec_new = torch.rand_like(e_vecs, dtype=torch.float32)
    Sigma = torch.sqrt(e_vals_new)
    """
    
    return (e_vecs@torch.diag(Sigma)@e_vecs.T).to(torch.float32)

#For two Gaussians N(m0,K0), N(m1,K1), compute the OT map F(x) = Ax + b, returning (A,b)
def OT_map(m0, K0, m1, K1, device):
    A = sqrtm(K1, device=device)
    tmp = A@K0@A #+ 1e-6*torch.eye(A.shape[0])
    sqtmp = sqrtm(tmp, device=device) #+1e-6*torch.eye(A.shape[0])
    if torch.isnan(sqtmp).any():
        print('Nans in sqtmp')
    return (A@torch.inverse(sqtmp)@A), m1


#Apply an affine map by centering the input first
def Affine_map(X, A, b, mean):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    Y = X - mean
    return A@Y + b


#Transfer M_sim [d x N_sim] dataset to M_real [d x N_real].
#Returns f_sim2real(M_sim), which is the transferrred dataset.
#Optionally also returns the AT map as a pair (A,b).
def sim2real(M_sim, M_real, map=None, return_map=False, do_procrustes=False, device='cpu', benchmark=False):
    M_sim = M_sim.T
    M_real = M_real.T
    #Compute mean and covariance matrix for the simulated data
    N_sim = M_sim.shape[1]
    m_sim = M_sim.mean(1).reshape(-1,1)

    K_sim = torch.cov(M_sim) + 1e-6*torch.eye(M_sim.shape[0], device=device) #the last term is no avoid singularities

    #Compute mean and covariance matrix for the real data
    N_real = M_real.shape[1]
    m_real = M_real.mean(1).reshape(-1,1)
    K_real = torch.cov(M_real) + 1e-6*torch.eye(M_sim.shape[0], device=device)

    if do_procrustes:
        R = procrustes(M_sim, M_real)
        m_sim = (R@m_sim)
        K_sim = (R@K_sim@R.T)

    #Compute the OT map from M_sim to M_real
    if map is None:
        proc_time = time.process_time()
        perf_time = time.perf_counter()
        A, b = OT_map(m_sim, K_sim, m_real, K_real, device=device)
        proc_time_end = time.process_time()
        perf_time_end = time.perf_counter()
        mean = M_sim.mean(1).reshape(-1, 1)
    else:
        A = map[0]
        b = map[1]

    if do_procrustes:
        f_sim2real = lambda X: Affine_map(R@X, A, b, R@mean)
        #f_sim2real = lambda X: R@X
    else:
        f_sim2real = lambda X: Affine_map(X, A, b, mean)


    if benchmark:
        if return_map:
            return proc_time_end - proc_time, perf_time_end - perf_time, f_sim2real(M_sim).T, f_sim2real, A, b
        else:
            return proc_time_end - proc_time, perf_time_end - perf_time, f_sim2real(M_sim).T
    else:
        if return_map:
            return f_sim2real(M_sim).T, f_sim2real, A, b
        else:
            return f_sim2real(M_sim).T


def rho_aff(M_sim, M_real):
    M_sim2real = sim2real(M_sim, M_real)
    N_sim = M_sim.shape[1]
    N_real = M_real.shape[1]
    C = cdist(M_sim2real.T, M_real.T)**2
    mu_sim = np.ones(N_sim)/N_sim
    mu_real = np.ones(N_sim)/N_sim
    K_real = np.cov(M_real)
    print(mu_sim.shape, mu_real.shape, C.shape)
    return 1 - np.sqrt(ot.emd2(mu_sim, mu_real, C)/2*(np.trace(K_real)))


def get_M(df, domain_name):
    df = df[df["domain_id"] == domain_name]
    states = np.stack(df["state"])
    actions = np.stack(df["action"])
    next_states = np.stack(df["next_state"])
    M = np.concatenate((states, actions, next_states), axis=-1)
    return M.T

def comp(x, y, v=1e6):
    if x == y or y == -1:
        return 0
    else:
        return v

def compute_cost_matrix(ys,yt,v=np.inf):

    M = np.zeros((len(ys), len(yt)))
    for i,y1 in enumerate(ys):
        for j,y2 in enumerate(yt):
            M[i,j]=comp(y1,y2,v)
    return M


def reverse_val(Sx, Sy, Tx, Ty, idx, clf):

    acc = []

    if idx is None:
        idx = []

    args = [Sx, Sy, Tx, Ty, idx]

    for idx, i in enumerate(args):
        if torch.is_tensor(i):
            args[idx] = i.detach().cpu().numpy()

    Sx, Sy, Tx, Ty, idx = tuple(args)

    del args

    uTx = np.asarray([x for i, x in enumerate(Tx) if i not in idx])

    for n in range(5):

        trainSx, testSx, trainSy, testSy = train_test_split(Sx, Sy, test_size=0.2, shuffle=True)

        clf.fit(trainSx, trainSy)
        TyR = clf.predict(uTx)

        clf.fit(np.vstack((uTx, Tx[idx])), np.hstack((TyR, Ty[idx])))
        acc.append(100 * clf.score(testSx, testSy))

    print('Reversed:', np.mean(acc))
    return np.mean(acc)

def RMTWassDist(X1,X2):
    #Function that compute the Wasserstein distance between Gaussian centered
    #distribution based on the article Random  Matrix-Improved  Estimation  of  the  Wasserstein  Distance
    #between  two  Centered  Gaussian  Distribution (Malik TIOMOKO & Romain COUILLET)
    #Input samples from the first class X1 of dimension p*n1 and the
    #samples from the second class X2 of size p*n2

    X1 = X1.T
    X2 = X2.T

    p, n1 = X1.shape
    _, n2 = X2.shape

    c1 = p / n1
    c2 = p / n2

    #Sample covariance estimate
    hatC1 = X1@X1.T/n1
    hatC2 = X2@X2.T/n2
    lambda_= torch.sort(torch.real(torch.linalg.eig(hatC1@hatC2)[0]))[0]

    m = lambda z : torch.mean(1/(lambda_-z))
    phi = lambda z : z/(1-c1-c1*z*m(z))
    psi = lambda z : 1-c2-c2*z*m(z)
    f = lambda z : torch.sqrt(z)

    eta = torch.sort(torch.linalg.eigh(torch.diag(lambda_)-(1/n1)*torch.sqrt(lambda_).reshape(p,1)@torch.sqrt(lambda_).reshape(1,p))[0])[0]
    zeta = torch.sort(torch.linalg.eigh(torch.diag(lambda_)-(1/n2)*torch.sqrt(lambda_).reshape(p,1)@torch.sqrt(lambda_).reshape(1,p))[0])[0]

    # Distinguish the case n1=n2 versus n1!=n2
    if n1 == n2:
        RMTDistEst = (1 / p) * torch.trace(hatC1 + hatC2) - 2 * (torch.sum(torch.sqrt(lambda_)) - torch.sum(torch.sqrt(zeta))) * (
                    2 * n1 / p)
    else:
        # Distinguish the case where n1n2
        if eta[0] < zeta[0]:
            my_eta = zeta
            my_zeta = eta
        else:
            my_zeta = zeta
            my_eta = eta

        other = lambda z: 2 * np.sum(1 / (z - zeta)) - 2 * np.sum(1 / (z - lambda_))

        integrand_real = lambda z: 1 / (2 * pi) * 2 * f(-(phi(z) / psi(z))) * other(z) * (psi(z) / c2)

        # Computing the second term (real_integral)
        real_integral = 0
        for i in range(len(my_zeta)):
            # xs=np.linspace(my_zeta[i],my_eta[i],1000)
            # xs=xs[1:-1]

            real_integral += scipy.integrate.quad(integrand_real, my_zeta[i], my_eta[i], full_output=1)[
                0]  # np.trapz([integrand_real(z) for z in xs],x=xs)

        # Computing the first term (pole in lambda)
        pole = 2 * (np.sqrt(c2 / c1)) * np.sum(np.sqrt(lambda_)) / c2
        esty = pole + real_integral

        RMTDistEst = (1 / p) * np.trace(hatC1 + hatC2) - 2 * esty

    return RMTDistEst

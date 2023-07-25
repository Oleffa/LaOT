import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from utils.ot_functions import sim2real, reverse_val
from utils.eval_functions import test_acc
import ot
from ot.utils import dist
import numpy as np
from dataloader.dataloader import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import scipy as sc
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from network.train_functions_global import standard_scaler
from sklearn.model_selection import train_test_split
import time

class CoupledAE(torch.nn.Module):
    def __init__(
            self,
            model,
            aeconfig,
            encoder_1,
            encoder_2,
            decoder_1,
            decoder_2,
            classifier):
        super(CoupledAE, self).__init__()
        self.vc = aeconfig  # This is later used to access the experiment specific loss
        # and housekeeping functions specified in the network/train_functions.py
        # of each experiment
        self.model = model
        self.encoder_1 = encoder_1
        self.decoder_1 = decoder_1
        self.encoder_2 = encoder_2
        self.decoder_2 = decoder_2
        self.classifier = classifier
        
        if self.model.model_params_1['method'] == 'vae' or self.model.model_params_2['method'] == 'vae':
            self.variational = True
        else:
            self.variational = False

        self.loss_function_ae = self.vc.loss_function_ae
        self.classification_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.model.training_params['learning_rate'])
        #self.scheduler = optim.lr_scheduler.ExponentialLR(
        #    self.optimizer, gamma=.95)

        self.get_data = self.vc.get_data
        self.get_z = self.vc.get_z
        self.get_l_probs = self.vc.get_l_probs

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, o_in, ae_idx):
        tmp = self.encode(o_in, ae_idx)
        # AE
        if type(tmp) == torch.Tensor:
            o_pred = self.decode(tmp, ae_idx)
            return o_pred, tmp
        elif type(tmp) == tuple:
            z = tmp[0]
            mu = tmp[1]
            logvar = tmp[2]
            o_pred = self.decode(z, ae_idx)
            return o_pred, z, mu, logvar

    def encode(self, o_in, ae_idx, benchmark=False):
        proc_time = time.process_time()
        perf_time = time.perf_counter()
        if self.variational:
            if ae_idx == 0:
                mu, logvar = self.encoder_1(o_in)
            else:
                mu, logvar = self.encoder_2(o_in)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            # OMG is this a bug?
            if ae_idx == 0:
                z = self.encoder_1(o_in)
            else:
                z = self.encoder_2(o_in)
        proc_time_end = time.process_time()
        perf_time_end = time.perf_counter()

        if benchmark:
            if type(z) == torch.Tensor:
                return z, proc_time_end - proc_time, perf_time_end - perf_time
            elif type(z) == tuple:
                mu = z[0]
                logvar = z[1]
                z = self.reparameterize(mu, logvar)
                return z, mu, logvar, proc_time_end - proc_time, perf_time_end - perf_time

        # Return zs directly if its an AE or reparamterize when its a VAE
        if type(z) == torch.Tensor:
            return z
        elif type(z) == tuple:
            mu = z[0]
            logvar = z[1]
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar

        print("Warning got weird zs from encoder in coupledAE.py")

    def decode(self, z, ae_idx):
        if ae_idx == 0:
            o_pred = self.decoder_1(z)
        else:
            o_pred = self.decoder_2(z)
        return o_pred

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def save_model(self, epoch):
        # Set model file
        self.model.model_file = self.state_dict()
        self.model.epoch = epoch
        # Update pkl file
        self.model.save_model()

    def load_model(self):
        self.load_state_dict(self.model.model_file)

    def benchmark(self):
        self.model.dataset_params_train_1['drop_last'] = False
        self.model.dataset_params_train_2['drop_last'] = False
        test_loader_1, test_dataset_1 = DataLoader(
            self.model.training_params, self.model.dataset_params_train_1, shuffle=False).get_loader()
        test_loader_2, test_dataset_2 = DataLoader(
            self.model.training_params, self.model.dataset_params_train_2, shuffle=False).get_loader()
        self.model.dataset_params_train_1['drop_last'] = True
        self.model.dataset_params_train_2['drop_last'] = True

        epochs = 1
        e_proc_times = []
        e_perf_times = []
        ot_perf_times = []
        ot_proc_times = []
        ot_perf_times_raw = []
        ot_proc_times_raw = []


        for e in range(epochs):
            # Go through both datasets and record zs
            z_1= np.zeros(
                (len(test_dataset_1),
                 self.model.training_params['latent_dim'])).astype(
                np.float32)
            z_2= np.zeros(
                (len(test_dataset_2), self.model.training_params['latent_dim'])).astype(
                np.float32)

            d1 = np.zeros((len(test_dataset_1), self.model.dataset_params_train_1['feature_dim'])).astype(np.float32)
            d2 = np.zeros((len(test_dataset_2), self.model.dataset_params_train_2['feature_dim'])).astype(np.float32)


            e_proc_time = 0.0
            e_perf_time = 0.0
            e_proc_time_2 = 0.0
            e_perf_time_2 = 0.0

            dim = -1

            with torch.no_grad():
                for i, main_data in enumerate(test_loader_1):
                    d = self.get_data(main_data)

                    proc_time_start = time.process_time()
                    perf_time_start  = time.perf_counter()
                    z = self.get_z(d[0], self, 0)
                    e_proc_time += time.process_time() - proc_time_start
                    e_perf_time += time.perf_counter() - perf_time_start

                    bs = self.model.dataset_params_train_1['batch_size']

                    # Store latent representations
                    z_1[i * bs:(i * bs + z[0].shape[0])] = z[2].detach().cpu().numpy()
                    d1[i * bs:(i * bs + d[0].shape[0])] = d[0].detach().cpu().numpy()
                    if i * bs >= dim and dim > 0:
                        break

                for i, secondary_data in enumerate(test_loader_2):
                    d = self.get_data(secondary_data)

                    proc_time_start_2 = time.process_time()
                    perf_time_start_2  = time.perf_counter()
                    z = self.get_z(d[0], self, 1)
                    e_proc_time_2 += time.process_time() - proc_time_start_2
                    e_perf_time_2 += time.perf_counter() - perf_time_start_2

                    bs = self.model.dataset_params_train_2['batch_size']

                    # Store latent representations
                    z_2[i * bs:(i * bs + z[0].shape[0])] = z[2].detach().cpu().numpy()
                    d2[i * bs:(i * bs + d[0].shape[0])] = d[0].detach().cpu().numpy()
                    if i * bs >= dim and dim > 0:
                        break

            if dim == -1:
                ot_proc_time, ot_perf_time = test_acc(z_1, z_2, None, None, 
                        self.model.training_params['experiment'], self.model.training_params['scale_latents'],
                        self.model.training_params['invariant'], self.model.training_params['device'], 
                        self.model.training_params['verbose'], benchmark=True)
            else:
                ot_proc_time, ot_perf_time = test_acc(z_1[:dim], z_2[:dim], None, None, 
                        self.model.training_params['experiment'], self.model.training_params['scale_latents'],
                        self.model.training_params['invariant'], self.model.training_params['device'], 
                        self.model.training_params['verbose'], benchmark=True)
            e_perf_times.append(max(e_perf_time, e_perf_time_2))
            e_proc_times.append(e_proc_time + e_proc_time_2)
            ot_perf_times.append(ot_perf_time)
            ot_proc_times.append(ot_proc_time)

            # ----------- raw data ----------
            proc_time_start = time.process_time()
            perf_time_start = time.perf_counter()

            if dim != -1:
                d1 = d1[:dim]
                d2 = d2[:dim]

            C1 = sc.spatial.distance.cdist(d1, d1)
            C2 = sc.spatial.distance.cdist(d2, d2)

            C1 /= C1.max()
            C2 /= C2.max()

            p = ot.unif(d1.shape[0])
            q = ot.unif(d2.shape[0])
            
            gw, log = ot.gromov.entropic_gromov_wasserstein(
                C1, C2, p, q, 'square_loss', epsilon=5e-4, log=True, verbose=True)
            """
            
            # Use emd2
            if dim == -1:
                mu_1 = torch.ones(d1.shape[0], device=self.model.training_params['device'])/d1.shape[0]
                mu_2 = torch.ones(d2.shape[0], device=self.model.training_params['device'])/d2.shape[0]
                proc_time_start = time.process_time()
                perf_time_start = time.perf_counter()
                C = dist(d1, d2, metric='sqeuclidean')
            else:
                mu_1 = torch.ones(dim, device=self.model.training_params['device'])/dim
                mu_2 = torch.ones(dim, device=self.model.training_params['device'])/dim
                proc_time_start = time.process_time()
                perf_time_start = time.perf_counter()
                C = dist(d1[:dim], d2[:dim], metric='sqeuclidean')
            ot_cost = ot.emd2(mu_1, mu_2, torch.from_numpy(C), numItermax=1000000)
            """
            proc_time_end = time.process_time()
            perf_time_end = time.perf_counter()
            ot_perf_times_raw.append(perf_time_end - perf_time_start)
            ot_proc_times_raw.append(proc_time_end - proc_time_start)
            print(f'{e+1}/{epochs}')

        e_perf_times = np.array(e_perf_times).mean()
        e_proc_times = np.array(e_proc_times).mean()
        ot_perf_times = np.array(ot_perf_times).mean()
        ot_proc_times = np.array(ot_proc_times).mean()
        ot_perf_times_raw = np.array(ot_perf_times_raw).mean()
        ot_proc_times_raw = np.array(ot_proc_times_raw).mean()

        print(f'Embeddings: embedding: {e_perf_times}, ot_map: {ot_perf_times}, combined: {e_perf_times + ot_perf_times}')
        print(f'Embeddings: ot_map: {ot_perf_times_raw}')

        #print(f'Embeddings: embedding: {e_proc_times}, ot_map: {ot_proc_times}, combined: {e_proc_times + ot_proc_times}')
        #print(f'Embeddings: ot_map: {ot_proc_times_raw}')


        """
        # Setup dataloaders
        print("Starting training")
        print("Setup dataloaders...")
        train_loader_1, train_dataset_1 = DataLoader(
            self.model.training_params, self.model.dataset_params_train_1, tag='source').get_loader()
        train_loader_2, train_dataset_2 = DataLoader(
            self.model.training_params, self.model.dataset_params_train_2, tag='target').get_loader()

        it1 = iter(train_loader_1)
        it2 = iter(train_loader_2)

        e_embedding = []
        e_ot_map = []
        r_ot_map = []

        e = 10
        for i in range(e):
            data = next(it1)
            d1, l, idxS = self.get_data(data)
            z_1, process_time, performance_time = self.encode(d1, 0, benchmark=True)
            e_embedding.append(process_time)

            data = next(it2)
            d2, l, idxS = self.get_data(data)
            z_2, process_time, performance_time = self.encode(d2, 1, benchmark=True)
            e_embedding[-1] = e_embedding[-1] + performance_time

            if self.model.training_params['experiment'] == 'nn_features':
                scale = self.model.training_params['scale_latents']
            else:
                scale = False

            #========= Embeddings ==========
            origC = 0
            if scale:
                z_1 = standard_scaler(z_1)
                z_2 = standard_scaler(z_2)

            process_time, performance_time, T_z_1, _, A, b = sim2real(z_1, z_2, return_map=True, do_procrustes=False, \
                    device=self.model.training_params['device'], benchmark=True)
            e_ot_map.append(process_time)


            #========= Raw data ==========
            mu_1 = torch.ones(d1.shape[0], device=self.model.training_params['device'])/d1.shape[0]
            mu_2 = torch.ones(d2.shape[0], device=self.model.training_params['device'])/d2.shape[0]

            process_time, performance_time, T_d1, _, A, b = sim2real(d1, d2, return_map=True, do_procrustes=False, \
                    device=self.model.training_params['device'], benchmark=True)
            r_ot_map.append(process_time)

            print(f'{i+1}/{e}')

        e_embedding = np.array(e_embedding).mean()
        e_ot_map = np.array(e_ot_map).mean()
        print(f'Embeddings: embedding: {e_embedding}, ot_map: {e_ot_map}, combined: {e_embedding + e_ot_map}')
        print(f'Raw: ot_map: {np.array(r_ot_map).mean()}')
        """

    def train(self):
        # Setup dataloaders
        print("Starting training")
        print("Setup dataloaders...")
        train_loader_1, train_dataset_1 = DataLoader(
            self.model.training_params, self.model.dataset_params_train_1, tag='source').get_loader()
        train_loader_2, train_dataset_2 = DataLoader(
            self.model.training_params, self.model.dataset_params_train_2, tag='target').get_loader()

        if abs(len(train_dataset_1) - len(train_dataset_2)) > 4 * len(train_dataset_1) or \
                abs(len(train_dataset_1) - len(train_dataset_2)) > 4 * len(train_dataset_2):
            print("network/doubleae.py@76: Warning, Datasets are imbalanced!")

        main_loader = train_loader_1
        main_dataset = train_dataset_1
        main_dataset_params = self.model.dataset_params_train_1
        main_ae = 0
        secondary_loader = train_loader_2
        secondary_dataset = train_dataset_2
        secondary_dataset_params = self.model.dataset_params_train_2
        secondary_ae = 1
        flipFlag = False

        if len(train_dataset_2) > len(train_dataset_1):
            main_loader = train_loader_2
            main_dataset = train_dataset_2
            main_dataset_params = self.model.dataset_params_train_2
            main_ae = 1
            secondary_loader = train_loader_1
            secondary_dataset = train_dataset_1
            secondary_dataset_params = self.model.dataset_params_train_1
            secondary_ae = 0
            flipFlag = True

        print("Dataloaders up and running!")

        clf_1 = KNeighborsClassifier(n_neighbors=1)
        clf_1.fit(main_dataset.features, main_dataset.labels)
        acc = clf_1.score(secondary_dataset.features, secondary_dataset.labels)
        print("No transfer baseline: ", acc)
        if self.model.epoch + \
                1 >= self.model.training_params['train_epochs'] + 1:
            print(
                "Model cannot continue, set a higher train epochs variable than load epoch")
            print("Training was not resumed, exiting ...")

        else:
            if flipFlag:
                self.test(
                    secondary_dataset_params,
                    main_dataset_params,
                    secondary_ae,
                    main_ae, 0)
            else:
                self.test(
                    main_dataset_params,
                    secondary_dataset_params,
                    main_ae,
                    secondary_ae, 0)
            if self.model.training_params['invariant']:
                map = (torch.eye(self.model.training_params['latent_dim']), 0)
            else:
                map = None

            for epoch in range(self.model.epoch + 1,
                               self.model.training_params['train_epochs'] + 1):

                epoch_loss = 0
                rec1 = 0
                rec2 = 0
                kld1 = 0
                kld2 = 0
                classif_loss = 0
                aff_loss = 0

                it = iter(secondary_loader)
                for i, main_data in enumerate(main_loader):
                    # Since the main loader contains usually more batches than the secondary loader,
                    # we have to start the secondary loader over again if it
                    # runs out of items to iterate over
                    try:
                        secondary_data = next(it)
                    except StopIteration:
                        it = iter(secondary_loader)
                        secondary_data = next(it)

                    # Obtaining data from dataloader
                    if flipFlag:
                        d1, l1, idxS = self.get_data(secondary_data)
                        d2, l2, idxT = self.get_data(main_data)
                    else:
                        d1, l1, idxS = self.get_data(main_data)
                        d2, l2, idxT = self.get_data(secondary_data)

                    if main_dataset.features.shape[1] == secondary_dataset.features.shape[1]:
                        origC = ot.dist(
                            train_dataset_1.features[idxS],
                            train_dataset_2.features[idxT])

                    elif d1.shape[0] == d2.shape[0]:
                        Cx = sc.spatial.distance.cdist(
                            train_dataset_1.features[idxS].detach().cpu().numpy(),
                            train_dataset_1.features[idxS].detach().cpu().numpy(),
                            metric='cosine')

                        Cy = sc.spatial.distance.cdist(
                            train_dataset_2.features[idxT].detach().cpu().numpy(),
                            train_dataset_2.features[idxT].detach().cpu().numpy(),
                            metric='cosine')

                        origC = torch.from_numpy(ot.dist(Cx, Cy)).to(
                            device=self.model.training_params['device'])

                    else:
                        origC = None

                    if train_dataset_2.idx_supervision is not None:
                        idxToUse = [idx for idx, x in enumerate(
                            idxT) if x in train_dataset_2.idx_supervision]
                    else:
                        idxToUse = []

                    # Computing latent and next state predictions
                    if flipFlag:
                        z1 = self.get_z(d1, self, secondary_ae)
                        z2 = self.get_z(d2, self, main_ae)
                    else:
                        z1 = self.get_z(d1, self, main_ae)
                        z2 = self.get_z(d2, self, secondary_ae)

                    if self.model.training_params['experiment'] == 'nn_features':
                        scale = self.model.training_params['scale_latents']
                    else:
                        scale = False

                    loss_rec, hk_data, A, b = self.loss_function_ae(epoch,
                        self.model, z1, z2, l1, l2, idxToUse, C=origC, map=map, scale=scale,
                        use_target=self.model.training_params['use_target'],
                        device=self.model.training_params['device'])

                    if self.model.training_params['supervision']:
                        sample = standard_scaler(z1[2])
                        labels = l1
                        l_probs = self.get_l_probs(sample, self.classifier)
                        loss_classification = self.classification_loss(
                            l_probs, labels)
                        loss = loss_rec + loss_classification
                        classif_loss += loss_classification.item()

                    else:
                        loss = loss_rec

                    # Update Gradients
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 2.)

                    self.optimizer.step()
                    epoch_loss += loss.item()
                    rec1 += hk_data[0]
                    rec2 += hk_data[1]
                    kld1 += hk_data[3]
                    kld2 += hk_data[4]

                    self.model.housekeeping.add(
                        'batch_train_loss_rec_1', hk_data[0])
                    self.model.housekeeping.add(
                        'batch_train_loss_rec_2', hk_data[1])
                    self.model.housekeeping.add(
                        'batch_train_loss_affinity', hk_data[2])
                    self.model.housekeeping.add(
                        'batch_train_loss_kld_1', hk_data[3])
                    self.model.housekeeping.add(
                        'batch_train_loss_kld_2', hk_data[4])

                    aff_loss += hk_data[2]

                epoch_loss = epoch_loss / len(main_loader)

                if self.model.training_params['verbose']:
                    print(
                            'Epoch {} done, rec 1: {}, rec 2: {}, kld 1: {}, kld 2: {}, aff. loss: {}'.format(
                            epoch, rec1/len(main_loader), rec2/len(main_loader), kld1/len(main_loader),
                            kld2/len(main_loader), aff_loss/len(main_loader)))

                # Save housekeeping
                self.model.housekeeping.add('epoch_train_loss', epoch_loss)
                
                if epoch % self.model.training_params['test_every'] == 0:
                    if flipFlag:
                        self.test(
                            secondary_dataset_params,
                            main_dataset_params,
                            secondary_ae,
                            main_ae, epoch)
                    else:
                        self.test(
                            main_dataset_params,
                            secondary_dataset_params,
                            main_ae,
                            secondary_ae, epoch)
                # This has to be last because it is also saving all the other
                # things set to pkl
                self.save_model(epoch)

    def test(
            self,
            main_dataset_params,
            secondary_dataset_params,
            main_ae,
            secondary_ae, epoch):

        # We need to set up the dataloaders here without the skipping of the
        # last uneven batch
        self.model.dataset_params_train_1['drop_last'] = False
        self.model.dataset_params_train_2['drop_last'] = False
        test_loader_1, test_dataset_1 = DataLoader(
            self.model.training_params, main_dataset_params, shuffle=False).get_loader()
        test_loader_2, test_dataset_2 = DataLoader(
            self.model.training_params, secondary_dataset_params, shuffle=False).get_loader()
        self.model.dataset_params_train_1['drop_last'] = True
        self.model.dataset_params_train_2['drop_last'] = True

        # Go through both datasets and record zs
        z_main_store = np.zeros(
            (len(test_dataset_1),
             self.model.training_params['latent_dim'])).astype(
            np.float32)
        z_secondary_store = np.zeros(
            (len(test_dataset_2), self.model.training_params['latent_dim'])).astype(
            np.float32)

        l_main_store = np.zeros((len(test_dataset_1),))
        l_secondary_store = np.zeros((len(test_dataset_2),))

        with torch.no_grad():
            for i, main_data in enumerate(test_loader_1):
                d = self.get_data(main_data)
                z = self.get_z(d[0], self, main_ae)

                # Store latent representations
                z_main_store[i * main_dataset_params['batch_size']:(i *
                                    main_dataset_params['batch_size'] +
                                    z[0].shape[0])] = z[2].detach().cpu().numpy()

                if d[1].detach().cpu().numpy().ndim > 1:
                    l_main_store[i * main_dataset_params['batch_size']:(i *
                                    main_dataset_params['batch_size'] +
                                    d[1].shape[0])] = np.argmax(d[1].detach().cpu().numpy(), -
                                                                                                1)
                else:
                    l_main_store[i * main_dataset_params['batch_size']:(i *
                                    main_dataset_params['batch_size'] +
                                    d[1].shape[0])] = d[1].detach().cpu().numpy()

            for i, secondary_data in enumerate(test_loader_2):
                d = self.get_data(secondary_data)
                z = self.get_z(d[0], self, secondary_ae)

                # Store latent representations
                z_secondary_store[i * secondary_dataset_params['batch_size']:(i *
                                    secondary_dataset_params['batch_size'] +
                                    z[0].shape[0])] = z[2].detach().cpu().numpy()

                if d[1].detach().cpu().numpy().ndim > 1:
                    l_secondary_store[i * secondary_dataset_params['batch_size']:(i *
                                    secondary_dataset_params['batch_size'] +
                                    d[1].shape[0])] = np.argmax(d[1].detach().cpu().numpy(), -1)
                else:
                    l_secondary_store[i * secondary_dataset_params['batch_size']:(i *
                                    secondary_dataset_params['batch_size'] +
                                    d[1].shape[0])] = d[1].detach().cpu().numpy()

            if self.model.training_params['store_latents']:
                self.model.housekeeping.add(
                    'z_{}'.format(main_ae + 1), z_main_store)
                self.model.housekeeping.add(
                    'z_{}'.format(
                        secondary_ae + 1),
                    z_secondary_store)
                self.model.housekeeping.add(
                    'l_{}'.format(main_ae + 1), l_main_store)
                self.model.housekeeping.add(
                    'l_{}'.format(
                        secondary_ae + 1),
                    l_secondary_store)
            
            acc = test_acc(z_main_store, z_secondary_store, l_main_store, l_secondary_store, 
                    self.model.training_params['experiment'], self.model.training_params['scale_latents'],
                    self.model.training_params['invariant'], self.model.training_params['device'], 
                    self.model.training_params['verbose'])
            self.model.housekeeping.add('acc_knn', acc)

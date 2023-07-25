import global_config as gc
import cv2
from network.coupledAE import CoupledAE
import os
import pickle
import numpy as np
import sys
sys.path.append('..')

class Model:
    def __init__(self, aeconfig):
        if None is not aeconfig:
            self.aeconfig = aeconfig
            self.training_params = aeconfig.training_params
            self.model_params_1 = aeconfig.model_params_1
            self.model_params_2 = aeconfig.model_params_2
            self.set_save_path()

            self.method_1 = self.model_params_1['method']
            self.method_2 = self.model_params_2['method']
            self.dataset_params_train_1 = aeconfig.dataset_params_train_1
            self.dataset_params_train_2 = aeconfig.dataset_params_train_2

            self.use_slowness = True

            # Make sure we use a known method from the list defined in the
            # global settings
            assert self.method_1 in gc.available_methods, "Warning unknown method {}. Must be {} as defined in global_config.py".format(
                self.method_1, gc.available_methods)
            assert self.method_2 in gc.available_methods, "Warning unknown method {}. Must be {} as defined in global_config.py".format(
                self.method_2, gc.available_methods)

            # Catch some problems when setting model params
            encoder_1, decoder_1, encoder_2, decoder_2, classifier = self.aeconfig.setup_network(
                self.model_params_1, self.model_params_2)
            self.ae = CoupledAE(
                self,
                self.aeconfig,
                encoder_1,
                encoder_2,
                decoder_1,
                decoder_2,
                classifier).to(
                self.training_params['device'])

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
                print("Created new save path: {}".format(self.save_path))
            else:
                print(
                    "Warning the output path {} already exists.".format(
                        self.save_path))
                if not self.training_params['overwrite']:
                    print("Exiting")
                    exit()

            # Initially there is no model file yet
            self.model_file = None
            self.epoch = 0
            # Initially there is no housekeeping data
            self.housekeeping = Housekeeping(
                self.training_params,
                self.model_params_1,
                self.model_params_2)
            self.test_reconstructions = []
            # This is a list of downstream experiments
            self.experiments = []
            # Overwrite existing model file if desired
            if os.path.isfile(self.save_path + 'model.pkl'):
                if self.training_params['overwrite']:
                    self.save_model()
                else:
                    print(
                        "Error, model already exists at {}. Consider setting training_params['overwrite']=True". format(
                            self.save_path + 'model.pkl'))
            else:
                self.save_model()
            print("Created new model at {}!".format(self.save_path))

    def set_save_path(self):
        self.save_path = '{}/{}/{}{}_{}{}_a_{}_b_{}_z1_{}_z2_{}_lr_{}_hd1_{}_hd2_{}_bs1_{}_bs2_{}_seed1_{}_seed2_{}/'.format(
            self.training_params['output_dir'],
            self.training_params['experiment'],
            self.training_params['domain1'],
            self.training_params['features1'],
            self.training_params['domain2'],
            self.training_params['features2'],
            self.training_params['alpha'],
            self.model_params_1['beta'],
            self.model_params_1['latent_dim'],
            self.model_params_2['latent_dim'],
            self.training_params['learning_rate'],
            self.training_params['hidden1'],
            self.training_params['hidden2'],
            self.training_params['batch_size1'],
            self.training_params['batch_size2'],
            self.training_params['torch_seed'],
            self.training_params['np_seed']
        )

    def train(self):
        self.ae.train()

    def benchmark(self):
        self.ae.benchmark()

    def vis_dataset(self):
        self.ae.vis_dataset()

    def load_model(self, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def save_model(self):
        # Todo also save the reconstructions here
        toSave = dict()
        if self.training_params['save_model']:
            # Save the training params
            for key in self.training_params:
                toSave[key] = self.training_params[key]
            # Save model params
            for key in self.model_params_1:
                toSave['model_1_' + key] = self.model_params_1[key]
            for key in self.model_params_2:
                toSave['model_2_' + key] = self.model_params_2[key]
            # Save the data form the housekeeping keys
            for key in self.housekeeping.data.keys():
                if key in self.training_params['keys_to_store']:
                    toSave[key] = self.housekeeping.data[key]
            with open(self.save_path + 'model.pkl', 'wb') as f:
                pickle.dump(toSave, f)
        else:
            print("Warning, training_params['save_model'] = False")
    # Visualisation and evaluation functions

    def verify(self):
        valid = True
        if self.epoch != self.model_params['train_epochs']:
            return False, "Verification failed: Current model epochs != model_params['train_epochs']"
        return valid, None

    def create_comp(self, o_1, o_pred_1, o_2, o_pred_2, pad=True, num=4):
        comp_1 = self.plot_batch(o_1, o_pred_1, pad=pad, num=num)
        comp_2 = self.plot_batch(o_2, o_pred_2, pad=pad, num=num)
        if len(self.dataset_params_train_1['resize_image']) == 3:
            comp_1 = np.concatenate(comp_1, axis=-3)
            comp_1 = np.concatenate(comp_1, axis=-2)
        else:
            comp_1 = np.concatenate(comp_1, axis=-3)
            comp_1 = np.concatenate(
                (comp_1[:, :, 0], comp_1[:, :, 1]), axis=-1)

        if len(self.dataset_params_train_2['resize_image']) == 3:
            comp_2 = np.concatenate(comp_2, axis=-3)
            comp_2 = np.concatenate(comp_2, axis=-2)
        else:
            comp_2 = np.concatenate(comp_2, axis=-3)
            comp_2 = np.concatenate(
                (comp_2[:, :, 0], comp_2[:, :, 1]), axis=-1)
        comp = np.concatenate((comp_1, comp_2), axis=-1)
        return comp

    def show_reconstruction(self, comp, epoch, border):
        comp = cv2.putText(comp,
                           "Epoch: {}/{}".format(epoch + 1,
                                                 len(self.test_reconstructions)),
                           (20,
                               27),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1,
                           (1,
                               0,
                               0),
                           2)
        cv2.imshow("Reconstructions", comp)
        cv2.waitKey(10000)

    def show_reconstruction_history(self, border='white'):
        if len(self.test_reconstructions) == 0:
            print("Can't show reconstructions for this model. Reconstruction list is empty. This experiment was probably run with save_reconstructions=False")
            return
        for epoch, comp in enumerate(self.test_reconstructions):
            self.show_reconstruction(comp, epoch, border)


class Housekeeping():
    def __init__(self, training_params, model_params_1, model_params_2):
        self.ae_name = 'a_{}_z1_{}_z2_{}/'.format(
            training_params['alpha'],
            model_params_1['latent_dim'],
            model_params_2['latent_dim'])

        keys = training_params['housekeeping_keys']
        self.data = {}
        for k in keys:
            self.data.update({k: []})

    def add(self, key, value):
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = []

    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def plot(
            self,
            key,
            ax,
            ylabel,
            xlabel,
            color,
            moving_average=None,
            legend=False):
        if moving_average is None:
            ax.plot(self.data[key], label=self.ae_name)
        else:
            ax.plot(
                self.running_mean(
                    self.data[key],
                    moving_average),
                label=self.ae_name,
                color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend()

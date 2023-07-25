from features_ae.network.train_functions import loss_function_ae, get_data, get_z,  \
    setup_housekeeping, housekeeping, get_l_probs
import global_config as gc
import os
import sys
from features_ae.network.encoder import EncoderFC, VariationalEncoderFC
from features_ae.network.decoder import DecoderFC
from features_ae.network.classifier import ClassifierFC
sys.path.append('..')

class AEConfig:
    def __init__(self):

        self.get_data = get_data
        self.get_z = get_z
        self.get_l_probs = get_l_probs
        self.housekeeping = housekeeping
        self.loss_function_ae = loss_function_ae
        self.dataset_params_train_1 = None
        self.dataset_params_train_2 = None

        self.model_params_1 = {
            'method': None,
            'beta': None,
            'latent_dim': None,
        }

        self.model_params_2 = {
            'method': None,
            'beta': None,
            'latent_dim': None,
        }

        self.training_params = {'config_version': 1,
                                'output_dir': os.path.abspath(gc.cfg['results_path']),
                                'experiment': 'nn_features',
                                'device': 'cpu',
                                'save_loss': True,
                                'save_model': True,
                                'verbose': True,
                                'reverse': False,
                                'invariant': False,
                                # Overwrite existing model.pkl files (throws
                                # warning)
                                'overwrite': True,
                                'housekeeping_keys': ['batch_train_loss_rec_1', 'batch_train_loss_rec_2',
                                                      'batch_train_loss_kld_1', 'batch_train_loss_kld_2',
                                                      'batch_train_loss_affinity', 'epoch_train_loss',
                                                      'z_1', 'z_2', 'l_1', 'l_2', 'acc_knn'
                                                      ],
                                'keys_to_store': ['batch_train_loss_rec_1', 'batch_train_loss_rec_2',
                                                      'batch_train_loss_kld_1', 'batch_train_loss_kld_2',
                                                      'batch_train_loss_affinity', 'epoch_train_loss',
                                                      'z_1', 'z_2', 'l_1', 'l_2', 'acc_knn'
                                                  ],
                                }

    def get_dataset(self, domain, features, shuffle, batch_size, drop_last):
        if domain in ['mnist', 'usps']:
            path = os.path.join(gc.cfg['dataset_path'], domain)
            size = 256
        if features == 'mnist_svhn':
            path = os.path.join(gc.cfg['dataset_path'], features, domain)
            size = 100
        elif features in ['sinusoid', 'second_order']:
            path = os.path.join(gc.cfg['dataset_path'], features, domain)
            size = 2
        elif features in ['ant_v2']:
            path = os.path.join(gc.cfg['dataset_path'], features, domain)
            if 'state' in domain:
                size = 27
            else:
                size = 8
        elif features in ['acrobat']:
            path = os.path.join(gc.cfg['dataset_path'], features, domain)
            if 'state' in domain:
                size = 4
            else:
                size = 1
        else:
            path = os.path.join(gc.cfg['dataset_path'], features, domain)
            size = features[-4:]

        return {'name': 'Domain: {}, Features: {}'.format(domain, features),
                'shuffle': shuffle,
                'feature_dim': int(size),
                'drop_last': drop_last,
                'data_path': f"{path}/", # we might need a trailing slash here
                'batch_size': batch_size,
                'verbose': False,
                }

    def setup_network(self, model_params_1, model_params_2):

        if model_params_1['method'] == 'ae':
            encoder = EncoderFC(
                [self.dataset_params_train_1['feature_dim'], self.training_params['hidden1'],
                    self.training_params['hidden1'],  model_params_1['latent_dim']])
        else:
            encoder = VariationalEncoderFC(
                [self.dataset_params_train_1['feature_dim'], self.training_params['hidden1'],
                    self.training_params['hidden1'],  model_params_1['latent_dim']])
        decoder = DecoderFC([model_params_1['latent_dim'], self.training_params['hidden1'],
                self.training_params['hidden1'], self.dataset_params_train_1['feature_dim']])


        if model_params_2['method'] == 'ae':
            encoder2 = EncoderFC(
                [self.dataset_params_train_2['feature_dim'], self.training_params['hidden2'],
                 self.training_params['hidden2'], model_params_2['latent_dim']])
        else:
            encoder2 = VariationalEncoderFC(
                [self.dataset_params_train_2['feature_dim'], self.training_params['hidden2'],
                 self.training_params['hidden2'], model_params_2['latent_dim']])

        decoder2 = DecoderFC(
            [model_params_2['latent_dim'], self.training_params['hidden2'], self.training_params['hidden2'],
            self.dataset_params_train_2['feature_dim']])

        classifier = ClassifierFC([model_params_1['latent_dim'], 10])
        return encoder, decoder, encoder2, decoder2, classifier

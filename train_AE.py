import argparse
import sys
# Import model util
from utils.model import Model
# Import global configs
import global_config as gc

import torch
import numpy as np

# Get args
parser = argparse.ArgumentParser(
    description='Train a model on the given features/pairs.')

parser.add_argument(
    '--a',
    dest='alpha',
    type=float,
    help='Alpha value.',
    default=0.0001)
parser.add_argument(
    '--b',
    dest='beta',
    type=float,
    help='Beta value for VAE.',
    default=1.0)
parser.add_argument(
    '--x',
    dest='experiment',
    type=str,
    help='Available experiments: {}.'.format(
        gc.available_experiments),
    default="features_ae")
parser.add_argument(
    '--d1',
    dest='domain1',
    type=str,
    help='Domain 1',
    default='svhn_small')
parser.add_argument(
    '--d2',
    dest='domain2',
    type=str,
    help='Domain 2.',
    default='mnist_small')
parser.add_argument(
    '--f1',
    dest='features1',
    type=str,
    help='Features 1.',
    default='mnist_svhn')
parser.add_argument(
    '--f2',
    dest='features2',
    type=str,
    help='Features 2.',
    default='mnist_svhn')
parser.add_argument(
    '--lr',
    dest='learning_rate',
    type=float,
    help='Learning Rate',
    default=5e-4)
parser.add_argument(
    '--z',
    dest='latent_dim',
    type=int,
    help='Dimensions of latent variable z.',
    default=512) # 512
parser.add_argument(
    '--e',
    dest='epochs',
    type=int,
    help='Epochs for coupled AE training. In iterations through the full dataset specified in aeconfig/dataset_params',
    default=100)
parser.add_argument(
    '--t',
    dest='test_every',
    type=int,
    help='Run the test function ever X epochs. Warning: if run with store_latents=True, the latents are only stored every X epochs.',
    default=1)
parser.add_argument(
    '--b1',
    dest='batch_size1',
    type=int,
    help='Batch size for Domain 1',
    default=1000)
parser.add_argument(
    '--b2',
    dest='batch_size2',
    type=int,
    help='Batch size for Domain 2',
    default=1000)
parser.add_argument(
    '--pc',
    dest='per_class',
    type=int,
    help='Supervision per class',
    default=0)
parser.add_argument(
    '--ths',
    dest='threshold',
    type=int,
    help='Threshold for supervision per batch',
    default=1)
parser.add_argument(
    '--hid1',
    dest='hidden1',
    type=int,
    help='Size of the hidden layer AE 1',
    default=256)
parser.add_argument(
    '--hid2',
    dest='hidden2',
    type=int,
    help='Size of the hidden layer AE 2',
    default=512)
parser.add_argument(
    '--sample',
    dest='sample',
    type=int,
    help='Fraction of samples to use per class (0=all data, else number per class)',
    default=0)
parser.add_argument(
    '--scale_latents',
    dest='scale_latents',
    action='store_true',
    help='Wether or not to scale the latents for the computation of the affinity loss.',
    default=False)
parser.add_argument(
    '--use_vae',
    dest='use_vae',
    action='store_true',
    help='Wether or not to use a VAE or AE architecture.',
    default=False)
parser.add_argument(
    '--store_latents',
    dest='store_latents',
    action='store_true',
    help='Wether or not to store the latents and labels in the test function.',
    default=False)
parser.add_argument(
    '--supervision',
    dest='supervision',
    type=bool,
    help='If to use a classifier in source domain',
    default=False)
parser.add_argument(
    '--use_target',
    dest='use_target',
    type=bool,
    help='If to use target supervision',
    default=False)
parser.add_argument(
    '--r',
    dest='restore',
    type=bool,
    help='Should a model with the given parameters be restored',
    default=False)
parser.add_argument(
    '--benchmark',
    dest='benchmark',
    action='store_true',
    help='Wether or not to benchmark the model. If active, it will not train!',
    default=False)
parser.add_argument(
    '--torch_seed',
    dest='torch_seed',
    type=int,
    help='Torch seed',
    default=0)
parser.add_argument(
    '--np_seed',
    dest='np_seed',
    type=int,
    help='Numpy seed',
    default=0)

args = parser.parse_args()

# Import experiment specific configs
if args.experiment in gc.available_experiments:
    #sys.path.append(args.experiment)
    pass
else:
    print(
        "Error, unknonwn experiment {}. If you created a new experiment you have to add it to the available_experiments list in the gloabl config.".format(
            args.experiment))

from features_ae import aeconfig

vc = aeconfig.AEConfig()
torch.manual_seed(args.torch_seed)
np.random.seed(args.np_seed)

if args.use_vae:
    method = 'vae'
else:
    method = 'ae'

vc.model_params_1['method'] = method
vc.model_params_2['method'] = method

vc.model_params_1['beta'] = args.beta
vc.model_params_2['beta'] = args.beta

if not args.restore:
    # Create new model and train it from scratch
    # Fill them in the aeconfig
    vc.model_params_1['latent_dim'] = args.latent_dim
    vc.model_params_2['latent_dim'] = args.latent_dim

    vc.training_params['domain1'] = args.domain1
    vc.training_params['domain2'] = args.domain2
    vc.training_params['features1'] = args.features1
    vc.training_params['features2'] = args.features2

    vc.training_params['learning_rate'] = args.learning_rate
    vc.training_params['train_epochs'] = args.epochs
    vc.training_params['latent_dim'] = args.latent_dim
    vc.training_params['alpha'] = args.alpha
    vc.training_params['per_class'] = args.per_class

    if vc.training_params['per_class'] > 0:
        vc.training_params['use_target'] = True
    else:
        vc.training_params['use_target'] = False

    vc.training_params['sample'] = args.sample
    vc.training_params['threshold'] = args.threshold
    vc.training_params['batch_size1'] = args.batch_size1
    vc.training_params['batch_size2'] = args.batch_size2
    vc.training_params['hidden1'] = args.hidden1
    vc.training_params['hidden2'] = args.hidden2
    vc.training_params['torch_seed'] = args.torch_seed
    vc.training_params['np_seed'] = args.np_seed
    vc.training_params['supervision'] = args.supervision
    vc.training_params['scale_latents'] = args.scale_latents
    vc.training_params['store_latents'] = args.store_latents
    vc.training_params['test_every'] = args.test_every
    vc.training_params['device'] = 'cuda'

    vc.dataset_params_train_1 = vc.get_dataset(
        args.domain1,
        args.features1,
        shuffle=True,
        batch_size=args.batch_size1,
        drop_last=True)
    vc.dataset_params_train_2 = vc.get_dataset(
        args.domain2,
        args.features2,
        shuffle=True,
        batch_size=args.batch_size2,
        drop_last=True)

    model = Model(vc)
else:
    print("Warning untested legacy code")
    # Load existing model and continue training
    model = Model(None)
    restore_path = gc.cfg['results_path'] + '/{}dim_b{}/model.pkl'.format(
        args.latent_dim)
    model = model.load_model(restore_path)

    # Fill params into the aeconfig in case something changed. The other
    # params should not be changed really
    model.aeconfig.model_params['train_epochs'] = args.epochs

try:
    if args.benchmark:
        model.benchmark()
    else:
        model.train()
except KeyboardInterrupt:
    print("Keyboard interrupt detected, trying to shutdown dataloaders...")
    try:
        train_dataset.stop()
        test_dataset.stop()
        print("Dataloaders stopped")
        exit()
    except Exception as e:
        print("Could not stop dataloaders after being done. Probably they already exited naturally")
        exit()

#if __name__ == "__main__":
#    model.train()

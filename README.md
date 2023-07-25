# Linearly Alignable OT

## Parts of the AE framework
Those parts are the general reusable components of the AE.

- `global_config.py`: Here one needs to specifiy the dataset directory and the output directory
to store trained models. When creating new experiments they should be added here as well, otherwise a check will fail.
- `dataloader/`: In this directory you can find the custom dataloader used with the parameters 
defined in `aeconfig.py` files.
- `utils/model.py`: This file contains the definition of a model defined by its encoders and decoders and dataset parameters.
It provides housekeeping, saving and loading functions.
- `utils/ot_functions.py`: This file contains the ot functions used to compute the affinity score.
- `network/coupledAE.py`: This file contains the main training and testing loops for a model.
- `network/train_functions_global.py`: This file contains function to compute for example 
reconstruction loss or affinity loss which can be re-used in experiment specific train function
definitions.

## Configuring an experiment
Experiments are defined in folders. For example features\_ae is the experiment for the mnist/svhn experiment.
In `features_ae/aeconfig.py` you can change all important parameters for the experiment.

## Examples
### Changing the encoder decoder input dimensionality
In order to do this you would go to the experiment/vaeconfig.py of the experiment you want to modify.
In the function `setup_network()` you can change the type of encoder 
(for example from EncoderConv to EncoderFC or your own classes) and its input parameters.

## Usage 
Train a model using
```
python train_AE.py --a [alpha] --z [latent dimenstion for both AEs] --e [train epcohs] --lr [learning_rate]
                    --x [experiment] --d1 [domain1] --d2 [domain2] --f1 [features1] --f2 [features2] 
                    --z [latent_dim] --c [test every c epochs] --use_vae [use a VAE or AE architecture]
                    --pc [number of labelled points per class] --hid1 [size of the hidden layer of source AE]
                    --hid2 [size of the hidden layer of target AE] 
                    --b1 [Batch size for source AE] --b2 [Batch size for target AE] "
```

For example to repeat the mnist / SVHN experiment:
```
'
n train_AE.py --a 3e-5 --lr 5e-4 --b1 1000 --b2 1000 --torch_seed 0 --np_seed 0 --z 2048 --hid1 512 --hid2 512
```

The models will be stored in the models folder specified in `global_config.py` in a subfolder 
called `[experiment]/[setup]/model.pkl`

## Evaluation
You can evaluate the experiments in an experiment folder using:
- `eval_mnist_svhn.py`: Find the best performing model for the SVHN vs MNIST experiment
- `save_embeddings.py`: Use this as an example on how to extract the latents of a fully trained moedl for further experiments.

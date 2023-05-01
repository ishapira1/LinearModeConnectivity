# Linear Mode Connectivity in Deep Learning
itai shaapira, May 1st, 2023

This repository contains the code, data, and results for the study of linear mode connectivity in deep learning. We perform empirical analysis on standard networks for MNIST, CIFAR-10, ImageNet, and Tiny ImageNet. We explore the relationship between training dynamics and linear mode connectivity properties of the models, providing insights for more efficient training algorithms and optimization strategies.

## Directory Structure

- `checkpoints/`: Contains the saved model checkpoints.
- `data/`: Contains the dataset images.
- `dataframes/`: Contains the computed dataframes, one for each dataset.
- `processed/`: Contains the processed results of the experiments.
- `src/`: Contains the source code for the project.
  - `cifar10.py`: Loads and defines the CIFAR-10 models.
  - `mnist.py`: Loads and defines the MNIST models.
  - `imagenet.py`: Loads and defines the ImageNet models.
  - `tiny_imagenet.py`: Loads and defines the Tiny ImageNet models.
  - `resultsHandler.py`: Handles the results of the experiments.
  - `run_experiment.py`: Runs the experiments.
  - `train.py`: Handles the model training process.  This script contains functions to train and evaluate PyTorch neural network models using early stopping.
  - `lmc_utils.py`: Computes linear mode connectivity and related metrics. This script contains utility functions for working with PyTorch neural networks, including interpolation, parameter extraction, distance calculation, and performance evaluation.

## Results

The results of the experiments can be found in the `processed/` directory in a raw form and `dataframes/` in the aggregated form. The files include:

- Loss and accuracy barriers as functions of the dataset, k1, and k2.
- Visualizations of the optimization landscape.
- Analysis of the linear mode connectivity properties of the models.

## Linear Mode Connectivity Experiment
The run_experiment.py file contains an implementation of the linear mode connectivity experiment. The experiment is designed to explore the connectivity between pretrained models and models that continue training from a pretrained model in the context of neural networks.

### On the Efficiency of the algorithm
In this implementation of the linear mode connectivity experiment, we adopt an efficient approach to reduce computational resources.
Instead of training two separate models for each (k1, k2) combination, we train a single model for the maximum number of epochs and then reload the model for different values of k1. 
For each k1 value, we continue training only one new model for an additional k2 steps and compare its performance with the original "main-branch" model at the k1+k2 position. 
This efficient approach reduces the amount of compute required to run the experiment.


### The algorithm
1. Train a model on a given dataset for a specified number of epochs (max_epochs) and batch size. The model checkpoints are saved at every epoch.
2. For selected epochs (k1), continue training the model from the pretrained checkpoint for the remaining epochs. Save these continued training model checkpoints at every epoch.
3. Calculate the distance in weight space between the pretrained model and continued training model for pairs of epochs (k1, k2).
4. Calculate the loss and accuracy along the linear path connecting the pretrained and continued training models in weight space.
5. Process the results and save them to a JSON file.

The results of the experiment will be saved in the processed directory, with one JSON file for each pair of epochs (k1, k2) for each dataset.


## Script Files 
### run_experiment
see previous section
### lmc_utilis
The lmc_utilis.py script provides a collection of utility functions to work with PyTorch neural networks. The functions include extracting and setting network parameters, interpolating between two network's parameters, checking if a network is on a given device, computing loss and accuracy of a model on a dataset, calculating L2 distance between two sets of network parameters, and getting model predictions on a given dataset. These utilities are used in the run_experiment.py script to evaluate and compare the performance of the models trained with different configurations.

### train
The train.py script provides a set of functions for training and evaluating PyTorch neural network models using early stopping. The functions include training a model for one epoch, evaluating the model on a validation or test set, and training the model with early stopping. The early stopping mechanism stops the training process if there is no improvement in the validation loss for a specified number of epochs (patience). This script is essential for training and evaluating the models in the run_experiment.py script while preventing overfitting and optimizing training time.





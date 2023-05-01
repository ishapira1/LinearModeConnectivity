# Linear Mode Connectivity in Deep Learning

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
  - `train.py`: Handles the model training process.
  - `lmc_utils.py`: Computes linear mode connectivity and related metrics.

## Results

The results of the experiments can be found in the `processed/` directory. The files include:

- Loss and accuracy barriers as functions of the dataset, k1, and k2.
- Visualizations of the optimization landscape.
- Analysis of the linear mode connectivity properties of the models.

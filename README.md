![Neural Network Architecture](https://raw.githubusercontent.com/njbabani/MNIST-NN/main/images/nn.svg)

# MNIST-NN

![GitHub License](https://img.shields.io/github/license/njbabani/MNIST-NN?logo=apache&style=flat-square)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&style=flat-square&logoColor=white)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/njbabani/MNIST-NN/.github%2Fworkflows%2Fpython-package-conda.yml?logo=github&style=flat-square)
![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/njbabani/MNIST-NN?logo=codefactor&logoColor=white&style=flat-square)
![Coveralls](https://img.shields.io/coverallsCoverage/github/njbabani/MNIST-NN?logo=coveralls&style=flat-square)
[![GitHub Clones](https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://gist.githubusercontent.com/njbabani/4cf1691c8f9fb7743d2b409f2fc9dff4/raw/clone.json&logo=github&style=flat-square)](https://github.com/MShawon/github-clone-count-badge)
[![PEP8](https://img.shields.io/badge/code%20style-PEP%208-orange?style=flat-square)](https://www.python.org/dev/peps/pep-0008/)
<!-- ![GitHub repo size](https://img.shields.io/github/repo-size/njbabani/MNIST-NN?style=flat-square) -->

This project implements a feedforward Artifical Neural Network (ANN) from scratch for the MNIST dataset using NumPy. This project was inspired by Andrew Ng's course on Deep Learning where we can code an L-layer Neural Network (NN) from scratch using a functional programming approach. This project also builds upon work by [am1tyadav](https://github.com/am1tyadav/Neural-Network-from-Scratch-Python/tree/master) which used an Object-Oriented Programming (OOP) for constructing NN models. For this project, I wanted to create an interface for building NNs that was intuitive and somewhat similar to Keras so users would feel comfortable implementing models quickly.

## Features
- Custom Architecture: Easily adjust network depth, layer sizes, and activation functions
- Activation Functions: Supports ReLU, Sigmoid, Softmax, and Linear functions
- Loss Functions: Includes Mean Squared Error (MSE), Binary Cross-Entropy (BCE), and Categorical Cross-Entropy (CCE)
- Optimizers: Implements Stochastic Gradient Descent (SGD) with options for learning rate scheduling
- Regularisation: Supports L2 regularisation to prevent overfitting
- Callbacks: Implements callbacks for early stopping, learning rate adjustment, progress bar reports, and save/loading of model parameters
- Visualisation: Displays correct and incorrect classification for images in the MNIST dataset
- Testing Framework: Includes unit tests using pytest to ensure code reliability

## Installation
### Prerequisites
- Anaconda or Miniconda: For managing the Conda environment
- Python 3.x
- Git: To clone the repository

### Option 1: Download the latest release
#### Download the release
```
git clone https://github.com/njbabani/MNIST-NN.git
cd ~/MNIST-NN
```

### Step 2: Setup virtual environment
```
conda env create -f env/environments.yml
conda activate mnist
```

## Usage

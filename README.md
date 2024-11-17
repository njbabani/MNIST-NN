![Neural Network Architecture](https://raw.githubusercontent.com/njbabani/MNIST-NN/main/images/nn-white.svg)

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
- Optimisers: Implements Stochastic Gradient Descent (SGD) with options for learning rate scheduling
- Mini-Batch Gradient Descent: Divides training set into batches and uses SGD to update weights and biases
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
[![Download Latest Release](https://img.shields.io/github/v/release/njbabani/MNIST-NN?color=brightgreen&label=Download%20Latest%20Release&style=flat-square)](https://github.com/njbabani/MNIST-NN/releases/latest)

1. **Click the button above** to go to the latest release
2. On the GitHub releases page, find the section titled **"Assets"**
3. Download the file named **`Source code (zip)`** or **`Source code (tar.gz)`**
4. Extract the downloaded archive to a directory of your choice

### Option 2: Clone the repository
If you prefer to have the latest code under development or want to contribute, you can clone the repository:

#### Step 1: Clone the repo
```bash
git clone https://github.com/njbabani/MNIST-NN.git
cd ~/MNIST-NN
```

#### Step 2: Setup virtual environment
```bash
conda env create -f env/environments.yml
conda activate mnist
```

#### Step 3: Verify environment (optional)
After activating the virtual environment, you can verify that all dependencies like ```numpy``` and ```matplotlib``` are installed:
```bash
conda list
```

## Usage
### Example: Training the Neural Network
To run the example script that trains the neural network on the MNIST dataset, run the following command:
```bash
python main.py
```

This script:

- Loads and preprocesses the MNIST dataset by normalising it & performing one-hot encoding 
- Defines the neural network architecture
- Compiles the model with the CCE loss function and SGD
- Trains the model and displays a progress bar
- Evaluates the model on the test set, showing test loss and accuracy
- Displays correctly and incorrectly predicted images for comparison

### Creating your own the Network Architecture
You can customise your own neural network by creating a new Python script:

#### Step 1: Import libraries

```python
from datasets.data import load_mnist_data
from src.layers.layer import Dense, Dropout
from src.layers.activation import ReLU, Softmax
from src.layers.loss import CCE
from src.optimisation.optimiser import SGD
from src.models.model import FeedForwardNN
from src.common.callback import ProgbarLogger
from src.common.utils import display_images
```

#### Step 2: Load the one-hot encoded MNIST dataset

```python
# `verbose` displays the shapes for the variables and `encode` enables one-hot encoding
x_train, y_train, x_test, y_test = load_mnist_data(verbose=True, encode=True)
```

#### Step 3: Define the NN model parameters

```python
# Specify the number of output neurons based on the total possible classification labels (10)
output_units = y_train.shape[0]

# Specify number of epochs
epochs = 5

# Batch size for gradient descent
batch_size = 32

# Regularisation factor
l2_regulariser = 0.1

# Learning rate
alpha = 0.01
```

#### Step 4: Define your hidden layer
You **do not** need to explicitly define an input layer since this is handled internally by the ```Dense``` layer when using its ```build``` method:

```python
# Create a `list` of the hidden layers
hidden_layers = [
    # First hidden layer with more neurons
    Dense(256),
    ReLU(),

    # Second layer with dropout
    Dropout(0.2),
    Dense(128),
    ReLU(),

    # Third dense layer
    Dense(64),
    ReLU(),

    # Output layer with softmax for categorical classification
    Dense(output_units),
    Softmax()
]
```

For more details on how the build method works, refer to the [Additional information: Input layer](#additional-information-input-layer) section

#### Step 5: Create the ANN with the specified layers, loss function, and optimiser

```python
# Create an Artificial Neural Network with L2 regularisation
ANN = FeedForwardNN(layers=hidden_layers, lambda_reg=l2_regulariser)

# Compile the model with a loss function and optimiser
ANN.compile(loss=CCE(), optimiser=SGD(learning_rate=alpha))
```

#### Step 6: Add callbacks like early stopping or a progress bar for training

```python
# Stops training early if the loss does not improve
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=True)

# Creates a progress bar to monitor training at the end of each epoch
progbar_logger = ProgbarLogger(verbose=True, total_epochs=epochs)

# Create a list of all epochs
callbacks = [progbar_logger, early_stopping]
```
#### Step 7: Train the model
```python
# Train the model by fitting it to the training data
ANN.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
```

#### Additional information: Input layer
The ```Dense``` class will call the ```build``` method when the first forward pass is conducted. As the expected activation functions to be used are ReLU functions, we use He initialisation for the weights and biases:
```python
class Dense(Layer)
  ...
      def build(self, data: np.ndarray):
        """
        Initialises weights and biases using He initialisation

        Args:
            data (np.ndarray): Input data for determining the shape.
        """
        self._input_units = data.shape[0]
        random_weights = np.random.randn(self._hidden_units, self._input_units)
        scaling_factor = np.sqrt(2.0 / self._input_units)
        self._weights = random_weights * scaling_factor
        self._bias = np.zeros((self._hidden_units, 1))

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass for the dense layer

        Args:
            data (np.ndarray): Input data to the layer

        Returns:
            np.ndarray: Output of the layer after applying weights and biases
        """
        if self._weights is None:
            self.build(data)

        self._output = np.dot(self._weights, data) + self._bias
        return self._output
```

## Project Structure
```plaintext
MNIST-NN/
├── datasets/
│   ├── data.py
│   └── __init__.py
├── env/
│   ├── environments.yml
│   └── requirements.txt
├── images/
│   ├── nn-white.svg
│   └── nn.svg
├── src/
│   ├── common/
│   │   ├── callback.py
│   │   ├── utils.py
│   │   └── __init__.py
│   ├── layers/
│   │   ├── activation.py
│   │   ├── layer.py
│   │   ├── loss.py
│   │   └── __init__.py
│   ├── models/
│   │   ├── model.py
│   │   └── __init__.py
│   └── optimisation/
│       ├── optimiser.py
│       └── __init__.py
├── tests/
│   ├── test_activation.py
│   ├── test_callback.py
│   ├── test_data.py
│   ├── test_layer.py
│   ├── test_loss.py
│   ├── test_model.py
│   ├── test_optimiser.py
│   ├── test_utils.py
│   └── __init__.py
├── .coverage
├── .gitignore
├── CLONE.md
├── CODE_OF_CONDUCT.md
├── LICENSE
├── main.py
└── README.md
```

## License
This project is licensed under the Apache License Version 2.0. See the [LICENSE](LICENSE) file for details.

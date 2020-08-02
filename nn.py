import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class Activation_ReLU:
	def forward(self, inputs):
        # Calculate output values from input ones
        self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities

# Create dataset
X, y = spiral_data(100, 3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs (each sample has 2 features), 3 outputs
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()
# Make a forward pass of our training data through this layer
dense1.forward(X)
# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)
# Make a forward pass through second Dense layer 
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)
print(activation2.output[:5])

# layer_outputs = [4.8, 1.21, 2.385]  # values we got as an output earlier when we described what neural network is

# # For each value in a vector, calculate the exponential value
# exp_values = np.exp(layer_outputs)
# print('exponentiated values:')
# print(exp_values)

# # Now normalize values
# norm_values = exp_values / np.sum(exp_values)

# print('normalized exponentiated values:')
# print(norm_values)
# print('sum of normalized values:', np.sum(norm_values))

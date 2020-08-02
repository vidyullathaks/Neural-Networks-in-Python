import numpy as np

inputs = np.array([[1, 2, 3, 2.5], [2., 5., -1., 2], [-1.5, 2.7, 3.3, -0.8]])  # now we have 3 samples (feature sets with 4 features in each set) of data
weights = np.array([[0.2, 0.8, -0.5, 1], # now we have 3 sets of weights - one set for each neuron
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T
biases = np.array([[2], [3], [0.5]]).T  # one bias for each neuron

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases  # forward pass thru Dense layer
relu_outputs = np.maximum(0, layer_outputs)  # forward pass thru ReLU activation

# Let's optimize and test backpropagation here
# ReLU activation
relu_dvalues = np.ones(relu_outputs.shape) # simulates derivative with respect to input values from next layer passed to current layer during backpropagation
relu_dvalues[layer_outputs <= 0] = 0

drelu = relu_dvalues

# Dense layer
dinputs = np.dot(drelu, weights.T)  # dinputs - multiply by weights
dweights = np.dot(inputs.T, drelu)  # dweights - multiply by inputs
dbiases = np.sum(drelu, axis=0, keepdims=True)  # dbiases - sum values, do this over samples (first axis), keepdims as this by default will produce a plain list - we discussed this earlier

# Update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases


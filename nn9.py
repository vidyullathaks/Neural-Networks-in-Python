import numpy as np
import random

random.seed(0)
np.random.seed(0)


# Our sample dataset
def create_data(n, k):
    X = np.zeros((n*k, 2))  # data matrix (each row = single example)
    y = np.zeros(n*k, dtype='uint8')  # class labels
    for j in range(k):
        ix = range(n*j, n*(j+1))
        r = np.linspace(0.0, 1, n)  # radius
        t = np.linspace(j*4, (j+1)*4, n) + np.random.randn(n)*0.2  # theta
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = j
    return X, y


# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, inputs, neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input ones
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable, 
        # let's make a copy of values first
        self.dvalues = dvalues.copy()

        # Zero gradient where input values were negative 
        self.dvalues[self.inputs <= 0] = 0 


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        self.dvalues = dvalues.copy()


# Cross-entropy loss
class Loss_CategoricalCrossentropy:

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = y_pred.shape[0]

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            y_pred = y_pred[range(samples), y_true]

        # Losses
        negative_log_likelihoods = -np.log(y_pred)

        # Mask values - only for one-hot encoded labels
        if len(y_true.shape) == 2:
            negative_log_likelihoods *= y_true

        # Overall loss
        data_loss = np.sum(negative_log_likelihoods) / samples
        return data_loss

    # Backward pass
    def backward(self, dvalues, y_true):

        samples = dvalues.shape[0]

        self.dvalues = dvalues.copy()  # Copy so we can safely modify
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples


# Create dataset
X, y = create_data(100, 3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs (each sample has 2 features), 3 outputs

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Make a forward pass of our training data thru this layer
dense1.forward(X)

# Make a forward pass thru activation function - we take output of previous layer here
activation1.forward(dense1.output)

# Make a forward pass thru second Dense layer - it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass thru activation function - we take output of previous layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])

# Calculate loss from output of activation2 (softmax activation)
loss = loss_function.forward(activation2.output, y)

# Print loss value
print('loss:', loss)

# Calculate accuracy from output of activation2 and targets
predictions = np.argmax(activation2.output, axis=1)  # calculate values along first axis
accuracy = np.mean(predictions==y)

print('acc:', accuracy)

# Backward pass
loss_function.backward(activation2.output, y)
activation2.backward(loss_function.dvalues)
dense2.backward(activation2.dvalues)
activation1.backward(dense2.dvalues)
dense1.backward(activation1.dvalues)

# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)

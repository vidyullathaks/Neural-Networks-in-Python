import numpy as np

np.random.seed(0)

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.05
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

# Dense layer
class Layer_Dense:
    def __init__(self, inputs, neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax:
    def forward(self, inputs):
        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
# Cross-entropy loss
class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Probabilities for target values
        y_pred = y_pred[range(samples), y_true]
        # Losses
        negative_log_likelihoods = -np.log(y_pred)
        # Average loss
        data_loss = np.sum(negative_log_likelihoods) / samples
        return data_loss

# Create dataset
X, y = create_data(100, 3)
dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs (each sample has 2 features), 3 outputs
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])
# Calculate loss from output of activation2 (softmax activation)
loss = loss_function.forward(activation2.output, y)
print('loss:', loss)

# Calculate accuracy from output of activation2 and targets
predictions = np.argmax(activation2.output, axis=1)  # calculate values along first axis
accuracy = np.mean(predictions==y)

print('acc:', accuracy)
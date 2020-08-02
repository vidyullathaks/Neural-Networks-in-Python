class Optimizer_SGD:

    def __init__(self, learning_rate=1., decay=0.1):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Create dataset
X, y = create_data(100, 3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 64)  # first dense layer, 2 inputs (each sample has 2 features), 64 outputs

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 64 input features (as we take output of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(64, 3)  # second dense layer, 64 inputs, 3 outputs

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_SGD()

# Train in loop
for epoch in range(10001):

    # Make a forward pass of our training data thru this layer
    dense1.forward(X)

    # Make a forward pass thru activation function
    # we take output of previous layer here
    activation1.forward(dense1.output)

    # Make a forward pass thru second Dense layer 
    # outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # Make a forward pass thru activation function
    # we take output of previous layer here
    activation2.forward(dense2.output)

    # Calculate loss from output of activation2 so softmax activation
    loss = loss_function.forward(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1) 
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print('epoch:', epoch, 'acc:', accuracy, 'loss:', loss)

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dvalues)
    dense2.backward(activation2.dvalues)
    activation1.backward(dense2.dvalues)
    dense1.backward(activation1.dvalues)

    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
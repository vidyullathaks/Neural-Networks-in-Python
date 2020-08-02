import numpy as np 

# input and output for output layer inputs = X
X = [[1, 2, 3, 2.5],
		[2.0,5.0,-1.0,2.0],
		[-1.5,2.7,3.3,-0.8]]

np.random.seed(0)

# n_neurons is the expected number of neurons in the output
class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
		self.biases = np.zeros((1, n_neurons))
		# pass
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases
		# pass

class Activation_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0,inputs)

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
# print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)
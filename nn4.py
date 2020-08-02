import numpy as np 
import nnfs
from nnfs.dataset import sprial_data
nnfs.init()


# input and output for output layer inputs = X
X = [[1, 2, 3, 2.5],
		[2.0,5.0,-1.0,2.0],
		[-1.5,2.7,3.3,-0.8]]

X,y = sprial_data(100, 3)


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

layer1 = Layer_Dense(2,5)
# n_input is 2 because in spiral_data we have 2 (x-asix and y-axis) inputs

Activation1 = Activation_ReLU()

layer1.forward(X)
Activation1.forward(layer1.output)
# print(Activation1.forward(layer1.output))
from abc import abstractmethod

import numpy as np


np.random.seed(0)

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss = loss

    def forward(self, input, y=None):
        current_output = input
        for layer in self.layers:
            layer.forward(current_output)
            current_output = layer.output

        if self.loss and y is not None:
            return self.loss.calculate(current_output, y)

        return current_output

    def backward(self, d_values, y):
        if self.loss:
            d_values = self.loss.backward(d_values, y)

        for layer in reversed(self.layers):
            layer.backward(d_values)
            d_values = layer.dinputs

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0

            # Iterate through every sample
            for i in range(len(X)):
                X_single = X[i]
                y_single = y[i]

                # Forward pass for single sample
                loss = self.forward(X_single, y_single)
                total_loss += loss

                # Backward pass
                self.backward(self.layers[-1].output, y_single)

                # Update weights and biases
                for layer in self.layers:
                    if hasattr(layer, 'weights'): # update layers with weights
                        layer.weights -= learning_rate * layer.d_weights
                        layer.biases -= learning_rate * layer.d_biases

            # Show average loss
            avg_loss = total_loss / len(X)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Avg Loss: {avg_loss}')



class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.inputs = None
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) # multiplied by `0.1` to gen smaller initial values
        self.biases = np.zeros((1, n_neurons))

        self.d_inputs = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, inputs):
        # Output is simple NN [output = input * weights + biases]
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values):

        # Get gradient of weights [d_weights = d_values * inputs^(-1)]
        self.d_weights = np.dot(self.inputs.reshape(-1, 1), d_values.reshape(1, -1))

        # Get gradient of bias [d_bias = d_values]
        self.d_biases = d_values

        # Get gradient of input
        self.d_inputs = np.dot(d_values, self.weights.T)


class ActivationReLU:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.d_inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, d_values):
        self.d_inputs = d_values.copy()

        # Get gradient of ReLU [x for x>=0 else 0]
        self.d_inputs[self.inputs <= 0] = 0

class ActivationSoftmax:
    def __init__(self):
        self.output = None

        self.d_inputs = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs))  # to not have to big values
        probabilities = exp_values / np.sum(exp_values)
        self.output = probabilities

    # https://medium.com/@jsilvawasd/softmax-and-backpropagation-625c0c1f8241
    def backward(self, d_values):
        # Get gradient of Softmax [d_inputs = d_output_i - d_output_i * d_output_j]
        single_output = self.output.reshape(-1, 1)
        jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
        self.d_inputs = np.dot(jacobian_matrix, d_values)


class Loss:
    def calculate(self, output, y):
        sample_loses = self.forward(output,y)
        data_loss = np.mean(sample_loses)
        return data_loss

    @abstractmethod
    def forward(self, output, y):
        pass

    @abstractmethod
    def backward(self, d_values, y):
        pass



class CategoricalCrossEntropyLoss(Loss):
    def __init__(self):
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if isinstance(y_true, int):
            correct_confidence = y_pred_clipped[y_true]
        elif isinstance(y_true, np.ndarray) and y_true.shape[0] == y_pred.shape[0]:
            correct_confidence = np.sum(y_pred_clipped * y_true)
        else:
            raise ValueError("Invalid shape or type for y_true")

        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood

    # https://medium.com/@jsilvawasd/softmax-and-backpropagation-625c0c1f8241
    def backward(self, d_values, y_true):
        # Get gradient [d_input = - y_true / d_output] (derivative of log)
        labels = len(d_values)

        if isinstance(y_true, int):
            y_true = np.eye(labels)[y_true]

        self.d_inputs = -y_true / d_values






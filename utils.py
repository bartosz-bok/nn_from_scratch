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

    def backward(self, dvalues, y):
        if self.loss:
            dvalues = self.loss.backward(dvalues, y)

        for layer in reversed(self.layers):
            layer.backward(dvalues)
            dvalues = layer.dinputs

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0

            # Iterowanie po każdej próbce osobno
            for i in range(len(X)):
                X_single = X[i]  # Pobieranie jednej próbki
                y_single = y[i]  # Pobieranie jednej etykiety

                # Forward pass dla jednej próbki
                loss = self.forward(X_single, y_single)
                total_loss += loss

                # Backward pass
                self.backward(self.layers[-1].output, y_single)

                # Aktualizacja wag i biasów
                for layer in self.layers:  # Stochastic Gradient Descent (SGD)
                    if hasattr(layer, 'weights'):
                        layer.weights -= learning_rate * layer.dweights
                        layer.biases -= learning_rate * layer.dbiases

            # Wyświetlanie średniego loss co 100 epok
            avg_loss = total_loss / len(X)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Avg Loss: {avg_loss}')



class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.inputs = None
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) # multiplied by `0.1` to gen smaller initial values
        self.biases = np.zeros((1, n_neurons))

        self.dinputs = None
        self.dweights = None
        self.dbiases = None

    def forward(self, inputs):
        # Zakładamy, że inputs jest wektorem dla jednej próbki, np. (n_inputs,)
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # dvalues to wektor, np. (n_neurons,)

        # Obliczenie gradientu wag (jednowymiarowe wejścia)
        self.dweights = np.dot(self.inputs.reshape(-1, 1), dvalues.reshape(1, -1))

        # Obliczenie gradientu biasów - bez sumowania, bo jest jedna próbka
        self.dbiases = dvalues

        # Obliczenie gradientu wejść - zwracane w formie wektora
        self.dinputs = np.dot(dvalues, self.weights.T)


class ActivationReLU:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class ActivationSoftmax:
    def __init__(self):
        self.output = None

        self.dinputs = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs))  # Działa na wektorze
        probabilities = exp_values / np.sum(exp_values)
        self.output = probabilities

    def backward(self, dvalues):
        single_output = self.output.reshape(-1, 1)
        jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
        self.dinputs = np.dot(jacobian_matrix, dvalues)


class Loss:
    def calculate(self, output, y):
        sample_loses = self.forward(output,y)
        data_loss = np.mean(sample_loses)
        return data_loss

    @abstractmethod
    def forward(self, output, y):
        pass

    @abstractmethod
    def backward(self, dvalues, y):
        pass


class CategoricalCrossEntropyLoss(Loss):
    def __init__(self):
        self.dinputs = None

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

    def backward(self, dvalues, y_true):
        labels = len(dvalues)

        if isinstance(y_true, int):
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues






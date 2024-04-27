import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

class NeuronParams:
    def __init__(self, dimensions, learning_rate=0.1, activation_function='linear', beta=100):
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.beta = beta

class NeuronNode:
    def __init__(self, params: NeuronParams, layer_num=0, id=0):
        self.weights = np.array([1]*params.dimensions)
        self.bias = 1
        self.parents = []
        self.children = []
        self.id = id
        self.layer_num = layer_num
        self.min_weights = None
        self.min_bias = None
        self.learning_rate = params.learning_rate
        self.activation_function = params.activation_function
        self.beta = params.beta
 
    def add_parent(self, parent):
        self.parents.append(parent)

    def add_children(self, child):
        self.children.append(child)

    def print_tree(self):
        print(f'Layer: {self.layer_num} - Neuron: {self.id} - Parents: {[parent.id for parent in self.parents]} - Weights: {self.weights}')
        for child in self.children:
            child.print_tree()       
            
    def set_min_weights(self):
        self.min_weights = self.weights
        self.min_bias = self.bias
        for child in self.children:
            child.set_min_weights()

    activation_functions = {
        'step': lambda x, b: 1 if x >= 0 else -1,
        'linear': lambda x, b=0: x,
        'sigmoid': lambda x, b: 1 / (1 + np.exp(-2*b*x)),
        'tan_h': lambda x, b: np.tanh(b*x)
    }

    derivative_activation_functions = {
        'linear': lambda x, b=0: 1,
        'sigmoid': lambda x, b: 2*b*np.exp(-2*b*x) / (1 + np.exp(-2*b*x))**2,
        'tan_h': lambda x, b: b*(1 - np.tanh(b*x)**2),
        'step': lambda x, b: 1
    }        

    def compute_excitement(self, data_input, best=False):
        weights = self.min_weights if best else self.weights
        bias = self.min_bias if best else self.bias
        if len(self.children) != 0:
            data_input = [child.predict(data_input, best) for child in self.children]
        return sum(data_input * weights) + bias

    def compute_activation(self, excitement):
        return self.activation_functions[self.activation_function](excitement, self.beta)

    def predict(self, data_input, best=False):
        return self.compute_activation(self.compute_excitement(data_input, best))
    
    def calculate_delta_weights(self, data_input, expected_output):
        aux = sum(parent.calculate_delta_weights(data_input, expected_output)[1]*parent.weights[self.id] for parent in self.parents)
        excitement = self.children[self.id].compute_excitement(data_input)
        derivative = self.derivative_activation_functions[self.activation_function](excitement, self.beta)

        if len(self.parents) == 0:
            difference = expected_output - self.predict(data_input)
            derivative = self.derivative_activation_functions[self.activation_function](self.compute_excitement(data_input), self.beta)
            delta = difference * derivative
            delta_weights = []
            for i in range(len(self.weights)):
                delta_weights.append(self.learning_rate * delta * self.children[i].predict(data_input))
            return delta_weights, delta
    
        delta_weights = []
        for i in range(len(self.weights)):
            delta_weights.append(self.learning_rate * aux * derivative * self.children[i].predict(data_input))
        delta = aux * derivative
        return delta_weights, delta

    def train(self, data_input, expected_output):
        delta_weights, delta = self.calculate_delta_weights(data_input, expected_output)
        self.weights = self.weights + delta_weights
        self.bias = self.bias + self.learning_rate * delta
        return delta, self.weights


class NeuronTree:
    def __init__(self, params: list, max_iter=1000):
        first_layer_params = params.pop()
        self.root = NeuronNode(first_layer_params[0])  # Create root node
        self.max_iter = max_iter
        self.previous_layer_nodes = [self.root]
        self.min_error = sys.maxsize
        self._populate_tree(params)  # Populate the rest of the tree

    def _populate_tree(self, params, layer_num=1):
        if params:  # If there are still layers to process
            layer_params = params.pop()  # Get the parameters for the current layer
            current_layer_nodes = []  # List to store the nodes in the current layer
            for i, param in enumerate(layer_params):
                node = NeuronNode(param, layer_num, i)  # Create a new node
                node.parents = self.previous_layer_nodes  # Set the parents to the nodes in the previous layer
                for parent in self.previous_layer_nodes:  # Add the new node as a child to all nodes in the previous layer
                    parent.children.append(node)
                current_layer_nodes.append(node)  # Add the new node to the current layer's nodes
            self.previous_layer_nodes = current_layer_nodes  # Update the previous layer's nodes to the current layer's nodes
            self._populate_tree(params, layer_num+1)  # Recursively populate the rest of the tree

    def train(self, data_input, expected_output):
        iterations = 0
        while iterations < self.max_iter and self.min_error > 0:
            mu = np.random.randint(0, len(data_input))
            input_value = data_input[mu]
            self.root.train(input_value, expected_output[mu])

            error = sum((expected_output[mu] - self.process(data_input[mu]))**2 for mu in range(0, len(data_input)))/2
            if error < self.min_error:
                self.min_error = error
                self.root.set_min_weights()
            iterations += 1
        return self.min_error

    def predict(self, data_input):
        return self.root.predict(data_input, True)
    
    def process(self, data_input):
        return self.root.predict(data_input, False)

def calculate_error(data, expected_output):
    return sum([abs(expected_output[mu] - data[mu])**2 for mu in range(0, len(data))])/2

def main():
    # data = pd.read_csv('TP3-ej2-conjunto.csv')

    # example_data_input = data[[col for col in data.columns if col.startswith('x')]].values    
    # example_data_output = data['y'].values

    example_data_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    # example_data_input = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]])
    example_data_input = np.insert(example_data_input, 0, 1, axis=1)
    example_data_output = np.array([1, 1, -1, -1])
    # example_data_output = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])

    dimensions = len(example_data_input[0])

    layer_one = [NeuronParams(dimensions, 0.01, 'linear', 0.1) for _ in range(3)]
    layer_two = [NeuronParams(len(layer_one), 0.01, 'linear', 0.1) for _ in range(1)]
    # layer_three = [NeuronParams(len(layer_two), 0.01, 'linear', 0.01) for _ in range(1)]

    neurons_params = [layer_one, layer_two]
    tree = NeuronTree(neurons_params, 1000)
    print(tree.root.print_tree())
    min_error = tree.train(example_data_input, example_data_output)
    print(f'Min error: {min_error}')

    result_1 = tree.predict(example_data_input[0])
    result_2 = tree.predict(example_data_input[1])
    result_3 = tree.predict(example_data_input[2])
    result_4 = tree.predict(example_data_input[3])
    print(f'Result 1: {result_1}')
    print(f'Result 2: {result_2}')
    print(f'Result 3: {result_3}')
    print(f'Result 4: {result_4}')

    fig, ax = plt.subplots()
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.array([[tree.predict([1, x_val, y_val]) for x_val in x] for y_val in y])
    print(Z)

    ax.contourf(X, Y, Z, levels=1, cmap='coolwarm')
    ax.scatter(example_data_input[:, 1], example_data_input[:, 2], c=example_data_output, cmap='coolwarm')

    plt.show()


if __name__ == "__main__":
    main()
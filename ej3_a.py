import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class NeuronParams:
    def __init__(self, dimensions, learning_rate=0.1, activation_function='linear', beta=100):
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.beta = beta
   
class NeuralNetwork:
    def __init__(self, params: list, max_iter=1000):
        self.layers = [Layer(params[i], i) for i in range(len(params))]
        self.min_error = sys.maxsize
        self.max_iter = max_iter

    def predict(self, data_input, best=True):
        results = []
        next_layer_input = data_input
        for layer in self.layers:
            next_layer_input = layer.process(next_layer_input, best)
            results.append(next_layer_input)
        return results[-1]
    
    def process(self, data_input):
        return self.predict(data_input, False)

    def get_weights(self):
        return [[(neuron.weights, neuron.bias) for neuron in layer.neurons] for layer in self.layers]
    
    def set_min_weights(self, weights):
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                neuron.min_weights = weights[i][j][0]
                neuron.min_bias = weights[i][j][1]

    def print_weights(self):
        for layer in self.layers:
            print(f'Layer: {layer.id}')
            for neuron in layer.neurons:
                print(f'Neuron: {neuron.id} - Weights: {neuron.weights} - Bias: {neuron.bias}')

    def train(self, data_input, expected_output):
        iteration = 0
        best_weights_history = []
        error_history = []
        while self.min_error > 1e-3 and iteration < self.max_iter:
            mu = np.random.randint(0, len(data_input))

            results = []
            next_layer_input = data_input[mu]
            for layer in self.layers:
                next_layer_input = layer.process(next_layer_input)
                results.append(next_layer_input)
            
            # print(f'Results: {results}')
            next_layer_deltas, next_layer_weights = self.layers[-1].train_last_layer(expected_output[mu], results[-2])

            for layer in list(reversed(self.layers))[1:]:
                training_input = results[layer.id-1]
                if layer.id == 0:
                    training_input = data_input[mu]
                next_layer_deltas, next_layer_weights = layer.train(training_input, next_layer_deltas, next_layer_weights)

            error = sum((expected_output[mu] - self.process(data_input[mu]))**2 for mu in range(0, len(expected_output)))/2

            # error = 0
            # for i in range(len(data_input)):
            #     result = self.process(data_input[i])[0]
            #     result = 0 if result <= 0 else 1
            #     expected = 0 if expected_output[i] == -1 else 1
            #     # print(f'Expected: {expected} - Result: {result}')
            #     error += binary_crossentropy(expected, result)


            if error < self.min_error:
                # print(f'Iteration: {iteration} - Error: {error}')
                error_history.append(error)
                best_weights_history.append(self.get_weights())
                self.min_error = error
                for layer in self.layers:
                    layer.set_min_weights()

            iteration += 1
        return self.min_error, iteration, best_weights_history, error_history
    
def binary_crossentropy(expected_output, result):
    epsilon = 1e-10
    result = np.clip(result, epsilon, 1 - epsilon)
    return -expected_output * np.log(result + epsilon) - (1 - expected_output) * np.log(1 - result + epsilon)

class Layer:
    def __init__(self, params: list, id):
        self.neurons = [Neuron(params[i], i, id) for i in range(len(params))]
        self.id = id

    def set_min_weights(self):
        for neuron in self.neurons:
            neuron.min_weights = neuron.weights
            neuron.min_bias = neuron.bias

    def process(self, data_input, best=False):
        return [neuron.predict(data_input, best) for neuron in self.neurons]

    def train_last_layer(self, expected_output, value):
        layer_deltas = []
        layer_weights = []
        for i, neuron in enumerate(self.neurons):
            neuron_delta, neuron_weights = neuron.train_last_layer(expected_output[i], value)
            layer_deltas.append(neuron_delta)
            layer_weights.append(neuron_weights)
        
        return layer_deltas, layer_weights

    def train(self, data_input, next_layer_deltas, next_layer_weights):
        layer_deltas = []
        layer_weights = []
        for i, neuron in enumerate(self.neurons):
            neuron_delta, neuron_weights = neuron.train(data_input, next_layer_deltas, next_layer_weights)
            layer_deltas.append(neuron_delta)
            layer_weights.append(neuron_weights)
        return layer_deltas, layer_weights


class Neuron:
    def __init__(self, params: NeuronParams, id, layer_id=None):
        self.weights = np.random.rand(params.dimensions)
        self.id = id
        self.layer_id = layer_id
        self.min_weights = None
        self.bias = np.random.rand()
        self.min_bias = None
        self.learning_rate = params.learning_rate
        self.activation_function = params.activation_function
        self.beta = params.beta

    activation_functions = {
        'linear': lambda x, b=0: x,
        'sigmoid': lambda x, b: 1 / (1 + np.exp(-2*b*x)),
        'tan_h': lambda x, b: np.tanh(b*x)
    }

    derivative_activation_functions = {
        'linear': lambda x, b=0: 1,
        'sigmoid': lambda x, b: 2*b*np.exp(-2*b*x) / (1 + np.exp(-2*b*x))**2,
        'tan_h': lambda x, b: b*(1 - np.tanh(b*x)**2)
    }        

    def compute_excitement(self, data_input, best=False):
        weights = self.min_weights if best else self.weights
        bias = self.min_bias if best else self.bias
        return sum(data_input * weights) + bias

    def compute_activation(self, excitement):
        return self.activation_functions[self.activation_function](excitement, self.beta)

    def predict(self, data_input, best=False):
        return self.compute_activation(self.compute_excitement(data_input, best))
    
    def calculate_delta_weights(self, data_input, next_layer_deltas, next_layer_weights):
        excitement = self.compute_excitement(data_input)
        aux = sum(next_layer_deltas[i] * next_layer_weights[i][self.id] for i in range(len(next_layer_deltas)))
        # print(f'Layer: {self.layer_id} - Neuron: {self.id} - Aux: {aux}')
        derivative = self.derivative_activation_functions[self.activation_function](excitement, self.beta)
        delta = aux * derivative
        return self.learning_rate * delta, delta
    
    def train_last_layer(self, expected_output, value):
        excitement = self.compute_excitement(value)
        activation = self.compute_activation(excitement)
        derivative = self.derivative_activation_functions[self.activation_function](excitement, self.beta)
        delta = (expected_output - activation) * derivative

        delta_weights = []
        for i in range(len(self.weights)):
            delta_weights.append(self.learning_rate * delta * value[i])

        self.weights = self.weights + delta_weights
        self.bias = self.bias + self.learning_rate * delta
        return delta, self.weights

    def train(self, data_input, next_layer_delta, next_layer_weights):
        delta_weight, delta = self.calculate_delta_weights(data_input, next_layer_delta, next_layer_weights)
        delta_weights = []

        for i in range(len(self.weights)):
            delta_weights.append(delta_weight * data_input[i])

        self.weights = self.weights + delta_weights
        self.bias = self.bias + delta_weight
        return delta, self.weights

def main():
    # data = pd.read_csv('TP3-ej2-conjunto.csv')

    # example_data_input = data[[col for col in data.columns if col.startswith('x')]].values    
    # example_data_output = data['y'].values

    example_data_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    # example_data_input = np.insert(example_data_input, 0, 1, axis=0)
    example_data_output = np.array([[1], [1], [-1], [-1]])

    dimensions = len(example_data_input[0])
    layer_one = [NeuronParams(dimensions, 1, 'linear', 0.1) for _ in range(1)]
    layer_two = [NeuronParams(len(layer_one), 1, 'tan_h', 0.2) for _ in range(5)]
    layer_three = [NeuronParams(len(layer_two), 1, 'tan_h', 0.5) for _ in range(1)]

    neurons_params = [layer_one, layer_two, layer_three]

    neural_network = NeuralNetwork(neurons_params, 10000)
    min_error, iterations, best_weights_history, error_history = neural_network.train(example_data_input, example_data_output)
    neural_network.print_weights()
    print(f'Min error: {min_error} - Iterations: {iterations}')


    for i, input_data in enumerate(example_data_input):
        result = neural_network.predict(input_data)
        print(f'Result {i}: {result} - expected: {example_data_output[i]}')

    fig, ax = plt.subplots()
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.array([[neural_network.predict([x_val, y_val])[0] for x_val in x] for y_val in y])
    
    cp = ax.contourf(X, Y, Z, levels=0,  cmap='coolwarm')

    ax.scatter(example_data_input[:, 0], example_data_input[:, 1], c=example_data_output, cmap='coolwarm')

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    
    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        index = frame % len(best_weights_history)
        local_weights = best_weights_history[index]
        neural_network.set_min_weights(local_weights)
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[neural_network.predict([x_val, y_val])[0] for x_val in x] for y_val in y])

        cp = ax.contourf(X, Y, Z, levels=0,  cmap='coolwarm')
        ax.set_title(f'{frame}/{len(best_weights_history)-1} - Error: {error_history[frame]}')
        ax.scatter(example_data_input[:, 0], example_data_input[:, 1], c=example_data_output, cmap='coolwarm')

        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(best_weights_history), init_func=init, blit=True, interval=10, repeat_delay=1000)

    plt.show()

if __name__ == "__main__":
    main()

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import Perceptron as SklearnPerceptron

class Perceptron:
    def __init__(self, max_iter, dimensions, learning_rate=0.1):
        self.weights = initialize_weights(dimensions)
        self.error = None
        self.min_error = sys.maxsize
        self.max_iter = max_iter
        self.min_weights = None
        self.learning_rate = learning_rate

def initialize_weights(dimensions):
    return np.random.rand(dimensions)

def compute_excitement(value, weight):
    return sum(value * weight)

def compute_activation(value, weights):
    return 1 if compute_excitement(value, weights) >= 0 else -1

def train(perceptron: Perceptron, data_input, expected_output):
    i = 0
    weight_history = []
    error_history = []
    weight_history.append(perceptron.weights)
    error_history.append(perceptron.min_error)
    while perceptron.min_error > 0 and i < perceptron.max_iter:
        mu = np.random.randint(0, len(data_input))
        value = data_input[mu]
        activation = compute_activation(value, perceptron.weights)
        delta_weight = perceptron.learning_rate * (expected_output[mu] - activation) * value
        perceptron.weights = [perceptron.weights[i] + delta_weight[i] for i in range(0, len(perceptron.weights))]

        error = sum([abs(expected_output[mu] - compute_activation(data_input[mu], perceptron.weights)) for mu in range(0, len(data_input))])

        weight_history.append(perceptron.weights)
        error_history.append(error)
        if error < perceptron.min_error:
            perceptron.min_error = error
            perceptron.min_weights = perceptron.weights
         
        i += 1

    print("Iterations: ", i)
    return perceptron.min_weights, weight_history, error_history

def main():
    example_data_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    example_data_input = np.insert(example_data_input, 0, 1, axis=1)
    example_data_output = np.array([-1, -1, -1, 1])
    perceptron = Perceptron(1000, len(example_data_input[0]), 0.01)

    weights, weight_history, error_history = train(perceptron, example_data_input, example_data_output)
    print(weights)

    example_data_input = np.delete(example_data_input, 0, axis=1)
    input_output_data = np.insert(example_data_input, 0, example_data_output, axis=1)

    df = pd.DataFrame(input_output_data, columns=['output', 'x1', 'x2'])

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.scatter(df['x1'], df['x2'], c=df['output'])

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    fps = 30  # frames per second
    delay_seconds = 2
    extra_frames = fps * delay_seconds
    weight_history_extended = weight_history + [weight_history[-1]] * extra_frames
    error_history_extended = error_history + [error_history[-1]] * extra_frames

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        local_weights = weight_history_extended[frame%len(weight_history_extended)]
        a = -local_weights[1] / local_weights[2]
        xx = np.linspace(-1, 1)
        yy = a * xx - (local_weights[0] / local_weights[2])
        ax.set_title('Error: ' + str(error_history_extended[frame%len(error_history_extended)]))
        line.set_data(xx, yy)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(weight_history_extended), init_func=init, blit=True, interval=100, repeat_delay=1000)
    # ani.save('results/result_animation.gif', writer='imagemagick', fps=fps)
    # ani.save('results/result_animation.mp4', writer='ffmpeg', fps=fps)

    plt.figure()
    a = -weights[1] / weights[2]
    xx = np.linspace(-1, 1)
    yy = a * xx - (weights[0] / weights[2])

    # Plot the line along with the data
    plt.plot(xx, yy, 'k-')
    plt.scatter(df['x1'], df['x2'], c=df['output'])
    plt.show()

if __name__ == "__main__":
    main()
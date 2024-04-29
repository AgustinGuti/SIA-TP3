import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import Perceptron as SklearnPerceptron

class Perceptron:
    def __init__(self, max_iter, dimensions, learning_rate=0.1):
        self.weights = np.random.rand(dimensions)
        self.bias = np.random.rand()
        self.error = None
        self.min_error = sys.maxsize
        self.max_iter = max_iter
        self.min_weights = None
        self.min_bias = None
        self.learning_rate = learning_rate

    def compute_excitement(self, value):
        return sum(value * self.weights) + self.bias

    def compute_activation(self, value):
        return 1 if self.compute_excitement(value) >= 0 else -1

    def train(self, data_input, expected_output):
        i = 0
        weight_history = []
        error_history = []
        bias_history = []
        weight_history.append(self.weights)
        error_history.append(self.min_error)
        bias_history.append(self.bias)
        while self.min_error > 0 and i < self.max_iter:
            mu = np.random.randint(0, len(data_input))
            value = data_input[mu]
            activation = self.compute_activation(value)
            base_delta = self.learning_rate * (expected_output[mu] - activation)

            self.bias = self.bias + base_delta
            self.weights = self.weights + base_delta * value

            # error = sum([abs(expected_output[mu] - self.compute_activation(data_input[mu])) for mu in range(0, len(data_input))])
            error = 0.5*sum((expected_output[mu] - self.compute_activation(data_input[mu]))**2 for mu in range(0, len(data_input)))

            weight_history.append(self.weights)
            bias_history.append(self.bias)
            error_history.append(error)
            if error < self.min_error:
                self.min_error = error
                self.min_weights = self.weights
                self.min_bias = self.bias
            i += 1

        print("Iterations: ", i)
        print("Error:", self.min_error)
        return self.min_weights, self.min_bias, weight_history, error_history, bias_history

def main():
    example_data_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    example_data_output = np.array([-1, -1, -1, 1])
    perceptron = Perceptron(1000, len(example_data_input[0]), 0.01)

    weights, bias, weight_history, error_history, bias_history = perceptron.train(example_data_input, example_data_output)
    print("Weights: ", weights, "Bias: ", bias)

    df = pd.DataFrame({
        'x1': example_data_input[:, 0],
        'x2': example_data_input[:, 1],
        'output': example_data_output
    })

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
    bias_history_extended = bias_history + [bias_history[-1]] * extra_frames
    
    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        index = frame % len(weight_history_extended)
        local_weights = weight_history_extended[index]
        a = -local_weights[0] / local_weights[1]
        xx = np.linspace(-1, 1)
        yy = a * xx - (bias_history_extended[index] / local_weights[1])
        ax.set_title('Error: ' + str(error_history_extended[index]))
        line.set_data(xx, yy)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(weight_history_extended), init_func=init, blit=True, interval=100, repeat_delay=1000)
    # ani.save('results/result_animation.gif', writer='imagemagick', fps=fps)
    # ani.save('results/result_animation.mp4', writer='ffmpeg', fps=fps)

    plt.figure()
    a = -weights[0] / weights[1]
    xx = np.linspace(-1, 1)
    yy = a * xx - bias / weights[1]

    # Plot the line along with the data
    plt.plot(xx, yy, 'k-')
    plt.scatter(df['x1'], df['x2'], c=df['output'])
    plt.show()

if __name__ == "__main__":
    main()
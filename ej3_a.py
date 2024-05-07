import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from NeuralNetwork import NeuralNetwork, NeuronParams, calculate_error

def main():
    example_data_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    example_data_output = np.array([[1], [1], [-1], [-1]])

    dimensions = len(example_data_input[0])
    layer_one = [NeuronParams(dimensions, 0.1, 'linear', 0.1, '') for _ in range(1)]
    layer_two = [NeuronParams(len(layer_one), 0.1, 'tan_h', 10, '') for _ in range(2)]
    layer_three = [NeuronParams(len(layer_two), 0.1, 'tan_h', 100, '') for _ in range(1)]

    neurons_params = [layer_one, layer_two, layer_three]

    neural_network = NeuralNetwork(neurons_params)
    train_errors = []
      
    best_weights_history = []
    best_biases_history = []
    error_history = []
    for _ in range(200):
        min_error, iterations, best_weights, best_biases, error = neural_network.train(example_data_input, example_data_output, 1)
        best_weights_history.append(best_weights)
        best_biases_history.append(best_biases)
        error_history.append(error)
        train_errors.append(min_error)

    plt.figure()

    plt.plot(range(len(train_errors)), train_errors, label='Training Error')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Training Error over Epochs')
    plt.savefig(f'results/train_error_3a.png')

    # min_error, iterations, best_weights_history, best_biases_history, error_history = neural_network.train(example_data_input, example_data_output)
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

    # plt.savefig('results/result_3a_3.png')

    fig, ax2 = plt.subplots()
    line, = ax2.plot([], [], lw=2)
    
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)

    fps = 20  # frames per second
    delay_seconds = 2
    extra_frames = fps * delay_seconds
    weight_history_extended = best_weights_history + [best_weights_history[-1]] * extra_frames
    error_history_extended = error_history + [error_history[-1]] * extra_frames
    bias_history_extended = best_biases_history + [best_biases_history[-1]] * extra_frames
    
    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        index = frame % len(weight_history_extended)
        local_weights = weight_history_extended[index]
        local_biases = bias_history_extended[index]
        neural_network.set_min_weights(local_weights, local_biases)
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[neural_network.predict([x_val, y_val])[0] for x_val in x] for y_val in y])
        
        cp = ax2.contourf(X, Y, Z, levels=0,  cmap='coolwarm')
        frame_index = frame if index < len(best_weights_history) else len(best_weights_history)-1
        ax2.set_title(f'{frame_index}/{len(best_weights_history)-1} - Error: {error_history_extended[frame]}')
        ax2.scatter(example_data_input[:, 0], example_data_input[:, 1], c=example_data_output, cmap='coolwarm')

        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(weight_history_extended), init_func=init, blit=False, interval=100, repeat_delay=1000)
    # ani.save('results/result_3a_3_animation.gif', writer='imagemagick', fps=fps)


    plt.show()

if __name__ == "__main__":
    main()

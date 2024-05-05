import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from NeuralNetwork import NeuralNetwork, NeuronParams

def main():
    with open('TP3-ej3-digitos.txt', 'r') as f:
        lines = [line.split() for line in f]
    data = [list(map(int, [num for sublist in lines[i:i+7] for num in sublist])) for i in range(0, len(lines), 7)]
        
    all_output_data = np.array([[i%2] for i in range(10)])

    cutoff_percentage = 0.8
    cutoff = int(len(data) * cutoff_percentage)

    input_data = data[:cutoff]
    output_data = all_output_data[:cutoff]

    dimensions = len(input_data[0])
    layer_one = [NeuronParams(dimensions, 0.01, 'linear', 0.2, optimizer='adam') for _ in range(100)]
    layer_two = [NeuronParams(len(layer_one), 0.01,  'linear', 0.5, optimizer='adam') for _ in range(50)]
    layer_three = [NeuronParams(len(layer_two), 0.01, 'sigmoid', 100, optimizer='adam') for _ in range(1)]
    

    neurons_params = [layer_one, layer_two, layer_three]


    cutoff_correct = []
    training_cutoff_correct = []

    for cutoff in range(1, 10):
        cutoff_percentage = cutoff/10
        cutoff = int(len(data) * cutoff_percentage)
        input_data = data[:cutoff]
        output_data = all_output_data[:cutoff]
        correct_history = []
        training_correct = []

        for i in range(10):
            indices = np.arange(len(input_data))  # Create an array of indices
            np.random.shuffle(indices)  # Shuffle the indices

            input_data = [input_data[i] for i in indices]  # Use the shuffled indices to shuffle input_data and output_data
            output_data = [output_data[i] for i in indices]

            neural_network = NeuralNetwork(neurons_params, 1000)
            min_error, iterations, best_weights_history, best_biases_history, ersror_history = neural_network.train(input_data, output_data)
            # print(f'Min error: {min_error} - Iterations: {iterations}')
            # neural_network.dump_weights_to_file('results/weights.json')

            # neural_network.load_weights_from_file('results/weights.json')
            # neural_network.print_weights()

            results = []
            # print('Training')
            correct = 0
            for i, val in enumerate(input_data):
                result = neural_network.predict(val)
                result = [round(res, 2) for res in result]
                # print(f'Result {i}: {result} - expected: {output_data[i]}')
                if round(result[0]) == output_data[i]:
                    correct += 1
                results.append(result)

            # print(f'Training correct: {correct}/{len(input_data)}')

            training_correct.append(correct/len(input_data))

            test_data = data[cutoff:]
            expected_output = all_output_data[cutoff:]

            # print('Testing')
            correct = 0
            for i, val in enumerate(test_data):
                result = neural_network.predict(val)
                # print(f'Result {i}: {round(result[0])} - expected: {expected_output[i]}')
                if round(result[0]) == expected_output[i]:
                    correct += 1
            
            # print(f'Testing correct: {correct}/{len(test_data)}')
            correct_history.append(correct/len(test_data))

        cutoff_correct.append(correct_history)
        training_cutoff_correct.append(training_correct)

    #  I want to plot the mean and std for each cutoff

    cutoff_correct = np.array(cutoff_correct)
    mean = np.mean(cutoff_correct, axis=1)
    std = np.std(cutoff_correct, axis=1)

    training_cutoff_correct = np.array(training_cutoff_correct)
    training_mean = np.mean(training_cutoff_correct, axis=1)
    training_std = np.std(training_cutoff_correct, axis=1)

    fig, ax = plt.subplots()
    ax.errorbar(range(1, 10), mean, fmt='o', label='Testing')
    # ax.errorbar(range(1, 10), training_mean,  fmt='o', label='Training')
    plt.legend()
    ax.set_xlabel('Cutoff percentage')
    ax.set_ylabel('Accuracy')


    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()

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
    layer_two = [NeuronParams(len(layer_one), 0.01,  'linear', 0.2, optimizer='adam') for _ in range(50)]
    layer_three = [NeuronParams(len(layer_two), 0.01, 'sigmoid', 100, optimizer='adam') for _ in range(1)]
    

    neurons_params = [layer_one, layer_two, layer_three]

    neural_network = NeuralNetwork(neurons_params, 1)

    min_error, iterations, best_weights_history, error_history = neural_network.train(input_data, output_data)
    # print(f'Min error: {min_error} - Iterations: {iterations}')
    neural_network.dump_weights_to_file('results/weights.json')

    neural_network.load_weights_from_file('results/weights.json')
    # neural_network.print_weights()

    results = []
    print('Training')
    correct = 0
    for i, val in enumerate(input_data):
        result = neural_network.predict(val)
        print('Result:', result)
        result = [round(res, 2) for res in result]
        print(f'Result {i}: {result} - expected: {output_data[i]}')
        if round(result[0]) == output_data[i]:
            correct += 1
        results.append(result)

    print(f'Training correct: {correct}/{len(input_data)}')

    test_data = data[cutoff:]
    expected_output = all_output_data[cutoff:]

    print('Testing')
    correct = 0
    for i, val in enumerate(test_data):
        result = neural_network.predict(val)
        print(f'Result {i}: {round(result[0])} - expected: {expected_output[i]}')
        if round(result[0]) == expected_output[i]:
            correct += 1
    
    print(f'Testing correct: {correct}/{len(test_data)}')


if __name__ == "__main__":
    main()

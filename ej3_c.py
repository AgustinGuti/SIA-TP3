import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import noise
from NeuralNetwork import NeuralNetwork, NeuronParams

def calculate_correct(neural_network, test_data, example_data_output):
    correct = 0
    for _ in range(50):
        for i, val in enumerate(test_data):
            guess = np.argmax(neural_network.predict(val))
            true_class = np.argmax(example_data_output[i])

            if guess == true_class:
                correct += 1

    correct = correct / 50

    return correct

def main():
    with open('TP3-ej3-digitos.txt', 'r') as f:
        lines = [line.split() for line in f]
    data = [list(map(int, [num for sublist in lines[i:i+7] for num in sublist])) for i in range(0, len(lines), 7)]

    example_data_input = data
    # example_data_input = np.insert(example_data_input, 0, 1, axis=0)
    example_data_output = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])

    # example_data_output = np.array([[i] for i in range(10)])
    # example_data_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    # example_data_output = np.array([[1], [1], [-1], [-1]])

    dimensions = len(example_data_input[0])
    layer_one = [NeuronParams(dimensions, 0.1, 'linear', 0.2) for _ in range(100)]
    layer_two = [NeuronParams(len(layer_one), 0.1,  'tan_h', 0.2) for _ in range(50)]
    layer_three = [NeuronParams(len(layer_two), 0.1, 'sigmoid', 100) for _ in range(10)]
    

    neurons_params = [layer_one, layer_two, layer_three]

    neural_network = NeuralNetwork(neurons_params, 20000)

    min_error, iterations, best_weights_history, error_history = neural_network.train(example_data_input, example_data_output)
    print(f'Min error: {min_error} - Iterations: {iterations}')
    neural_network.dump_weights_to_file('results/weights.json')

    neural_network.load_weights_from_file('results/weights.json')
    # neural_network.print_weights()

    results = []
    for i, input_data in enumerate(example_data_input):
        result = neural_network.predict(input_data, True)
        result = [round(res, 2) for res in result]
        print(f'Result {i}: {result} - expected: {example_data_output[i]}')
        results.append(result)

    # intensity = 0.3

    # test_data = [noise.gaussian_noise(example_data_input[i], intensity) for i in range(len(example_data_input))]

    # for i, input_data in enumerate(test_data):
    #     noise.print_number(i, input_data)

    accuracies = []

    intensities = range(0, 50)
    intensities = [i/100 for i in intensities]

    for intensity in intensities:
        test_data = [noise.gaussian_noise(example_data_input[i], intensity) for i in range(len(example_data_input))]

        correct = calculate_correct(neural_network, test_data, example_data_output)
        accuracies.append(correct/len(example_data_input))

    plt.figure()

    plt.plot(intensities, accuracies)
    plt.xlabel('Intensity')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs Intensity - Gaussian Noise')

    # accuracies = []
    # intensities = range(1, 50)
    # intensities = [i/100 for i in intensities]

    # for intensity in intensities:
    #     test_data = [noise.random_noise(example_data_input[i], intensity) for i in range(len(example_data_input))]

    #     accuracy = 0
    #     for _ in range(100):
    #         correct = 0
    #         for i, val in enumerate(test_data):
    #             guess = np.argmax(neural_network.predict(val))
    #             if example_data_output[i][guess] == 1:
    #                 correct += 1

    #         accuracy += float(correct)/len(example_data_input)
        
    #     accuracy = accuracy / 100
    #     accuracies.append(accuracy)

    # plt.figure()
    # plt.plot(intensities, accuracies)
    # plt.xlabel('Intensity')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs Intensity - Random Noise')

    # accuracies = []
    # intensities = range(1, 50)
    # intensities = [i/100 for i in intensities]

    # for intensity in intensities:
    #     test_data = [noise.salt_and_pepper_noise(example_data_input[i], intensity) for i in range(len(example_data_input))]

    #     accuracy = 0
    #     for _ in range(100):
    #         correct = 0
    #         for i, val in enumerate(test_data):
    #             guess = np.argmax(neural_network.predict(val))
    #             if example_data_output[i][guess] == 1:
    #                 correct += 1

    #         accuracy += float(correct)/len(example_data_input)
        
    #     accuracy = accuracy / 100
    #     accuracies.append(accuracy)

    # plt.figure()
    # plt.plot(intensities, accuracies)
    # plt.xlabel('Intensity')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs Intensity - Salt and Pepper Noise')
      
   
    # fig, axs = plt.subplots(len(results), figsize=(10, 30))

    #   # For each input, create a bar chart of the result and expected result
    # for i, (result, expected) in enumerate(zip(results, example_data_output)):
    #     axs[i].bar(np.arange(len(result)), result, alpha=0.7, label='Result')
    #     axs[i].bar(np.arange(len(expected)), expected, alpha=0.7, label='Expected')
    #     axs[i].set_title(f'Input {i+1}')
    #     axs[i].legend()

    plt.show()

if __name__ == "__main__":
    main()

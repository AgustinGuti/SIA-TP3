import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from NeuralNetwork import NeuralNetwork, NeuronParams, calculate_error

def main():
    with open('TP3-ej3-digitos.txt', 'r') as f:
        lines = [line.split() for line in f]
    data = [list(map(int, [num for sublist in lines[i:i+7] for num in sublist])) for i in range(0, len(lines), 7)]
        
    all_output_data = np.array([[i%2] for i in range(10)])

    cutoff_percentage = 0.80
    cutoff = int(len(data) * cutoff_percentage)

    input_data = data[:cutoff]
    output_data = all_output_data[:cutoff]

    test_data = data[cutoff:]
    expected_output = all_output_data[cutoff:]

    dimensions = len(input_data[0])
    layer_one = [NeuronParams(dimensions, 0.01, 'linear', 0.2, optimizer='') for _ in range(100)]
    layer_two = [NeuronParams(len(layer_one), 0.01,  'linear', 0.5, optimizer='') for _ in range(50)]
    layer_three = [NeuronParams(len(layer_two), 0.01, 'sigmoid', 100, optimizer='') for _ in range(1)]
    

    neurons_params = [layer_one, layer_two, layer_three]

    train_errors = []
    test_errors = []
    iters_without_error = 0
    neural_network = NeuralNetwork(neurons_params)
    for _ in range(1000):
        min_error, iterations, best_weights, best_biases, error = neural_network.train(input_data, output_data, 1)
        if min_error <= 1e-5:
            iters_without_error += 1

        predictions = [neural_network.predict(val) for val in test_data]
        test_error = calculate_error(predictions, expected_output)

        train_errors.append(min_error)
        test_errors.append(test_error)
        if iters_without_error == 10:
            break

    plt.figure()

    plt.plot(range(len(train_errors)), train_errors, label='Training Error')
    plt.plot(range(len(test_errors)), test_errors, label='Test Error')
  
    # Plot the average test error as a horizontal line
    avg_test_error = np.mean(test_errors)
    plt.axhline(avg_test_error, color='r', linestyle=':', label='Average Test Error')

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Error over Epochs')
    plt.savefig(f'results/train_error_3b.png')

    avg_test_errors_list = []
    for _ in range(10):
        avg_test_errors = []

        for cutoff in range(1, 10):
            cutoff_percentage = cutoff/10
            cutoff = int(len(data) * cutoff_percentage)
            input_data = data[:cutoff]
            output_data = all_output_data[:cutoff]     

            test_errors = []
            iters_without_error = 0
            neural_network = NeuralNetwork(neurons_params)
            for _ in range(1000):
                min_error, iterations, best_weights, best_biases, error = neural_network.train(input_data, output_data, 1)
                if min_error <= 1e-5:
                    iters_without_error += 1

                predictions = [neural_network.predict(val) for val in test_data]
                test_error = calculate_error(predictions, expected_output)

                test_errors.append(test_error)
                if iters_without_error == 10:
                    break
            
            avg_test_errors.append(np.mean(test_errors))
        avg_test_errors_list.append(avg_test_errors)

    
    plt.figure()

    mean = np.mean(avg_test_errors_list, axis=0)
    std = np.std(avg_test_errors_list, axis=0)

    plt.errorbar(range(1, 10), mean, fmt='o', yerr=std, capsize=6, label='Average Test Error')
    plt.legend()
    plt.xlabel('Cutoff Percentage')
    plt.ylabel('Error')
    plt.title(f'Error over Cutoff Percentage')
    plt.savefig(f'results/cutoff_error_3b.png')



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

            neural_network = NeuralNetwork(neurons_params)
            min_error, iterations, best_weights_history, best_biases_history, ersror_history = neural_network.train(input_data, output_data, 1000)
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


    fig, ax = plt.subplots()
    ax.errorbar(range(1, 10), mean, fmt='o', yerr=std,  label='Testing')

    # plt.legend()
    ax.set_xlabel('Cutoff percentage')
    ax.set_ylabel('Accuracy')
    ax.legend()

    plt.savefig(f'results/accuracy_3b.png')
    plt.show()


if __name__ == "__main__":
    main()

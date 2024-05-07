import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import noise
from NeuralNetwork import NeuralNetwork, NeuronParams, calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score, create_confusion_matrix
import seaborn as sns

def calculate_correct(neural_network, test_data, example_data_output):
    correct = 0
    correct_guesses = [0 for _ in range(10)]
    for i, val in enumerate(test_data):
        guess = np.argmax(neural_network.predict(val))
        true_class = np.argmax(example_data_output[i])

        if guess == true_class:
            correct += 1
            correct_guesses[true_class] += 1

    return correct, correct_guesses


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
    layer_one = [NeuronParams(dimensions, 0.1, 'tan_h', 0.2, optimizer="") for _ in range(100)]
    layer_two = [NeuronParams(len(layer_one), 0.1,  'tan_h', 0.5, optimizer="") for _ in range(50)]
    layer_three = [NeuronParams(len(layer_two), 0.1, 'sigmoid', 100, optimizer="") for _ in range(10)]
    
    

    neurons_params = [layer_one, layer_two, layer_three]

    neural_network = NeuralNetwork(neurons_params, 20000)
    min_error, iterations, best_weights_history , best_biases_history, error_history = neural_network.train(example_data_input, example_data_output)
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

    intensity = 0.4

    test_data = [noise.gaussian_noise(example_data_input[i], intensity) for i in range(len(example_data_input))]

    # for i, input_data in enumerate(test_data):
    #     noise.print_number(i, input_data)


    intensities = range(0, 50)
    intensities = [i/100 for i in intensities]

    correct_guesses_by_number = [0 for _ in range(10)]
    intensity=0.4

    y_true = []
    y_pred = []

    for j in range(1000):
        test_data = [noise.gaussian_noise(example_data_input[i], intensity) for i in range(len(example_data_input))]
        correct, correct_guesses = calculate_correct(neural_network, test_data, example_data_output)
        correct_guesses_by_number = [correct_guesses_by_number[i] + correct_guesses[i] for i in range(10)]
        # for i in range(len(test_data)):
        #     noise.print_number(i, test_data[i])
        #     plt.savefig(f'results/gaussian_noise_{i}.png')
        for i in range(len(test_data)):
            y_true.append(np.argmax(example_data_output[i]))
            y_pred.append(np.argmax(neural_network.predict(test_data[i])))
            
    cm = create_confusion_matrix(y_true, y_pred, 10)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(f'Confusion Matrix - Gaussian Noise - Intensity {intensity}')
    plt.savefig(f'results/confusion_matrix_gaussian_noise_{intensity}.png')

    correct_guesses_percentage_by_number = [correct_guesses_by_number[i]/1000 for i in range(10)]
    plt.figure()
    plt.xlabel('Digit')
    plt.ylabel('Correct Guesses')
    plt.xticks(np.arange(10))
    plt.title(f'Correct Guesses by Digit - Gaussian Noise - Intensity {intensity}')
    plt.bar(np.arange(10), correct_guesses_percentage_by_number)
    plt.savefig(f'results/correct_guesses_gaussian_noise_{intensity}.png')

   

    # print(f'Correct: {correct}')
    # for j in range(10):
    #     print(f'Number - {j}: {correct_guesses[j]}')
    succes_rate = []

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for intensity in intensities:
        acum= []
        y_true = []
        y_pred = []
        for _ in range(10):
            test_data = [noise.gaussian_noise(example_data_input[i], intensity) for i in range(len(example_data_input))]
            correct, correct_guesses = calculate_correct(neural_network, test_data, example_data_output)
            acum.append(correct/len(example_data_input))
            for i in range(len(test_data)):
                y_true.append(np.argmax(example_data_output[i]))
                y_pred.append(np.argmax(neural_network.predict(test_data[i])))
        succes_rate.append(acum)

        cm = create_confusion_matrix(y_true, y_pred, 10)

        precisions.append(calculate_precision(cm))
        accuracies.append(calculate_accuracy(cm))
        recalls.append(calculate_recall(cm))
        f1_scores.append(calculate_f1_score(cm))
    
    avgs = np.mean(succes_rate, axis=1)
    stds = np.std(succes_rate, axis=1)

    plt.figure()
    plt.errorbar(intensities, avgs, yerr=stds, fmt='o')
    plt.xlabel('Intensity')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs Intensity - Gaussian Noise')
    plt.savefig(f'results/success_rate_gaussian_noise.png')

    plt.figure()
    plt.errorbar(intensities, precisions, fmt='o')
    plt.xlabel('Intensity')
    plt.ylabel('Precision')
    plt.title('Precision vs Intensity - Gaussian Noise')
    plt.savefig(f'results/precision_gaussian_noise.png')

    plt.figure()
    plt.errorbar(intensities, accuracies, fmt='o')
    plt.xlabel('Intensity')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Intensity - Gaussian Noise')
    plt.savefig(f'results/accuracy_gaussian_noise.png')

    plt.figure()
    plt.errorbar(intensities, recalls, fmt='o')
    plt.xlabel('Intensity')
    plt.ylabel('Recall')
    plt.title('Recall vs Intensity - Gaussian Noise')
    plt.savefig(f'results/recall_gaussian_noise.png')

    plt.figure()
    plt.errorbar(intensities, f1_scores, fmt='o')
    plt.xlabel('Intensity')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Intensity - Gaussian Noise')
    plt.savefig(f'results/f1_score_gaussian_noise.png')

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

__author__ = "Deniz Yaradanakul"

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
import time
# HYPER PARAMETERS
input_size = 784
output_size = 10
number_of_neurons_in_hidden_layer = 533
# 'He Initialization' for weights and initial 0.01 values for all biases
W_B = {
    'W1': np.random.randn(number_of_neurons_in_hidden_layer, 784) * np.sqrt(2/number_of_neurons_in_hidden_layer),
    'b1': np.ones((number_of_neurons_in_hidden_layer, 1)) * 0.01,
    'W2': np.random.randn(10, number_of_neurons_in_hidden_layer) * np.sqrt(2/number_of_neurons_in_hidden_layer),
    'b2': np.ones((10, 1)) * 0.01
}
learning_rate = 0.01
number_of_epochs = 1
path = "./mnist"  # please use relative path like this


def activation_function(layer):
    # the activation function is ReLU (if x <= 0 x=0, else x = x)
    return np.maximum(0.0, layer)


def derivation_of_activation_function(signal):
    # derivative of ReLU (if x<=0 x = 0, else x=1)
    return 1. * (signal > 0)


def loss_function(true_labels, probabilities):
    # cross entropy loss for a data
    x = probabilities['Y_pred']
    x = np.array(x)
    true_labels = np.array(true_labels, dtype=int)
    logp = - np.log(x[np.arange(1), true_labels.argmax(axis=1)])
    loss = np.sum(logp)
    return loss


def softmax(layer):
    # softmax is used to turn activations into probability distribution
    e_x = np.exp(layer - np.max(layer))
    return e_x / e_x.sum(axis=1)


def derivation_of_loss_function(true_labels, probabilities):
    # (this function calculates the result of derivative of softmax(z)) x (derivation of loss function)
    res = probabilities - true_labels
    return res


def forward_pass(data):
    # Calculate the Z for the hidden layer
    z1 = np.dot(data, W_B['W1'].T) + W_B['b1'].T
    # Calculate the activation output for the hidden layer
    a = activation_function(z1)
    # Calculate the Z for the output layer
    z2 = np.dot(a, W_B['W2'].T) + W_B['b2'].T

    # Calculate the activation output for the output layer
    y_pred = softmax(z2)
    # Save hidden layer output and z's in a dictionary
    forward_results = {"Z1": z1,
                       "A": a,
                       "Z2": z2,
                       "Y_pred": y_pred}
    return forward_results


def backward_pass(input_layer, output_layer, loss):
    # calculate layer deltas
    output_delta = loss
    z2_delta = np.dot(output_delta, W_B['W2'])
    a_delta = z2_delta * derivation_of_activation_function(output_layer['A'])
    # update weights and biases
    W_B['W2'] -= learning_rate * np.dot(output_delta.T, output_layer['A'])
    W_B['b2'] -= learning_rate * np.sum(output_delta, axis=1, keepdims=True)
    W_B['W1'] -= learning_rate * np.dot(a_delta.T, input_layer)
    W_B['b1'] -= learning_rate * np.sum(a_delta, axis=1)


def train(train_data, train_labels, valid_data, valid_labels):
    # array initializations for accuracy and loss plot
    accuracy_list = np.array([])
    loss_list = np.array([])

    for epoch in range(number_of_epochs):
        index = 0

        for data, labels in zip(train_data, train_labels):
            data = np.array([data])
            data = data/np.abs(data).max(axis=1)  # to normalize data values between (0,1)
            labels = np.array([labels])

            predictions = forward_pass(data)
            loss_signals = derivation_of_loss_function(labels, predictions['Y_pred'])
            backward_pass(data, predictions, loss_signals)
            loss_function(labels, predictions)
            if index % 2000 == 0:  # at each 2000th sample, we run validation set to see our model's improvements
                accuracy_of_valid, loss_of_valid = test(valid_data, valid_labels)
                loss_list = np.append(loss_list, [loss_of_valid])
                accuracy_list = np.append(accuracy_list, [accuracy_of_valid])
                print("Epoch= " + str(epoch) + ", Coverage= %" + str(100 * (index / len(train_data))) +
                      ", Accuracy= " + str(accuracy_of_valid) + ", Loss= " + str(loss_of_valid))

            index += 1

    return accuracy_list, loss_list


def test(test_data, test_labels):
    avg_loss = 0
    predictions = []
    labels = []

    for data, label in zip(test_data, test_labels):  # Turns through all data

        data = np.array([data])
        data = data/np.abs(data).max(axis=1)  # to normalize data values between (0,1)
        label = np.array([label])

        prediction = forward_pass(data)
        predictions.append(prediction['Y_pred'])
        labels.append(label)
        avg_loss += np.sum(loss_function(label, prediction))

    # Maximum likelihood is used to determine which label is predicted, highest prob. is the prediction
    # And turn predictions into one-hot encoded

    one_hot_predictions = np.zeros(shape=(len(predictions), output_size))
    for k in range(len(predictions)):
        one_hot_predictions[k][np.argmax(predictions[k])] = 1

    predictions = one_hot_predictions

    accuracy_score = accuracy(labels, predictions)

    return accuracy_score, avg_loss / len(test_data)


def accuracy(true_labels, predictions):
    true_pred = 0

    for j in range(len(predictions)):
        if np.argmax(predictions[j]) == np.argmax(true_labels[j]):  # if 1 is in same index with ground truth
            true_pred += 1

    return true_pred / len(predictions)


def plot_loss_accuracy(accuracy_list, loss_list):
    # to plot loss and accuracy graphs, it works when epoch is equal to 1.
    x1 = np.linspace(0.0, 46.0, 24)
    x2 = np.linspace(0.0, 46.0, 24)

    plt.subplot(2, 1, 1)
    plt.plot(x1, accuracy_list*100, 'ko--')
    plt.title('Change of Accuracy and Loss')
    plt.ylabel('% Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(x2, loss_list, 'ko--')
    plt.xlabel('number of data trained(x 1000)')
    plt.ylabel('Loss')

    plt.show()


if __name__ == "__main__":

    mndata = MNIST(path)
    train_x, train_y = mndata.load_training()

    test_x, test_y = mndata.load_testing()

    # creating one-hot vector notation of labels. (Labels are given numeric in MNIST)
    new_train_y = np.zeros(shape=(len(train_y), output_size))
    new_test_y = np.zeros(shape=(len(test_y), output_size))

    for i in range(len(train_y)):
        new_train_y[i][train_y[i]] = 1

    for i in range(len(test_y)):
        new_test_y[i][test_y[i]] = 1

    train_y = new_train_y
    test_y = new_test_y

    # Training and validation split. (%80-%20)
    valid_x = np.asarray(train_x[int(0.8 * len(train_x)):-1])
    valid_y = np.asarray(train_y[int(0.8 * len(train_y)):-1])
    train_x = np.asarray(train_x[0:int(0.8 * len(train_x))])
    train_y = np.asarray(train_y[0:int(0.8 * len(train_y))])
    start_time = time.time()
    accuracies, losses = train(train_x, train_y, valid_x, valid_y)
    print("seconds: ",time.time()-start_time)
    print("Test Scores:")
    print(test(test_x, test_y))

    #plot_loss_accuracy(accuracies, losses)

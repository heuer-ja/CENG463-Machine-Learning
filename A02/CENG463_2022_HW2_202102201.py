# Student Name: ???
# Student ID: ???

import numpy as np
import pandas as pd
import random

# HYPERPARAMETERS
input_size = 
output_size = 
[hidden_layers_sizes] = 
learning_rate = 
number_of_epochs = 
train_data_path = "./data/drugLibTrain_raw.tsv" # please use relative path like this
test_data_path = "./data/drugLibTest_raw.tsv" # please use relative path like this

def activation_function(layer):


def derivation_of_activation_function(signal):


def loss_function(true_labels, probabilities):


def rss(layer):
# sum-of-squares error (rss) is used to turn activations into probability distribution

def derivation_of_loss_function(true_labels, probabilities):
# the derivation should be with respect to the output neurons

def forward_pass(data):

# [hidden_layers] is not an argument, so replace it with your desired hidden layers
def backward_pass(input_layer, [hidden_layers], output_layer, loss): 


def train(train_data, train_labels, valid_data, valid_labels):

	for epoch in range(number_of_epochs):
		index = 0

		# Same thing about [hidden_layers] mentioned above is valid here also
		for data, labels in zip(train_data, train_labels):
			predictions, [hidden_layers] = forward_pass(data)
			loss_signals = derivation_of_loss_function(labels, predictions)
			backward_pass(data, [hidden_layers], predictions, loss_signals)
			loss = loss_function(labels, predictions)

			if index % 400 == 0: # at each 400th sample, we run validation set to see our model's improvements
				accuracy, loss = test(valid_data, valid_labels)
				print("Epoch= "+str(epoch)+", Coverage= %"+ str(100*(index/len(train_data))) + ", Accuracy= "+ str(accuracy) + ", Loss= " + str(loss))

			index += 1

	return losses


def test(test_data, test_labels):

	avg_loss = 0
	predictions = []
	labels = []

	for data, label in zip(test_data, test_labels): # Turns through all data
		prediction, _, _ = forward_pass(data)
		predictions.append(prediction)
		labels.append(label)
		avg_loss += np.sum(loss_function(label, prediction))

	# Maximum likelihood is used to determine which label is predicted, highest prob. is the prediction
	# And turn predictions into one-hot encoded 

	one_hot_predictions = np.zeros(shape=(len(predictions), output_size))
	for i in range(len(predictions)):
		one_hot_predictions[i][np.argmax(predictions[i])] = 1

	predictions = one_hot_predictions

	accuracy_score = accuracy(labels, predictions)

	return accuracy_score,  avg_loss / len(test_data)


def accuracy(true_labels, predictions):
	true_pred = 0

	for i in range(len(predictions)):
		if np.argmax(predictions[i]) == np.argmax(true_labels[i]): # if 1 is in same index with ground truth
			true_pred += 1

	return true_pred / len(predictions)


if __name__ == "__main__":

	train_data = pd.read_csv(train_data_path, sep='\t')
	test_data = pd.read_csv(test_data_path, sep='\t')
	train_x = [input_features] # use train_data['commentsReview'] or concatenate benefitsReview, sideEffectsReview, and commentsReview
	train_y = train_data['rating']
	test_x = [input_features] # use test_data['commentsReview'] or concatenate benefitsReview, sideEffectsReview, and commentsReview
	test_y = test_data['rating']

	# creating one-hot vector notation of labels. (Labels are given numeric in the dataset)
	new_train_y = np.zeros(shape=(len(train_y), output_size))
	new_test_y = np.zeros(shape=(len(test_y), output_size))

	for i in range(len(train_y)):
		new_train_y[i][train_y[i]] = 1

	for i in range(len(test_y)):
		new_test_y[i][test_y[i]] = 1

	train_y = new_train_y
	test_y = new_test_y

	# Training and validation split. (%75-%25)
	valid_x = np.asarray(train_x[int(0.75*len(train_x)):-1])
	valid_y = np.asarray(train_y[int(0.75*len(train_y)):-1])
	train_x = np.asarray(train_x[0:int(0.75*len(train_x))])
	train_y = np.asarray(train_y[0:int(0.75*len(train_y))])

	train(train_x, train_y, valid_x, valid_y)
	print("Test Scores:")
	print(test(test_x, test_y))
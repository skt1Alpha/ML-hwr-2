import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'

# RNN (Recurrent Neural Network) Class Definition
class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1  # Number of RNN layers
        # Define the RNN layer with tanh non-linearity
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        # Linear layer to map the hidden state to the output classes
        self.W = nn.Linear(h, 5)
        # Softmax function to produce probability distributions
        self.softmax = nn.LogSoftmax(dim=1)
        # Loss function (Negative Log-Likelihood Loss for classification)
        self.loss = nn.NLLLoss()

    # Function to compute loss between predicted output and actual label
    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    # Forward pass of the RNN
    def forward(self, inputs):
        # Pass input through the RNN layer
        _, hidden = self.rnn(inputs)  # Obtain the hidden state from the RNN
        # Apply linear transformation on the hidden state to get output scores
        output = self.W(hidden[-1])  # Use the last layer's hidden state
        # Apply softmax to obtain probabilities for each class
        predicted_vector = self.softmax(output)
        return predicted_vector

# Load data from JSON files
def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        # Each entry consists of tokenized text and a sentiment label (adjusted to range 0-4)
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))
    return tra, val

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Define RNN model with input and hidden dimensions
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Use Adam optimizer for training
    # Load pre-trained word embeddings
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0
    last_train_accuracy = 0
    last_validation_accuracy = 0

    while not stopping_condition:
        random.shuffle(train_data)  # Shuffle training data for each epoch
        model.train()  # Set model to training mode
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)
        loss_total = 0
        loss_count = 0

        # Mini-batch processing
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()  # Zero gradients for each batch
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation and split words
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Convert words to embeddings, using 'unk' for unknown words
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

                # Transform the input into the required shape for the RNN
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)

                # Compute loss
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                # Track accuracy
                correct += int(predicted_label == gold_label)
                total += 1
                # Accumulate loss
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            # Average and backpropagate loss
            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print(loss_total / loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        trainning_accuracy = correct / total

        # Validation phase
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct / total

        # Stopping condition to prevent overfitting
        if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
            stopping_condition = True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = trainning_accuracy

        epoch += 1

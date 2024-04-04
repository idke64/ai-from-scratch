import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

def main():
    neural_network = NeuralNetwork(784, 1000, 10)
    correct = 0
    for i in range(len(x_train)):
        neural_network.backpropagation(x_train[i].reshape(1, 784), y_train[i], 0.1)
        if neural_network.feedforward(x_train[i].reshape(1, 784))[1] == y_train[i]:
            correct += 1
        if i % 100 == 0 and i != 0:
            print(f"Epoch {i}")
            print(f"Accuracy: {correct * 100 / i}%")
    

def show_results(neural_network, num_results):
    fig, axs = plt.subplots(num_results // 10, 10, figsize=(10, 10))
    for i in range(num_results):
        ax = axs[i // 10, i % 10]
        ax.imshow(x_test[i], cmap="gray")
        guess = neural_network.feedforward(x_test[i].reshape(1, 784))[1]
        actual = y_test[i]
        ax.text(-15, 0, f"actual: {actual}", color="green", fontsize=5)
        ax.text(-15, 5, f"guess: {guess}", color="green", fontsize=5)
        ax.axis("off")
    plt.show()

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        #initialize weights and biases
        self.input_size = input_size  
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias2 = np.zeros((1, self.output_size))

    def feedforward(self, x):
        hidden = np.dot(x, self.weights1) + self.bias1
        hidden_sigmoid = self.sigmoid(hidden)
        output = np.dot(hidden_sigmoid, self.weights2) + self.bias2
        output_sigmoid = self.sigmoid(output)
        guess = 0
        for i in range(len(output_sigmoid[0])):
            if output_sigmoid[0][i] > output_sigmoid[0][guess]:
                guess = i
        return output_sigmoid, guess
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backpropagation(self, x, y, learning_rate):
        # Forward pass
        output_sigmoid, guess = self.feedforward(x)
        
        # Convert y to one-hot encoded vector
        target = np.zeros((1, self.output_size))
        target[0][y] = 1
        
        # Compute the error at the output
        error_output = output_sigmoid - target
        
        # Compute gradients for W2 and b2
        delta_output = error_output * self.sigmoid_derivative(output_sigmoid)
        hidden_sigmoid, _ = self.feedforward_hidden(x)
        gradients_w2 = np.dot(hidden_sigmoid.T, delta_output)
        gradients_b2 = np.sum(delta_output, axis=0, keepdims=True)
        
        # Backpropagate the error to the hidden layer
        error_hidden = np.dot(delta_output, self.weights2.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(hidden_sigmoid)
        
        # Compute gradients for W1 and b1
        gradients_w1 = np.dot(x.T, delta_hidden)
        gradients_b1 = np.sum(delta_hidden, axis=0)
        
        # Update weights and biases
        self.weights1 -= learning_rate * gradients_w1
        self.bias1 -= learning_rate * gradients_b1
        self.weights2 -= learning_rate * gradients_w2
        self.bias2 -= learning_rate * gradients_b2

    def feedforward_hidden(self, x):
        # Compute the output of the hidden layer
        hidden = np.dot(x, self.weights1) + self.bias1
        hidden_sigmoid = self.sigmoid(hidden)
        return hidden_sigmoid, np.argmax(hidden_sigmoid, axis=1)

        

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    

if __name__ == "__main__":
    main()
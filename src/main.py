import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk

def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    plt.imshow(x_train[0], cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()
    pass

if __name__ == "__main__":
    main()
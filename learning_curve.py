""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


def display_digits():
    digits = load_digits()
    print(digits.DESCR)
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()

def divide_data():
    data = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
    train_size=0.5)
    model = LogisticRegression(C=10**-10)
    model.fit(X_train, y_train)
    print("Train accuracy %f" %model.score(X_train, y_train))
    print("Test accuracy %f"%model.score(X_test, y_test))


def train_model():
    data = load_digits()
    num_trials = 20
    train_percentages = range(5, 95, 5)
    test_accuracies = numpy.zeros(len(train_percentages))
    for n in range(num_trials):
        a = 0
        for i in train_percentages:
            one_test_accuracies = numpy.zeros(len(train_percentages))
            X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
            train_size=(i/100))
            model = LogisticRegression(C=10**-10)
            model.fit(X_train, y_train)
            test_accuracies[a] = model.score(X_train, y_train) + test_accuracies[a]
            a+=1
    test_accuracies = test_accuracies/num_trials
    fig = plt.figure()
    plt.plot(train_percentages, test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    plt.show()


if __name__ == "__main__":
    # Feel free to comment/uncomment as needed
    #divide_data()
    #display_digits()
     train_model()

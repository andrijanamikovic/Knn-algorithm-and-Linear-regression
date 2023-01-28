from math import sqrt, pow
import numpy as np


def calculate_euclidean(sample1, sample2):
    distance = 0.0
    for i in range(len(sample1)-1):
        distance += pow((sample1[i] - sample2[i]), 2)
    return sqrt(distance)


class KnnImpl:

    def __init__(self, k, X_train, y_train):
        self.k = k
        self.x = X_train.to_numpy()
        self.y = y_train.to_numpy()

    def nearest(self, sample):
        distance = []
        for i in range(len(self.x)):
            euclidean = calculate_euclidean(self.x[i], sample)
            distance.append((self.y[i], euclidean))
            distance.sort(key=lambda x: x[1])  # sort in ascending order, based on a distance value
            neighbors = []
        for i in range(self.k):  # get first k samples
            neighbors.append(distance[i][0])
        return neighbors

    def predict(self, data):
        predictions = []
        data = data.to_numpy()
        for sample in data:
            neighbors = self.nearest(sample)
            labels = [sample for sample in neighbors]
            prediction = max(labels, key=labels.count)
            predictions.append(prediction)
        return predictions

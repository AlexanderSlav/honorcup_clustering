import pandas as pd
import numpy as np


class KMeans:
    def __init__(self, num_clusters=4, max_iter=500):
        self.num_clusters = num_clusters
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.num_clusters):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.num_clusters):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


data = pd.read_csv('experiments.csv')
X = np.array([[x, y] for x, y in zip(data['x'], data['y'])])

kmeans = KMeans()
kmeans.fit(X)

answers = []
for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    answers.append(prediction)

answers = np.array(answers)
pd.DataFrame(answers).to_csv("answers.csv", header=None, index=None)

import random
import numpy as np

random.seed(42)


class KMeansClustering:

    def __init__(self, number_of_clusters=5):
        self.number_of_clusters = number_of_clusters
        self.centroids = []

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def predict(self, data):
        predictions = []

        for data_point in data:
            distances = self.euclidean_distance(data_point, np.array(self.centroids))
            predictions.append(np.argmin(distances))

        return predictions

    def fit(self, data, max_iterations=10):
        self.centroids = np.random.uniform(np.amin(data, axis=0), np.amax(data, axis=0),
                                           size=(self.number_of_clusters, data.shape[1]))

        for iteration in range(max_iterations):
            print(iteration)

            # Assign each data point to the nearest centroid
            distances = np.array([self.euclidean_distance(data_point, self.centroids) for data_point in data])
            closest_centroids = np.argmin(distances, axis=1)

            # Update centroids to the mean of assigned points
            new_centroids = np.zeros((self.number_of_clusters, data.shape[1]))
            for i in range(self.number_of_clusters):
                indices = np.where(closest_centroids == i)[0]

                if len(indices) > 0:
                    new_centroids[i] = np.mean(data[indices], axis=0)
                else:
                    new_centroids[i] = self.centroids[i]

            # Check for convergence (if centroids do not change)
            if self.has_converged(new_centroids):
                break

            self.centroids = new_centroids

    def has_converged(self, new_centroids):
        if np.max(self.centroids - np.array(new_centroids)) < 0.0001:
            return True
        return False



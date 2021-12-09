
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

import numpy as np
import scipy as sp
from scipy import spatial as spatial




# Purity scores function
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)  # print(contingency_matrix)

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int) * -1

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:  # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new





data = pd.read_csv("Iris.csv")




labels = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
map_labels = lambda x: labels[x]
true_predictions = data.Species.apply(map_labels)



# # Answer1




pca = PCA(n_components=4)
reduced_data = pd.DataFrame(pca.fit_transform(data.iloc[:, :4]))
evecs = pca.components_
evals = pca.explained_variance_
print("Eigen values of projected data are: ")
plt.plot([1, 2, 3, 4], evals)
plt.xticks([1, 2, 3, 4])
plt.xlabel("No. of components")
plt.ylabel("Eigen Values")
plt.figure()



reduced_data = reduced_data.iloc[:, :2]
reduced_data

# # Answer2




K = 3
kmeans = KMeans(n_clusters=K)
kmeans.fit(reduced_data)
kmeans_prediction = kmeans.predict(reduced_data)

kmean_centers = kmeans.cluster_centers_
print("The center of the cluster are: ")
print(kmean_centers)

# ## A




plt.figure(figsize=(8, 6))
plt.scatter(reduced_data.iloc[:, 0], reduced_data.iloc[:, 1], c=kmeans_prediction.astype(float))
plt.plot(kmean_centers[:, 0], kmean_centers[:, 1], "r.", markersize=20)
plt.figure()
# ## B




print("The distortion measure is: ")
print(kmeans.inertia_)

# ## C




print(f"The purity score is: {purity_score(true_predictions, kmeans_prediction)}")

# # Answer3




Ks = [2, 3, 4, 5, 6, 7]

distortion_measures = []
p_scores = []

for k in Ks:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reduced_data)
    kmeans_prediction = kmeans.predict(reduced_data)
    distortion_measures.append(kmeans.inertia_)
    p_score = purity_score(true_predictions, kmeans_prediction)
    p_scores.append(p_score)

plt.plot(Ks, distortion_measures)
plt.xlabel("Number of clusters")
plt.ylabel("Distortion Measure")
plt.figure()



print(f"The purity scores are: {(p_scores)}")

# ## Answer4




K = 3
gmm = GaussianMixture(n_components=K)
gmm.fit(reduced_data)
GMM_prediction = gmm.predict(reduced_data)

# ## A




plt.figure(figsize=(8, 6))
plt.scatter(reduced_data.iloc[:, 0], reduced_data.iloc[:, 1], c=GMM_prediction.astype(float))
plt.figure()
# ## B




print(f"Total data log likelihood is: {sum(gmm.score_samples(reduced_data))}")

# ## C




print(f"The purity score is: {purity_score(true_predictions, GMM_prediction)}")

# # Answer5




gmm_total_data_logLikelihood = []
gmm_p_scores = []

for k in Ks:
    gmm = GaussianMixture(n_components=k)
    gmm.fit(reduced_data)
    GMM_prediction = gmm.predict(reduced_data)
    gmm_total_data_logLikelihood.append(sum(gmm.score_samples(reduced_data)))
    gmm_p_score = purity_score(true_predictions, GMM_prediction)
    gmm_p_scores.append(gmm_p_score)

plt.plot(Ks, gmm_total_data_logLikelihood)
plt.xlabel("Number of clusters")
plt.ylabel("Total data log likelihood")

plt.figure()


print(f"The purity scores are: {gmm_p_scores}")

# # Answer6

# ## Model1

# ## Model2

configs = [(1, 4), (1, 10), (5, 4), (5, 10)]

for config in configs:
    eps, min_samples = config
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced_data)
    DBSCAN_predictions = dbscan_model.labels_
    plt.figure(figsize=(8, 6))
    p_score = purity_score(true_predictions, DBSCAN_predictions)
    plt.title(f"Epsilon = {eps}, MinSamples = {min_samples}, PurityScore = {p_score}")
    plt.scatter(reduced_data.iloc[:, 0], reduced_data.iloc[:, 1], c=DBSCAN_predictions.astype(float))



plt.show()





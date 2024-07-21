import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

# Passo 1: Gerar um dataset aleatório
n_samples = 15000
n_features = 20
centers = 5
cluster_std = 1.0
random_state = 80
scale = 5

def gerar_arquivo(file_path, num_clusters, num_features, cluster_sizes, X, labels):
  with open(file_path, 'w') as f:
    # Primeira linha: quantidade de clusters e quantidade de features
    f.write(f"{num_clusters} {num_features}\n")

    # Segunda linha: tamanho de cada agrupamento
    f.write(" ".join(map(str, cluster_sizes)) + "\n")

    # Escrevendo os pontos por cluster
    for cluster in range(num_clusters):
        points_in_cluster = X[labels == cluster]
        for point in points_in_cluster:
            f.write(" ".join(map(str, point)) + f"\n")

  print("Arquivo '" + file_path + "' gerado com sucesso.")

X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=random_state)
# X = np.random.rand(n_samples, n_features) * scale

kmeans_k3 = KMeans(n_clusters=3, random_state=random_state)
kmeans_k5 = KMeans(n_clusters=5, random_state=random_state)
kmeans_k7 = KMeans(n_clusters=7, random_state=random_state)

kmeans_k3.fit(X)
kmeans_k5.fit(X)
kmeans_k7.fit(X)

labels_k3 = kmeans_k3.labels_
labels_k5 = kmeans_k5.labels_
labels_k7 = kmeans_k7.labels_

cluster_sizes_k3 = np.bincount(labels_k3)
cluster_sizes_k5 = np.bincount(labels_k5)
cluster_sizes_k7 = np.bincount(labels_k7)


centroids_k3 = kmeans_k3.cluster_centers_
centroids_k5 = kmeans_k5.cluster_centers_
centroids_k7 = kmeans_k7.cluster_centers_


# Create a scatter plot for each KMeans model
plt.figure(figsize=(15, 6))

# KMeans with 3 clusters
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels_k3, cmap='viridis', s=50)
plt.scatter(centroids_k3[:, 0], centroids_k3[:, 1], marker='x', color='red', s=50)
plt.title('KMeans (3 clusters)')

# KMeans with 5 clusters
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_k5, cmap='viridis', s=50)
plt.scatter(centroids_k5[:, 0], centroids_k5[:, 1], marker='x', color='red', s=100)
plt.title('KMeans (5 clusters)')

# KMeans with 7 clusters
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels_k7, cmap='viridis', s=50)
plt.scatter(centroids_k7[:, 0], centroids_k7[:, 1], marker='x', color='red', s=100)
plt.title('KMeans (7 clusters)')

plt.tight_layout()
plt.show()



gerar_arquivo(f"random_k3_f{n_features}_{n_samples}.dat" , 3, n_features, cluster_sizes_k3, X, labels_k3)
gerar_arquivo(f"random_k5_f{n_features}_{n_samples}.dat",  5, n_features, cluster_sizes_k5, X, labels_k5)
gerar_arquivo(f"random_k7_f{n_features}_{n_samples}.dat",  7, n_features, cluster_sizes_k7, X, labels_k7)


# CONFERENCIA
db_index_k3 = davies_bouldin_score(X, labels_k3)
db_index_k5 = davies_bouldin_score(X, labels_k5)
db_index_k7 = davies_bouldin_score(X, labels_k7)

print("\n")
print(f"DBI para a clusterizacao k3 {db_index_k3}")
print(f"DBI para a clusterizacao k5 {db_index_k5}")
print(f"DBI para a clusterizacao k7 {db_index_k7}")



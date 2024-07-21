import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


num_samples = 900000
num_features = 15
num_clusters = 3


X, y = make_blobs(n_samples=num_samples, n_features=num_features, centers=num_clusters, random_state=42)


kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)


labels = kmeans.labels_


cluster_sizes = np.bincount(labels)

db_index = davies_bouldin_score(X, labels)
print(f'Índice de Davies-Bouldin: {db_index}')


file_path = 'cluster_output.txt'
with open(file_path, 'w') as f:

    f.write(f"{num_clusters} {num_features}\n")


    f.write(" ".join(map(str, cluster_sizes)) + "\n")


    for cluster in range(num_clusters):
        points_in_cluster = X[labels == cluster]
        for point in points_in_cluster:
            f.write(" ".join(map(str, point)) + f"\n")

print("Arquivo 'cluster_output.txt' gerado com sucesso.")
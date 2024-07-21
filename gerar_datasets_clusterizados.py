import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# PRIMEIRO PEGUE O DATASET GERADO E FAÇA O CALCULO NA PLANILHA PRA VER SE OS RESULTADOS VAO BATER
# DANDO CERTO, AGORA QUEREMOS VER SE O ALGORITMO FEITO NO TCC VAI BATER COM OS RESULTADOS DESSE DATASET RANDOMICO

# Parâmetros do dataset
num_samples = 900000  # número de amostras
num_features = 15   # número de características (dimensionalidade do espaço)
num_clusters = 3   # número de clusters

# Gerando dataset randomico
X, y = make_blobs(n_samples=num_samples, n_features=num_features, centers=num_clusters, random_state=42)

# Aplicando KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Resultados
labels = kmeans.labels_

# Calculando o tamanho de cada agrupamento
cluster_sizes = np.bincount(labels)

db_index = davies_bouldin_score(X, labels)
print(f'Índice de Davies-Bouldin: {db_index}')

# Escrevendo os resultados em um arquivo
file_path = 'cluster_output.txt'
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

print("Arquivo 'cluster_output.txt' gerado com sucesso.")
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double davies_bouldin_score(double **, int *, int *, int , int, int);
int get_cluster_atual(int, int, int *, int *);
double get_distancia_pontos(double *, double *, int);

int main() {
  const char *path = "../datasets/digits_k10_f64_1797.dat";
  int n_clusters, n_features, dataset_len = 0;
  int *n_size_clusters, *index_inicio_cluster;
  double **dataset;
  size_t memoria_alocada;

  FILE *fp = fopen(path, "r");
  
  // lendo a qtd de clusters e qtd de features
  fscanf(fp, "%d %d", &n_clusters, &n_features);

  // alocando array que guarda o tamanho de cada cluster
  n_size_clusters = (int*) malloc(n_clusters*sizeof(int));
  index_inicio_cluster = (int*) malloc(n_clusters*sizeof(int));

  // lendo a quantidade de pontos em cada cluster
  for(int i = 0; i < n_clusters; i++) {
    fscanf(fp, "%d", &n_size_clusters[i]);
    index_inicio_cluster[i] = dataset_len; // guarda o indice de inicio de cada cluster
    dataset_len += n_size_clusters[i];

  }

  // alocando matriz para guardar o dataset
  dataset = (double**) malloc(dataset_len*sizeof(double*));
  memoria_alocada = sizeof(dataset_len *sizeof(double*));

  for(int i = 0; i < dataset_len; i++) {
    dataset[i] = (double*) malloc(n_features*sizeof(double));
    memoria_alocada += n_features*sizeof(double);
  }

  printf("\n");
  printf("Qtd Clusters : %d    |   Qtd de features : %d\n", n_clusters, n_features);
  printf("Qtd de linhas do dataset : %d\n", dataset_len);
  printf("Memoria Alocada para o dataset: %zu MB\n", memoria_alocada/(1024 * 1024));
  printf("\n");

  // aloca todo o dataset em memoria
  double temp;
  for(int i = 0; i < dataset_len; i++) {
    for(int j = 0; j < n_features; j++) {
       fscanf(fp, "%lf", &temp);
       dataset[i][j] = temp;
       // calcular o centroid enquanto estou lendo o arquivo
    }
  }

  // calcular o davies bouldin
  double db = davies_bouldin_score(dataset, n_size_clusters, index_inicio_cluster, n_clusters, n_features, dataset_len);

  printf("==> DB SCORE : %lf\n", db);

  // libera memoria alocada
  free(n_size_clusters);

  for(int i = 0; i < dataset_len; i++) {
    free(dataset[i]);
  }

  free(dataset);
  return 0;
}

int get_cluster_atual(int linha, int n_clusters, int *arr_cluster_size, int *arr_index_start_cluster) {
  if(linha < arr_cluster_size[0]) {
    return 0;
  }
  int limit = arr_cluster_size[0];
  for (int i = 1; i < n_clusters; i++) {
    limit += arr_cluster_size[i];
    if (linha < (limit + 1)) {
      return i;
    }
  }
  
  return -1;
}

double get_distancia_pontos(double *p1, double *p2, int n_features) {
  double distancia = 0;
  for (int i = 0; i < n_features; i++) {
    distancia += ((p1[i] - p2[i])*(p1[i] - p2[i]));
  }

  return sqrt(distancia);
}

double davies_bouldin_score(double **dataset, int *arr_cluster_size, int *arr_index_start_cluster, int n_clusters, int n_features, int dataset_len) {
  int label;
  double *s = (double*) calloc(n_clusters, sizeof(double));
  double **arr_centroid = (double **) malloc(n_clusters*sizeof(double*));
  for (int i = 0; i < n_clusters; i++) {
    arr_centroid[i] = (double*) calloc(n_features, sizeof(double)); // aloca e inicializa com 0
  }
  
  // ================= Calculando o centroide ==========================

  // soma todos os pontos do dataset organizando a soma por cluster
  for (int i = 0; i < dataset_len; i++) {
    for (int j = 0; j < n_features; j++) {
      label = get_cluster_atual(i, n_clusters, arr_cluster_size, NULL);
      if(label == -1) {
        printf("ERRO: Erro ao calcular a label do elemento");
        return label;
      }
      arr_centroid[label][j] += dataset[i][j];
    }
  }

  // faz a media para obter o valor do centroide
  for (int i = 0; i < n_clusters; i++) {
    for (int j = 0; j < n_features; j++) {
      arr_centroid[i][j] = arr_centroid[i][j]/arr_cluster_size[i];
    }
  }

  // calculando a media de distancia intra-cluster

  for (int x = 0; x < n_clusters; x++) {

    // calculando a media de distancia intra cluster do cluster x

    int index_start = arr_index_start_cluster[x];
    int limit = (index_start + arr_cluster_size[x]);
    double media_distancia = 0;
    
    for (int i = index_start; i < limit; i++) {
      media_distancia += get_distancia_pontos(dataset[i], arr_centroid[x], n_features);
    }
    s[x] = media_distancia/arr_cluster_size[x];
    printf("Media Distancia : %.4lf\n", s[x]);
  }

  double **matriz_db = (double**) malloc(sizeof(double)*n_clusters);
  for (int i = 0; i < n_clusters; i++) {
    matriz_db[i] = (double*) calloc(n_clusters, sizeof(double));
  }

  // calculando db_{ij}
  for (int i = 0; i < n_clusters; i++) {
    for (int j = 0; j < n_clusters; j++) {
      if(i == j) {
        matriz_db[i][j] = -1;
        continue;
      } else {
        double dij = get_distancia_pontos(arr_centroid[i], arr_centroid[j], n_features);
        matriz_db[i][j] = (s[i] + s[j])/dij;
        continue;
      }
    }
  }
  double DB = 0, maior;
  double maior_soma = 0;
  // calculando DB
  for (int i = 0; i < n_clusters; i++) {
    maior = -INFINITY;
    for (int j = 0; j < n_clusters; j++) {
      if(i == j) {
        continue;
      } else {
        if(matriz_db[i][j] > maior) {
          maior = matriz_db[i][j];
        }
      }
    }
    DB += (maior/n_clusters);
  }

  return DB;
}
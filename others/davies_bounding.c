#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_CHAR_BUFFER 1000

struct dataset {
  int  n_clusters;
  int  n_features;
  int  *clusters_len;
  long *addr_cluster_file;
  FILE* file;
};
struct dataset df;

struct s_cluster {
  double *centroid;
  double dispersion_ic;
  double db;
};

// PATHS FILES
char *files[] = {
  "./data/luna_k5_f20_5000.dat",
  "./data/luna_k5_f20_10000.dat",
  "./data/luna_k5_f20_25000.dat"
};

/*  HEADER FUNCTIONS */

int    initialize_dataset(char *);
void   print_infos_dataset(struct dataset *);
double calc_centroid_cluster(int, struct s_cluster *);
double calc_centroid_cluster(int index_cluster, struct s_cluster *);
double average_euclidean_distance(int, struct s_cluster *);
double calc_db_between_clusters(int, int, struct s_cluster *);
double calc_distance_between_points(double *point1, double *point2);

int index_file = 0;

int main() {
  clock_t start, end;
  double cpu_time_used;

  start = clock();

  initialize_dataset(files[index_file]);
  
  if(df.file == NULL) {
    printf("ERROR: could not read the file");
    return 0;
  }
  struct s_cluster clusters[df.n_clusters];
  double DB_ij[df.n_clusters];

  for(int i = 0; i < df.n_clusters; i++) {
    clusters[i].centroid = (double*) malloc(sizeof(double)*df.n_features);
    calc_centroid_cluster(i, &clusters[i]);
    average_euclidean_distance(i, &clusters[i]);  
    DB_ij[i] = 0;  
  }
  


  for(int i = 0; i < df.n_clusters; i++) {
    for(int j = 0; j < df.n_clusters; j++) {
      if(i == j) continue;
      double s_i = clusters[i].dispersion_ic;
      double s_j = clusters[j].dispersion_ic;
      
      double *centroid_i = clusters[i].centroid;
      double *centroid_j = clusters[j].centroid;
      double distance_ij = 0;
      double root = 0;

      for(int x = 0; x < df.n_features; x++) {
        distance_ij += (centroid_i[x] - centroid_j[x])*(centroid_i[x] - centroid_j[x]);
      }

      root = sqrt(distance_ij);

      if(DB_ij[i] < root) {
        DB_ij[i] = root;
      }
    }
  }
  
  
  double DB = 0;

  for(int i = 0; i < df.n_clusters; i++) {
    DB +=  DB_ij[i];
  }

  DB = DB/df.n_clusters;

  fclose(df.file);
  printf("DB = %lf\n", DB);
  printf("Arquivo : %s\n", files[index_file]);

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Tempo de execução: %f segundos\n", cpu_time_used);

  return 0;
}

double calc_centroid_cluster(int index_cluster, struct s_cluster *c) {
  long addr_cluster = df.addr_cluster_file[index_cluster];
  int len_cluster = df.clusters_len[index_cluster];
  double *points = c->centroid;
  double temp_buffer = 0;

  for(int i = 0; i < df.n_features; i++) {
    points[i] = 0;
  }

  fseek(df.file, addr_cluster, SEEK_SET);
  
  for(int i = 0; i < len_cluster; i++) {
    for(int j = 0; j < df.n_features; j++) {
      fscanf(df.file, "%lf", &temp_buffer);
      points[j] += temp_buffer;
    }
  }

  for(int i = 0; i < df.n_features; i++) {
    points[i] = points[i]/len_cluster;
  }
}

double average_euclidean_distance(int index_cluster, struct s_cluster *c) {

  long addr_cluster = df.addr_cluster_file[index_cluster];
  int len_cluster = df.clusters_len[index_cluster];
  double temp_buffer = 0;
  double distance = 0;
  double sum = 0;
  double *centroid = c->centroid;

  fseek(df.file, addr_cluster, SEEK_SET);
  
  // reading file
  for(int i = 0; i < len_cluster; i++) {
    distance = 0;
    for(int j = 0; j < df.n_features; j++) {
      fscanf(df.file, "%lf", &temp_buffer);
      distance += (temp_buffer - centroid[j])*(temp_buffer - centroid[j]);
    }
    sum += sqrt(distance);
  }
  
  c->dispersion_ic = sum/len_cluster;
}

int initialize_dataset(char *path) {
  FILE *fp = fopen(path, "r");
  df.file = fp;

  if(fp == NULL) {
    return -1;
  }

  fscanf(fp, "%d %d", &df.n_clusters, &df.n_features);

  df.clusters_len = (int*) malloc(sizeof(int)*df.n_clusters);
  df.addr_cluster_file = (long*) malloc(sizeof(long)*df.n_clusters);

  for (int i = 0; i < df.n_clusters; i++) {
    fscanf(fp, "%d", &df.clusters_len[i]);
  }
  
  char buffer[MAX_CHAR_BUFFER];
  df.addr_cluster_file[0] = ftell(df.file);

  int current_row = 1; // linha atual
  int current_cluster = 0; // cluster atual
  int limit = df.clusters_len[current_cluster]; // tamanho do cluster

  while( (fgets(buffer, sizeof(buffer), df.file)) != NULL ) {
    
    if(current_row == (limit + 1)) {
      current_cluster++;
      if(current_cluster == df.n_clusters) break;
      limit += df.clusters_len[current_cluster];
      df.addr_cluster_file[current_cluster] = ftell(df.file);
    } 
    current_row++;
  }

  // for(int i = 0; i < df.n_clusters; i++) {
  //   long addr = df.addr_cluster_file[i];
  //   fseek(df.file, addr, SEEK_SET);
  //   fgets(buffer, sizeof(buffer), df.file);
  //   printf("new row: %s\n", buffer);
  // }

  fseek(df.file, df.addr_cluster_file[0], SEEK_SET);
  print_infos_dataset(&df);

  return 1;
}

void print_infos_dataset(struct dataset *data) {
  printf("=============== INFOS DATASET ==================\n");
  printf("Number of Clusters: %d\n", data->n_clusters);
  printf("Number of Features: %d\n", data->n_features);
  printf("Clusters Sizes: ");
  for (int i = 0; i < data->n_clusters; i++) {
    if(i != data->n_clusters - 1) {
      printf("%d, ", data->clusters_len[i]);
    } else {
      printf("%d\n", data->clusters_len[i]);
    }
  }

  printf("Clusters Address: ");
  for (int i = 0; i < data->n_clusters; i++) {
    if(i != data->n_clusters - 1) {
      printf("%ld, ", data->addr_cluster_file[i]);
    } else {
      printf("%ld\n", data->addr_cluster_file[i]);
    }
  }
  printf("================================================\n");
}

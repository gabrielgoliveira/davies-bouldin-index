#include <iostream>
#include <vector>
#include <map>
#include <fstream>

#define DEBUG 1
#define BLOCK_SIZE 128
#define BASE_PATH "/home/gabriel/Desktop/ufg/tcc/dunn-index/"

using namespace std;

char paths_datasets[][100] = {
    "datasets/digits_k10_f64_1797.dat", 
    "datasets/iris_k3_f4_150.dat"
};

char* get_path_dataset(int dataset_id) {
    return paths_datasets[dataset_id];
}

__global__ void centroids() {
    return ;
}

// Função para imprimir uma matriz
void printMatrix(float **matrix, int n_rows, int n_columns) {
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_columns; j++) {
            float val = matrix[i][j];
            cout << val << " ";
        }
        cout << endl;
    }
}

float** malloc_matrix(int n_rows, int n_columns) {
    float** matrix = (float**) malloc(n_rows*sizeof(float*));

    for (int i = 0; i < n_rows; i++) {
        matrix[i] = (float*)malloc(n_columns * sizeof(float));
    }

    return matrix; 
}

void free_matrix(float** matrix, int n_rows, int n_columns) {

    for (int i = 0; i < n_rows; ++i) {
        free(matrix[i]);
    }

    // Liberando memória para o array de ponteiros para linhas
    free(matrix);
}

int main() {

    int n_clusters, n_feat, count = 0;
    vector<int> size_clusters;
    map<int, float**> clusters;
    map<int, float*> d_clusters;

    clock_t start, stop;
    double running_time;

    char *path_dataset = get_path_dataset(0);
    ifstream dataset(path_dataset);
    
    dataset >> n_clusters >> n_feat;

    cout<<"================= INFOS DATASET LIDO ========================\n";
    cout<<"Qtd. clusters: "<<n_clusters<<" Qtd. Features: "<<n_feat<<endl;
    cout<<"=============================================================\n";


    for (int i = 0; i < n_clusters; i++) {
        int size_cluster = 0;
        dataset >> size_cluster;
        size_clusters.push_back(size_cluster);
    }

    
    for (int i = 0; i < size_clusters.size(); i++) { // percorrer o arquivo em relação a cada cluster
        int size_current_cluster = size_clusters[i];
        float** current_cluster = malloc_matrix(size_current_cluster, n_feat);
           
        for (int j = 0; j < size_current_cluster; j++) { // le o cluster
            for(int k = 0; k < n_feat; k++) { // le a linha do arquivo
                float value;
                dataset >> value;
                current_cluster[j][k] = value;
            }
        }

        clusters.insert(pair<int, float**>(i, current_cluster));
    }

    if(DEBUG == 1) {
        count = 0;
        for (map<int, float**>::iterator it = clusters.begin(); it != clusters.end(); ++it) {
            printMatrix(it->second, size_clusters[count], n_feat);
            count++;
        }
    }
    
    start = clock(); // marca o tempo 0 do calculo

    for (int i = 0; i < n_clusters; i++) {
        float* d_cluster;
        int rows_current_cluster = size_clusters[i];
        size_t size = rows_current_cluster * n_feat * sizeof(float);

        // aloca o cluster na GPU
        cudaMalloc(&d_cluster, (rows_current_cluster+1)*n_feat*sizeof(float));
        d_clusters.insert(pair<int, float*>(i, d_cluster));
        
        float **cluster = clusters[i];

        cudaMemcpy(d_cluster, cluster, (rows_current_cluster+1)*n_feat*sizeof(float), cudaMemcpyHostToDevice);
    }



    stop = clock(); // marca o tempo final do calculo

    // Print the time taken
    running_time = (double)(stop - start) / CLOCKS_PER_SEC;
    printf("\nTime taken: %lf milissegundos\n", 1000.0*running_time);

    // libera memoria
    count = 0;
    for (map<int, float**>::iterator it = clusters.begin(); it != clusters.end(); ++it) {
        free_matrix(it->second, size_clusters[count], n_feat);
        count++;
    }

    return 0;    
}
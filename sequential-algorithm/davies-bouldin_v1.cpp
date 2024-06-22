#include <iostream>
#include <vector>
#include <map>
#include <fstream>

#define DEBUG 1
#define BLOCK_SIZE 128
#define BASE_PATH "/home/gabriel/Desktop/ufg/tcc/dunn-index/"
#define NF 64

using namespace std;

char paths_datasets[][100] = {
    "../datasets/digits_k10_f64_1797.dat", 
    "../datasets/iris_k3_f4_150.dat"
};

int get_nblocks(int size_cluster) {
    return (size_cluster + BLOCK_SIZE - 1) / BLOCK_SIZE;
}


char* get_path_dataset(int dataset_id) {
    return paths_datasets[dataset_id];
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


float* get_centroid(float **cluster, int size_cluster, int n_feat) {
    float *centroid = (float*) malloc(n_feat*sizeof(float));
    
    for (int i = 0; i < n_feat; i++) {
        centroid[i] = 0.0;
    }

    for (int i = 0; i < size_cluster; i++) {
        for (int j = 0; j < n_feat; j++) {
            centroid[j] += cluster[i][j];
        }
    }

    for (int i = 0; i < n_feat; i++) {
        centroid[i] = centroid[i] / size_cluster;
        cout << centroid[i] << " ";
    }
    cout << endl;
    return centroid;

}

int main() {

    int n_clusters, n_feat, count = 0;
    vector<int>       size_clusters;
    map<int, float*>  centroids;
    map<int, float**> clusters;

    clock_t start, stop;
    double running_time;

    char *path_dataset = get_path_dataset(0);
    ifstream dataset(path_dataset);

    /*
        ==> STEP 1: LER O ARQUIVO
    */
    
    dataset >> n_clusters >> n_feat; // primeira linha do arquivo

    cout<<"================= INFOS DATASET LIDO ========================\n";
    cout<<"Qtd. clusters: "<<n_clusters<<" Qtd. Features: "<<n_feat<<endl;
    cout<<"=============================================================\n";

    
    for (int i = 0; i < n_clusters; i++) {
        // segunda linha do arquivo (lê o tamanho dos clusters)
        int size_cluster = 0;
        dataset >> size_cluster;
        size_clusters.push_back(size_cluster);
    }

    
    for (int i = 0; i < size_clusters.size(); i++) {
        // percorrer o arquivo em relação a cada cluster
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
            // printMatrix(it->second, size_clusters[count], n_feat);
            count++;
        }
    }


    /*
        ==> STEP 3: Calcular o centroide
    */

    for (int i = 0; i < n_clusters; i++) {
        float **current_cluster = clusters[i];
        int size = size_clusters[i];
        float *centroid = get_centroid(current_cluster, size, n_feat);

        centroids.insert(pair<int, float*>(i, centroid));

        if(DEBUG == 1) {
            cout << "\n ===> Centroid do cluster " << i << " : ";

            for (int j = 0; j < n_feat; j++) {
                cout << centroid[j] << " ";
            }
            cout << endl;
        }
    }

   

    // libera memoria
    count = 0;
    for (map<int, float**>::iterator it = clusters.begin(); it != clusters.end(); ++it) {
        free_matrix(it->second, size_clusters[count], n_feat);
        count++;
    }

    return 0;    
}
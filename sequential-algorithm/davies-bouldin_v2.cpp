#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <cmath>

/*

==> Algoritmo Sequencia 2 :

  O objetivo deste algoritmo é eficiencia, portanto ele aloca o dataset inteiramente em memoria,
porém ele não aloca memoria de forma contigua, ou seja, cada dataset é alocado de forma dinamica em
memoria usando ponteiros e o malloc.

*/

#define DEBUG 0
#define BASE_PATH "/home/gabriel/Desktop/ufg/tcc/dunn-index/"

using namespace std;

char paths_datasets[][100] = {
    "../datasets/digits_k10_f64_1797.dat",              // 0
    "../datasets/iris_k3_f4_150.dat",                   // 1
    "../datasets/electricity_k2_f8_45311.dat",          // 2
    "../datasets/random_k3_f15_100000.txt",             // 3
    "../datasets/random_k3_f15_900000.txt",             // 4
    "../datasets/random_k3_f20_5000.dat",               // 5
    "../datasets/random_k5_f20_5000.dat",               // 6
    "../datasets/random_k7_f20_5000.dat",               // 7
};


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

float calc_distance(float *p1, float *p2, int dim) {
    float sum = 0.0;
    for (int i = 0; i < dim; i++) {
        float x = p1[i];
        float y = p2[i];
        sum += (x-y)*(x-y);
    }

    return sqrt(sum);
}

float get_spread(float **cluster, float *centroid, int size_cluster, int n_feat) {
    float sum = 0.0;
    for (int i = 0; i < size_cluster; i++) {
        float *p1 = cluster[i];
        float distance = calc_distance(p1, centroid, n_feat);

        sum += distance;        
    }

    return sum/size_cluster;
}

int main() {

    int n_clusters, n_feat, count = 0;
    vector<int>       size_clusters;
    vector<float>     spreads;
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
    int dataset_size = 0;
    for (int i = 0; i < n_clusters; i++) {
        // segunda linha do arquivo (lê o tamanho dos clusters)
        int size_cluster = 0;
        dataset >> size_cluster;
        size_clusters.push_back(size_cluster);
        dataset_size += size_cluster;
    }

    cout<<"================= INFOS DATASET LIDO ========================\n";
    cout<<"Qtd. clusters: "<<n_clusters<<" Qtd. Features: "<<n_feat << " Qtd pontos dataset: "<<dataset_size<<endl;
    cout<<"=============================================================\n";

    
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

    // start clock to measure running time
    start = clock();


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

    /*
        ==> STEP 4: Calcular spread
    */

    for (int i = 0; i < n_clusters; i++) {
        float **current_cluster = clusters[i];
        int size = size_clusters[i];
        float *centroid = centroids[i];
        float spread = get_spread(current_cluster, centroid, size, n_feat);
        spreads.push_back(spread);
    }

    if(DEBUG == 1) {
        for (int i = 0; i < spreads.size(); i++) {
            cout << "Spread do cluster " <<i<< " = "<<spreads[i]<<endl;
        }
    }

    /* 
        ==> Step 5: Calculo do DB
    */


    vector<float> DB_ij;
    float db_index = 0.0;

    for (int i = 0; i < n_clusters; i++) {
        float spread_i = spreads[i];
        float *centroid_i = centroids[i];
        float max_dbij = 0.0;

        for (int j = 0; j < n_clusters; j++) {
            if(i == j) continue;
            float spread_j = spreads[j];
            float *centroid_j = centroids[j];
            float dist_centroids = calc_distance(centroid_i, centroid_j, n_feat);
            float db = (spread_i + spread_j)/dist_centroids;
            if (db > max_dbij) max_dbij = db;
        }
        db_index += max_dbij;
    }
    db_index = db_index/n_clusters;

    cout << "DB INDEX : " << db_index << endl;


    stop = clock();
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
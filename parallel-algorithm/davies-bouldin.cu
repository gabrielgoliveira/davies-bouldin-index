#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <cmath>

#define DEBUG 1
#define BLOCK_SIZE 128
#define BASE_PATH "/home/gabriel/Desktop/ufg/tcc/dunn-index/"
#define MAXDATASET_SIZE 2000 
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


void cuda_verifica_erros(cudaError_t error) {
    if(error != cudaSuccess) { 
        printf("CUDA error: %s\n", cudaGetErrorString(error)); 
        exit(-1); 
    }
}

/*
    ====> KERNELS
*/

__global__ void reduce_points(float *d_cluster, float *d_centroid_tmp, int size, int n_feat) {
    //extern __shared__ float s_centroid[];
     __shared__ float s_centroid[BLOCK_SIZE * NF];

    int tid = threadIdx.x;
    int i = (blockIdx.x * blockDim.x) + tid;

    for (int d = 0; d < n_feat; d++) {
        s_centroid[tid * n_feat + d] = 0.0;
    }

    __syncthreads();

    if (i < size) {
        // copia a linha do cluster referente a thread para s_centroid
        for (int d = 0; d < n_feat; d++) {
            s_centroid[tid * n_feat + d] = (float ) d_cluster[i*n_feat + d];
        }
    }
    __syncthreads();

    int p = blockDim.x / 2; // numero de threads dentro do bloco dividido por 2
    while (p != 0) {
        if (tid < p) {
            for (int d = 0; d < n_feat; d++) {
            s_centroid[tid*n_feat+d] = s_centroid[tid*n_feat+d] + s_centroid[(tid+p)*n_feat+d];
        }
        }
        __syncthreads();
        p = p/2;
    }

    // Thread zero of each block moves the local result to the global memory
    if (tid == 0) {
        for (int d = 0; d < n_feat; d++) {
            d_centroid_tmp[blockIdx.x * n_feat + d] = (float )s_centroid[d];
        }
    }

    return ;
}


__global__ void print_row(float *d_cluster, int row, int n_feat) {
    int base = row * n_feat;
    for ( int i = 0; i < n_feat; i++ ) {
        printf("%f ", d_cluster[base]);
        base++;
    }
    // __syncthreads();
    printf("\n");
    return ;

}
int main () {
    int n_clusters, n_feat, count = 0;
    int*       size_clusters;
    int*       start_clusters;

    vector<float>     spreads;
    map<int, float*>  centroids;
    map<int, float**> clusters;
    float dataset[MAXDATASET_SIZE][NF];

    float*  d_start_clusters;         // Enderecos dos clusters alocados na device (gpu)
    float*  d_dataset;                // DEVICE: centroides parciais obtidos por meio de redução

    clock_t start, stop;
    double running_time;

    /*
        ==> STEP 1: LER O ARQUIVO
    */

    char *path_dataset = get_path_dataset(0);
    FILE *fp = fopen(path_dataset, "r");
    // lendo a qtd de clusters e qtd de features
    fscanf(fp, "%d %d", &n_clusters, &n_feat);
    
    int tam_dataset = 0;
    size_clusters   = (int*) malloc(sizeof(int) * n_clusters);
    start_clusters  = (int*) malloc(sizeof(int) * n_clusters);
    for (int i = 0; i < n_clusters; i++) {
        // segunda linha do arquivo (lê o tamanho dos clusters)
        int size_cluster = 0;
        fscanf(fp, "%d", &size_cluster);
        size_clusters[i] = size_cluster;
        tam_dataset += size_cluster;
    }

    cout<<"================= INFOS DATASET LIDO ========================\n";
    cout<<"Qtd. clusters: "<<n_clusters<<" Qtd. Features: "<<n_feat<<" Tam Dataset: "<<tam_dataset<<endl;
    cout<<"=============================================================\n";

    // percorrer o arquivo em relação a cada cluster
    int start_cluster = 0;
    for (int i = 0; i < n_clusters; i++) {
        int size_current_cluster = size_clusters[i];
        float temp;

        start_clusters[i] = start_cluster;
        cout << "Linha onde o cluster se inicia : " << start_clusters[i] << endl;

        for (int j = 0; j < size_current_cluster; j++) { 
            for(int k = 0; k < n_feat; k++) { 
                fscanf(fp, "%f", &temp);
                dataset[start_cluster + j][k] = temp;
            }
        }

        start_cluster +=size_clusters[i];
    }

    if (DEBUG == 1) {
        for (int i = 0; i < n_clusters; i++) {
            int base = start_clusters[i];
            int size = size_clusters[i];
            //cout<<"==================== CLUSTER "<<i<<" =============================="<<endl;
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < n_feat; k++) {
                    //cout<<dataset[base + j][k]<<" ";
                }
                //cout<<endl;

            }
        }
    }

    // ALOCA MEMORIA NA DRAM
    cudaMalloc(&d_start_clusters, n_clusters*sizeof(int));
    cudaMalloc(&d_dataset, MAXDATASET_SIZE*NF*sizeof(float));

    // COPIA DADOS PARA A DRAM
    cudaMemcpy(d_dataset, dataset, MAXDATASET_SIZE*n_feat*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_clusters, start_clusters, n_clusters*sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();



    /*
        ==> STEP 3: Calcular o centroide
    */

    for (int i = 0; i < n_clusters; i++) {
        int base = start_clusters[i];
        for (int j = 0; j < size_clusters[i]; j++) {
    

            int linha = base + j;
            print_row<<<1, 1>>>(d_dataset, linha, n_feat);
            cudaDeviceSynchronize();

        }
    }

   float *d_centroid_tmp;
   for (int i = 0; i < n_clusters; i++) {
    cudaDeviceSynchronize();
    int cluster_size = size_clusters[i];
    int base = start_clusters[i] * n_feat;
    int nblocks = get_nblocks(cluster_size);
    float *d_current_cluster = &d_dataset[base];
    float *h_reduce = (float*) malloc(sizeof(float) * nblocks*n_feat);

    cudaMalloc(&d_centroid_tmp, nblocks*n_feat*sizeof(float));

    reduce_points <<<nblocks, BLOCK_SIZE>>>(
        d_current_cluster, // Ponteiro do cluster no device
        d_centroid_tmp,    // reducao dos pontos em relacao aos blocos
        cluster_size,      // tamanho do cluster
        n_feat             // numero de features
    );

    // verifica erros e sincroniza
    cudaError_t error = cudaGetLastError();
    cuda_verifica_erros(error);

    cudaDeviceSynchronize();
    cudaMemcpy(h_reduce, d_centroid_tmp, nblocks*n_feat*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float *centroid_current_cluster = (float*) malloc(sizeof(float)*n_feat);

    for (int i_f = 0; i_f < n_feat; i_f++) {
        float sum = 0.0;
        for (int j = 0; j < nblocks; j++) {
            int current_index = j * n_feat + i_f;
            sum += h_reduce[current_index];
        }
        centroid_current_cluster[i_f] = sum/cluster_size;
        // printf("%f ", sum);
    }

    centroids.insert(pair<int, float*>(i, centroid_current_cluster));

    if(DEBUG == 1) {
        cout << "\n ===> Centroid do cluster " << i << " : ";

        for (int j = 0; j < n_feat; j++) {
            cout << centroid_current_cluster[j] << " ";
        }
        cout << endl;
    }
   }

    return 0;
}
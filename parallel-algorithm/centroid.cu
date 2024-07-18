#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <cmath>

#define DEBUG 1
#define BLOCK_SIZE 256
#define BASE_PATH "/home/gabriel/Desktop/ufg/tcc/dunn-index/"
#define MAXDATASET_SIZE 45312 
#define NF 12

using namespace std;

char paths_datasets[][100] = {
    "../datasets/digits_k10_f64_1797.dat", 
    "../datasets/iris_k3_f4_150.dat",
    "../datasets/electricity_k2_f8_45311.dat"
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

float calc_distance(float *p1, float *p2, int dim) {
    float sum = 0.0;
    for (int i = 0; i < dim; i++) {
        float x = p1[i];
        float y = p2[i];
        sum += (x-y)*(x-y);
    }

    return sqrt(sum);
}


void cuda_verifica_erros(cudaError_t error) {
    if(error != cudaSuccess) { 
        printf("CUDA error: %s\n", cudaGetErrorString(error)); 
        exit(-1); 
    }
}

void cuda_copy_vector_host_to_device(float* d_matrix, float* h_vector, int size) {
    cudaMemcpy(d_matrix, h_vector, size * sizeof(float), cudaMemcpyHostToDevice);
    return ;
}

float* cuda_malloc_matrix(int n_rows, int n_columns) {
    float* d_matrix;
    cudaMalloc(&d_matrix, n_rows * n_columns * sizeof(float));
    return d_matrix;
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

    if(i > size) return ;
    
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


__global__ void reduce_spread(float *d_cluster, float *d_centroid, float *d_reducers, int size, int n_feat, int n_blocks, int index) {
    extern __shared__ float s_reduce_spread[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Inicializa a memória compartilhada
    s_reduce_spread[tid] = 0.0;

    __syncthreads();

    // Calcula a distância se o índice for válido
    if (idx < size) {
        float sum = 0.0;
        for (int d = 0; d < n_feat; d++) {
            float x = d_cluster[idx * n_feat + d];
            float y = d_centroid[d];
            sum += (x - y) * (x - y);
        }
        s_reduce_spread[tid] = sqrt(sum);
    }

    __syncthreads();

    // Redução paralela na memória compartilhada
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduce_spread[tid] += s_reduce_spread[tid + s];
        }
        __syncthreads();
    }

    // O primeiro thread escreve o resultado final para a memória global
    if (tid == 0) {
        d_reducers[blockIdx.x] = s_reduce_spread[0];
    }
}

__global__ void print_row(float *d_cluster, int row, int n_feat) {
    int base = row * n_feat;
    for ( int i = 0; i < n_feat; i++ ) {
        printf("%f ", d_cluster[base]);
        base++;
    }
    printf("\n");
    return ;

}

int main () {
    int n_clusters, n_feat, count = 0;
    int*       size_clusters;
    int*       start_clusters;
    int        max_cluster_size = 0;

    vector<float>     spreads;
    map<int, float*>  centroids;
    map<int, float**> clusters;
    float dataset[MAXDATASET_SIZE][NF];

    float*  d_start_clusters;         // Enderecos dos clusters alocados na device (gpu)
    float*  d_dataset;                // DEVICE: centroides parciais obtidos por meio de redução
    float*  d_reduce;
    float*  d_centroid_tmp;
    clock_t start, stop;
    double running_time;

    /*
        ==> STEP 1: LER O ARQUIVO
    */

    char *path_dataset = get_path_dataset(2);
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
        if(size_cluster > max_cluster_size) max_cluster_size = size_cluster;
        tam_dataset += size_cluster;
    }

    cout<<"================= INFOS DATASET LIDO ========================\n";
    cout<<"Qtd. clusters: "<<n_clusters<<" Qtd. Features: "<<n_feat<<" Tam Dataset: "<<tam_dataset<<endl;
    cout<<"=============================================================\n";

    // percorrer o arquivo em relação a cada cluster
    int start_cluster = 0;
    const int MAX_BLOCKS = get_nblocks(max_cluster_size);
    for (int i = 0; i < n_clusters; i++) {
        int size_current_cluster = size_clusters[i];
        float temp;

        start_clusters[i] = start_cluster;
        for (int j = 0; j < size_current_cluster; j++) { 
            for(int k = 0; k < n_feat; k++) { 
                fscanf(fp, "%f", &temp);
                dataset[start_cluster + j][k] = temp;
            }
        }

        start_cluster +=size_clusters[i];
    }

    // ALOCA MEMORIA RAM
    float *h_reduce = (float*) malloc(sizeof(float) * MAX_BLOCKS*n_feat);

    // ALOCA MEMORIA NA DRAM
    cudaMalloc(&d_dataset,  MAXDATASET_SIZE*NF*sizeof(float));
    cudaMalloc(&d_reduce,   MAX_BLOCKS*sizeof(float));
    cudaMalloc(&d_centroid_tmp,   MAX_BLOCKS*NF*sizeof(float));
    

    // COPIA DADOS PARA A DRAM
    cudaMemcpy(d_dataset, dataset, MAXDATASET_SIZE*n_feat*sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();



    /*
        ==> STEP 3: Calcular o centroide
    */

   start = clock();

   for (int i = 0; i < n_clusters; i++) {
    int cluster_size = size_clusters[i];
    int base = start_clusters[i] * n_feat;
    int nblocks = get_nblocks(cluster_size);
    float *d_current_cluster = &d_dataset[base];
    
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
    // cudaDeviceSynchronize();

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

    for (int i = 0; i < n_clusters; i++) {        
        int cluster_size = size_clusters[i];
        int base = start_clusters[i] * n_feat;
        int nblocks = get_nblocks(cluster_size);

        float *centroid = centroids[i];
        float *d_current_cluster = &d_dataset[base];

        float *h_reduce = (float*) malloc(sizeof(float)*nblocks);
        float *d_centroid = cuda_malloc_matrix(1, n_feat);
        cuda_copy_vector_host_to_device(d_centroid, centroid, n_feat); 

        reduce_spread <<<nblocks, BLOCK_SIZE>>> (
            d_current_cluster,
            d_centroid,
            d_reduce, 
            cluster_size,
            n_feat,
            nblocks,
            i
        ); 

        cudaError_t error = cudaGetLastError();
        cuda_verifica_erros(error);

        cudaDeviceSynchronize();
        cudaMemcpy(h_reduce, d_reduce, nblocks*sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        float sum = 0.0;
        for (int b = 0; b < nblocks; b++) {
            float dist = h_reduce[b];
            sum += dist;
        }

        float spread = sum/cluster_size;

        spreads.push_back(spread);

    }


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

    return 0;
}
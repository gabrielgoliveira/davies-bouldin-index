#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <cmath>

#define DEBUG 0
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

/* ===> FUNÇÕES DE KERNEL */

// Função de kernel para imprimir a matriz na GPU
__global__ void cuda_print_matrix(float* d_matrix, int n_rows, int n_columns) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n_rows && idy < n_columns && idx == 179) {
        printf("matrix[%d][%d] = %f\n", idx, idy, d_matrix[idx * n_columns + idy]);
    }
}

float* cuda_malloc_matrix(int n_rows, int n_columns) {
    float* d_matrix;
    cudaMalloc(&d_matrix, n_rows * n_columns * sizeof(float));
    return d_matrix;
}

void cuda_copy_matrix_host_to_device(float* d_matrix, float** h_matrix, int n_rows, int n_columns) {
    for (int i = 0; i < n_rows; i++) {  
        cudaMemcpy(d_matrix + i * n_columns, h_matrix[i], n_columns * sizeof(float), cudaMemcpyHostToDevice);
    }

    return ;
}

void cuda_copy_vector_host_to_device(float* d_matrix, float* h_vector, int size) {
    cudaMemcpy(d_matrix, h_vector, size * sizeof(float), cudaMemcpyHostToDevice);
    return ;
}

__global__ void d_reduce_points(float *d_cluster, float *d_centroid_tmp, int size, int n_feat) {
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

__global__ void ___d_reduce_spread(float *d_cluster, float *d_centroid, float *d_reducers, int size, int n_feat, int n_blocks, int index) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x) + tid;
    __shared__ float s_reduce_spread;
    s_reduce_spread = 0.0;
    d_reducers[blockIdx.x] = 0.0;
  
    __syncthreads();
    
    if(idx < size) {
        float sum = 0.0;
        for (int d = 0; d < n_feat; d++) {
            float x = d_cluster[idx*n_feat + d];
            float y = d_centroid[d];
            sum += (x-y)*(x-y);
        }

        s_reduce_spread += sqrt(sum);

        if (blockIdx.x == 0) {
            printf("Distancia tid %d =  %f : total -> %f \n", tid, sqrt(sum), s_reduce_spread); 
        }

        __syncthreads();

        if (tid == 0) {
            // cout << " " << s_reduce_spread << endl;
            d_reducers[blockIdx.x] = s_reduce_spread;
        }

        return ;
    }

    return ;
    
}


__global__ void d_reduce_spread(float *d_cluster, float *d_centroid, float *d_reducers, int size, int n_feat, int n_blocks, int index) {
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

__global__ void cuda_print_vector(float* d_matrix, int n_columns) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_columns) {
        printf("matrix[%d] = %f\n", idx,  d_matrix[idx]);
    }
}


void cuda_verifica_erros(cudaError_t error) {
    if(error != cudaSuccess) { 
        printf("CUDA error: %s\n", cudaGetErrorString(error)); 
        exit(-1); 
    }
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

int main() {

    int n_clusters, n_feat, count = 0;
    vector<int>       size_clusters;
    vector<float>     spreads;
    map<int, float*>  centroids;
    map<int, float**> clusters;
    map<int, float*>  d_clusters;         // Enderecos dos clusters alocados na device (gpu)
    map<int, float*>  d_partial_centroid; // DEVICE: centroides parciais obtidos por meio de redução

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
            cout<<"==================== CLUSTER "<<count<<" =============================="<<endl;
            printMatrix(it->second, size_clusters[count], n_feat);
            count++;
        }
    }
    
    /*
        ==> STEP 2: ALOCA MEMORIA NA GPU E COPIA DADOS PARA A MEMORIA DA GPU
    */

    for (int i = 0; i < n_clusters; i++) {
        float* d_cluster, *d_centroid_temp;
        int size_current_cluster = size_clusters[i];

        // aloca memoria na gpu
        d_cluster = cuda_malloc_matrix(size_current_cluster, n_feat);
        d_clusters.insert(pair<int, float*>(i, d_cluster));
 

        // copia matriz em memoria para a GPU
        float **h_cluster = clusters[i];
        cuda_copy_matrix_host_to_device(d_cluster, h_cluster, size_current_cluster, n_feat);
    }

    if(DEBUG == 1) {
        printf("Memoria alocada na GPU e dados copiados !!\n");
    }

/*
    float *last_cluster = d_clusters[9];
    int size_last_cluster = size_clusters[9];

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((size_last_cluster + threadsPerBlock.x - 1) / threadsPerBlock.x,  (n_feat + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cuda_print_matrix<<<numBlocks,  threadsPerBlock>>>(last_cluster, size_last_cluster, n_feat);
    cudaDeviceSynchronize();
*/

    /*
        ==> STEP 3: Calcular o centroide
    */

    // start clock to measure running time
    start = clock();

    float *d_centroid_tmp;
    for (int i = 0; i < n_clusters; i++) {
        cudaDeviceSynchronize();
        int cluster_size = size_clusters[i];
        int nblocks = get_nblocks(cluster_size);
        float *d_current_cluster = d_clusters[i];
        float *h_reduce = (float*) malloc(sizeof(float)*nblocks*n_feat);
        
        d_centroid_tmp = cuda_malloc_matrix(nblocks, n_feat);
        d_reduce_points <<<nblocks, BLOCK_SIZE>>>(
            d_current_cluster, // Ponteiro do cluster no device
            d_centroid_tmp,    // reducao dos pontos em relacao aos blocos
            cluster_size,      // tamanho do cluster
            n_feat            // numero de features
        );

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

    /*
        ==> STEP 4: Calcular o spread

        Agora para cada cluster vamos ter um numero chamado de spread

        1. colocar os centroids na gpu
        2. 
    */

    cudaDeviceSynchronize();
    for (int i = 0; i < n_clusters; i++) {        
        int cluster_size = size_clusters[i];
        int nblocks = get_nblocks(cluster_size);
        float *centroid = centroids[i];
        float *d_current_cluster = d_clusters[i];

        float *h_reduce = (float*) malloc(sizeof(float)*nblocks);
        float *d_reduce = cuda_malloc_matrix(1, nblocks);
        float *d_centroid = cuda_malloc_matrix(1, n_feat);
        cuda_copy_vector_host_to_device(d_centroid, centroid, n_feat); 


        // cuda_print_vector<<<1, n_feat>>>(d_centroid, n_feat);

        d_reduce_spread <<<nblocks, BLOCK_SIZE>>> (
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

    for (int i = 0; i < spreads.size(); i++) {
        cout << "Spread do cluster " <<i<< " = "<<spreads[i]<<endl;
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


    // libera memoria
    count = 0;
    for (map<int, float**>::iterator it = clusters.begin(); it != clusters.end(); ++it) {
        free_matrix(it->second, size_clusters[count], n_feat);
        count++;
    }

    return 0;    
}